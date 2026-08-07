"""Monta a base de referência de atores.

Para cada nome em `data/actors.py`, baixa algumas fotos da Wikipédia/Wikimedia
Commons, detecta o rosto, calcula o embedding e guarda a média dos embeddings
daquele ator em `data/actors.npz`. A melhor foto de cada um também vira uma
miniatura em `static/actors/` para aparecer no resultado.

Uso:
    python -m scripts.build_actors                     # base completa
    python -m scripts.build_actors --limit 20          # teste rápido
    python -m scripts.build_actors --photos-dir fotos  # usa fotos locais

Modo offline: se você já tem imagens, monte uma pasta assim e use --photos-dir

    fotos/
      Tom Hanks/foto1.jpg
      Tom Hanks/foto2.jpg
      Fernanda Torres/foto1.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, face  # noqa: E402
from data.actors import ACTORS  # noqa: E402
from data.photo_urls import PHOTO_URLS  # noqa: E402

log = logging.getLogger("build_actors")

API = "https://en.wikipedia.org/w/api.php"
# A Wikimedia exige um User-Agent identificável em requisições automatizadas.
UA = "Iaface/1.0 (projeto pessoal de estudo) python-urllib"

IMAGE_EXT = (".jpg", ".jpeg", ".png")
# Arquivos de artigo que nunca são retrato do ator.
JUNK_IN_TITLE = ("logo", "icon", "commons", "wiki", "flag", "signature", "map", "star")
THUMB_SIZE = (320, 320)


def slugify(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Apóstrofo sai sem deixar rastro: "Lupita Nyong'o" vira "lupita-nyongo",
    # e não "lupita-nyong-o".
    plain = plain.replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def _get(url: str, timeout: int = 30, attempts: int = 3) -> bytes:
    """Baixa uma URL, insistindo quando a rede falha.

    A Wikimedia responde 429 quando o build pede rápido demais, e conexão
    doméstica cai sozinha de vez em quando — nos dois casos a mesma URL
    funciona alguns segundos depois.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # 404 e afins não melhoram com insistência; 429 e 5xx melhoram.
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("inalcançável")


def _api(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    return json.loads(_get(f"{API}?{query}"))


def _main_photo(actor: str) -> list[str]:
    """A imagem principal do artigo — quase sempre o melhor retrato."""
    data = _api(
        {
            "action": "query",
            "titles": actor,
            "prop": "pageimages",
            "piprop": "original",
            "redirects": "1",
        }
    )
    urls = []
    for page in data.get("query", {}).get("pages", []):
        source = page.get("original", {}).get("source", "")
        if source.lower().endswith(IMAGE_EXT):
            urls.append(source)
    return urls


def _gallery(actor: str) -> list[str]:
    """Demais imagens do artigo, já em tamanho reduzido."""
    data = _api(
        {
            "action": "query",
            "titles": actor,
            "generator": "images",
            "gimlimit": "30",
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": "600",
            "redirects": "1",
        }
    )
    urls = []
    for page in data.get("query", {}).get("pages", []):
        title = page.get("title", "").lower()
        if not title.endswith(IMAGE_EXT) or any(w in title for w in JUNK_IN_TITLE):
            continue
        for info in page.get("imageinfo", []):
            url = info.get("thumburl") or info.get("url")
            if url:
                urls.append(url)
    return urls


def photo_urls(actor: str, limit: int) -> list[str]:
    """URLs das fotos de um ator, da mais provável para a menos provável."""
    # Link definido na mão ganha da busca: é como se corrige um ator que a
    # Wikipédia não cobre ou para o qual ela devolve a foto errada.
    manual = PHOTO_URLS.get(actor)
    if manual:
        return list(manual)[:limit]

    urls: list[str] = []
    for source in (_main_photo, _gallery):
        if len(urls) >= limit:
            break
        try:
            urls.extend(u for u in source(actor) if u not in urls)
        except Exception as exc:  # rede instável não pode derrubar o build todo
            log.warning("%s: falha em %s (%s)", actor, source.__name__, exc)
    return urls[:limit]


def download(actor: str, urls: list[str]) -> list[Path]:
    folder = config.CACHE_DIR / slugify(actor)
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, url in enumerate(urls):
        ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
        dest = folder / f"{i:02d}{ext if ext in IMAGE_EXT else '.jpg'}"
        if not dest.exists():
            try:
                dest.write_bytes(_get(url))
            except Exception as exc:
                log.warning("download falhou (%s): %s", url, exc)
                continue
        paths.append(dest)
    return paths


def local_photos(photos_dir: Path, actor: str) -> list[Path]:
    for candidate in (photos_dir / actor, photos_dir / slugify(actor)):
        if candidate.is_dir():
            return sorted(p for p in candidate.iterdir() if p.suffix.lower() in IMAGE_EXT)
    return []


def save_thumb(detected: face.DetectedFace, actor: str) -> str:
    """Guarda o recorte do rosto que aparece no card do resultado."""
    config.THUMB_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.THUMB_DIR / f"{slugify(actor)}.jpg"

    x1, y1, x2, y2 = detected.box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 0.85
    img = detected.image.crop(
        (
            max(0, int(cx - half)),
            max(0, int(cy - half)),
            min(detected.image.width, int(cx + half)),
            min(detected.image.height, int(cy + half)),
        )
    )
    img.thumbnail(THUMB_SIZE, Image.LANCZOS)
    img.save(dest, "JPEG", quality=88)
    return dest.name


def build_actor(actor: str, per_actor: int, photos_dir: Path | None) -> dict | None:
    """Vetor médio de um ator. None se nenhuma foto rendeu um rosto."""
    if photos_dir is not None:
        paths = local_photos(photos_dir, actor)[:per_actor]
    else:
        paths = download(actor, photo_urls(actor, per_actor))

    if not paths:
        log.info("  - %s: sem fotos", actor)
        return None

    vectors: list[np.ndarray] = []
    thumb = ""
    for path in paths:
        try:
            detected = face.detect(face.load_image(path.read_bytes()))
            vectors.append(face.embed(detected))
        except face.NoFaceFound:
            continue
        except Exception as exc:
            log.warning("  ! %s (%s): %s", actor, path.name, exc)
            continue

        # A primeira foto aproveitável costuma ser o retrato principal.
        if not thumb:
            try:
                thumb = save_thumb(detected, actor)
            except Exception as exc:
                log.warning("  ! miniatura de %s: %s", actor, exc)

    if not vectors:
        log.info("  - %s: nenhum rosto reconhecido nas fotos", actor)
        return None

    mean = np.mean(vectors, axis=0)
    mean /= np.linalg.norm(mean)
    log.info("  + %s: %d foto(s)", actor, len(vectors))
    return {"name": actor, "vector": mean.astype(np.float32), "thumb": thumb}


def main() -> int:
    parser = argparse.ArgumentParser(description="Monta a base de atores do Iaface")
    parser.add_argument("--limit", type=int, default=0, help="usa apenas os N primeiros atores")
    parser.add_argument("--per-actor", type=int, default=4, help="fotos por ator (padrão: 4)")
    parser.add_argument("--workers", type=int, default=4, help="downloads em paralelo")
    parser.add_argument("--photos-dir", type=Path, help="usa fotos locais em vez da Wikipédia")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="soma à base existente em vez de substituí-la (atualiza quem repetir)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    actors = ACTORS[: args.limit] if args.limit else ACTORS
    log.info("Preparando %d atores...", len(actors))
    face.warmup()

    # Download é espera de rede e o embedding é CPU: baixar tudo em paralelo
    # antes evita que a máquina fique ociosa esperando cada foto.
    if args.photos_dir is None:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pool.map(lambda a: download(a, photo_urls(a, args.per_actor)), actors)

    entries: dict[str, dict] = load_existing() if args.merge else {}
    before = set(entries)

    failed: list[str] = []
    for actor in actors:
        entry = build_actor(actor, args.per_actor, args.photos_dir)
        if entry:
            entries[actor] = entry
        else:
            failed.append(actor)

    if not entries:
        log.error("\nNenhum ator processado — a base não foi gravada.")
        log.error("Confira sua conexão, ou defina links na mão em data/photo_urls.py")
        return 1

    save(list(entries.values()))
    report(len(entries), before, failed)
    return 0


def load_existing() -> dict[str, dict]:
    """Base já gravada, indexada por nome, para o modo --merge."""
    if not config.ACTORS_NPZ.exists():
        return {}
    data = np.load(config.ACTORS_NPZ, allow_pickle=False)
    return {
        str(name): {"name": str(name), "thumb": str(thumb), "vector": vector}
        for name, thumb, vector in zip(data["names"], data["thumbs"], data["vectors"])
    }


def save(entries: list[dict]) -> None:
    config.ACTORS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.ACTORS_NPZ,
        names=np.array([e["name"] for e in entries]),
        thumbs=np.array([e["thumb"] for e in entries]),
        vectors=np.stack([e["vector"] for e in entries]),
    )


def report(total: int, before: set[str], failed: list[str]) -> None:
    """Fecha o build dizendo o que entrou, o que faltou e como consertar."""
    log.info("\n%s", "-" * 60)
    log.info("Base salva em %s", config.ACTORS_NPZ)
    log.info("Atores na base: %d%s", total, f" (antes: {len(before)})" if before else "")

    if not failed:
        log.info("Todos os atores da lista entraram.")
        return

    log.info("\nFicaram de fora (%d):", len(failed))
    for name in failed:
        log.info("  - %s", name)
    log.info(
        "\nMotivo comum: a Wikipédia não tem foto boa da pessoa, ou a rede falhou.\n"
        "Para resolver, abra data/photo_urls.py, coloque links diretos das fotos\n"
        "e rode de novo com --merge (assim o que já funcionou não é refeito)."
    )


if __name__ == "__main__":
    raise SystemExit(main())
