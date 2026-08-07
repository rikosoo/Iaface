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
import threading
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
# A política de bots da Wikimedia exige um User-Agent que identifique o projeto
# E dê um contato. Sem isso, o upload.wikimedia.org responde 429 com "your
# request does not comply with our robot policy" mesmo em volume baixo.
# https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy
UA = "Iaface/1.0 (https://github.com/rikosoo/Iaface) python-urllib"

# Intervalo mínimo entre requisições. A Wikimedia pede no máximo uma por
# segundo para acesso automatizado; ir mais rápido faz o servidor cortar.
MIN_REQUEST_INTERVAL = 1.0

# Largura pedida à API. A Wikimedia pede explicitamente que se use miniatura em
# vez do arquivo original — as miniaturas estão em cache e o original não.
THUMB_WIDTH = 800

IMAGE_EXT = (".jpg", ".jpeg", ".png")
# Arquivos de artigo que nunca são retrato do ator.
JUNK_IN_TITLE = ("logo", "icon", "commons", "wiki", "flag", "signature", "map", "star")
THUMB_SIZE = (320, 320)

_rate_lock = threading.Lock()
_last_request = 0.0


def _wait_turn() -> None:
    """Segura a requisição até completar o intervalo mínimo desde a anterior.

    O lock é global de propósito: com vários workers, o que importa é o ritmo
    somado que o servidor enxerga, não o de cada thread.
    """
    global _last_request
    with _rate_lock:
        espera = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request)
        if espera > 0:
            time.sleep(espera)
        _last_request = time.monotonic()


def slugify(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    # Apóstrofo sai sem deixar rastro: "Lupita Nyong'o" vira "lupita-nyongo",
    # e não "lupita-nyong-o".
    plain = plain.replace("'", "").replace("’", "")
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def _get(url: str, timeout: int = 30, attempts: int = 4) -> bytes:
    """Baixa uma URL respeitando o ritmo pedido pelo servidor.

    Um 429 significa "diminua o passo", então a espera cresce bastante entre as
    tentativas — e quando o servidor manda um Retry-After, é ele que manda.
    """
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(1, attempts + 1):
        _wait_turn()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # 404 e afins não melhoram com insistência; 429 e 5xx melhoram.
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts:
                raise
            espera = _retry_after(exc) or 5 * 3 ** (attempt - 1)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts:
                raise
            espera = 2**attempt

        log.debug("nova tentativa em %.0fs: %s", espera, url)
        time.sleep(espera)
    raise RuntimeError("inalcançável")


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Segundos pedidos pelo servidor no cabeçalho Retry-After, se houver."""
    valor = exc.headers.get("Retry-After") if exc.headers else None
    try:
        # O cabeçalho também aceita data; aí não insistimos em interpretar.
        return min(float(valor), 120.0) if valor else None
    except (TypeError, ValueError):
        return None


def _api(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    return json.loads(_get(f"{API}?{query}"))


def _main_photo(actor: str) -> list[str]:
    """A imagem principal do artigo — quase sempre o melhor retrato.

    Pedimos a miniatura, e não o arquivo original: 800px sobra para reconhecer
    um rosto, e é o formato que a Wikimedia quer que o acesso automatizado use.
    """
    data = _api(
        {
            "action": "query",
            "titles": actor,
            "prop": "pageimages",
            "piprop": "thumbnail",
            "pithumbsize": str(THUMB_WIDTH),
            "redirects": "1",
        }
    )
    urls = []
    for page in data.get("query", {}).get("pages", []):
        source = page.get("thumbnail", {}).get("source", "")
        if source.lower().split("?")[0].endswith(IMAGE_EXT):
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
            "iiurlwidth": str(THUMB_WIDTH),
            "redirects": "1",
        }
    )
    urls = []
    for page in data.get("query", {}).get("pages", []):
        title = page.get("title", "").lower()
        if not title.endswith(IMAGE_EXT) or any(w in title for w in JUNK_IN_TITLE):
            continue
        for info in page.get("imageinfo", []):
            # Só a miniatura: o arquivo original é justamente o que a política
            # de bots da Wikimedia pede para não buscar em massa.
            if url := info.get("thumburl"):
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
    """Fotos de uma pessoa na pasta local, por subpasta ou arquivo solto."""
    for candidate in (photos_dir / actor, photos_dir / slugify(actor)):
        if candidate.is_dir():
            return sorted(p for p in candidate.iterdir() if p.suffix.lower() in IMAGE_EXT)

    # Arquivo solto com o nome da pessoa: fotos/Tom Hanks.jpg
    soltos = sorted(
        p
        for p in photos_dir.glob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXT and p.stem == actor
    )
    return soltos


def local_actors(photos_dir: Path) -> list[str]:
    """Quem está na pasta de fotos, lido dos próprios nomes de arquivo.

    É o que permite ir enchendo a base aos poucos: basta jogar mais uma pasta
    (ou mais um arquivo) ali dentro e rodar de novo com --merge, sem precisar
    editar lista nenhuma no código.
    """
    if not photos_dir.is_dir():
        return []

    nomes = {
        item.name if item.is_dir() else item.stem
        for item in photos_dir.iterdir()
        if not item.name.startswith(".")
        and (item.is_dir() or item.suffix.lower() in IMAGE_EXT)
    }
    return sorted(nomes)


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
    global MIN_REQUEST_INTERVAL

    parser = argparse.ArgumentParser(description="Monta a base de atores do Iaface")
    parser.add_argument("--limit", type=int, default=0, help="usa apenas os N primeiros atores")
    parser.add_argument("--per-actor", type=int, default=4, help="fotos por ator (padrão: 4)")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="downloads em paralelo (o ritmo total continua limitado)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=MIN_REQUEST_INTERVAL,
        help=f"segundos entre requisições (padrão: {MIN_REQUEST_INTERVAL})",
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        nargs="?",
        const=config.PHOTOS_DIR,
        help=f"usa suas fotos em vez da Wikipédia (padrão: {config.PHOTOS_DIR.name}/)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="soma à base existente em vez de substituí-la (atualiza quem repetir)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    MIN_REQUEST_INTERVAL = args.rate

    if args.photos_dir is not None:
        # Na pasta de fotos, quem manda são os nomes que estão lá dentro — é
        # assim que dá para ir enchendo a base aos poucos, sem editar código.
        actors = local_actors(args.photos_dir)
        if not actors:
            log.error("Nenhuma foto em %s/", args.photos_dir)
            log.error(
                "Crie uma pasta por pessoa (%s/Tom Hanks/foto.jpg) ou jogue\n"
                "arquivos soltos com o nome dela (%s/Tom Hanks.jpg).",
                args.photos_dir,
                args.photos_dir,
            )
            return 1
        log.info("Encontrei %d pessoa(s) em %s/", len(actors), args.photos_dir)
    else:
        actors = ACTORS

    if args.limit:
        actors = actors[: args.limit]

    log.info("Preparando %d...", len(actors))
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
        "\nSe apareceu 'HTTP Error 429' no log, foi a Wikimedia pedindo calma:\n"
        "rode de novo com --merge e --rate 2 (o que já baixou fica em cache).\n"
        "\nSe a pessoa simplesmente não tem foto boa no acervo, abra\n"
        "data/photo_urls.py, coloque links diretos das imagens e rode com --merge."
    )


if __name__ == "__main__":
    raise SystemExit(main())
