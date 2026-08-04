"""Monta a base de referência de atores.

Para cada nome em `data/actors.py`, baixa algumas fotos da Wikipédia/Wikimedia
Commons, detecta o rosto, calcula o embedding e guarda a média dos embeddings
daquele ator em `data/actors.npz`. A melhor foto de cada um também é copiada
para `static/actors/` para aparecer no resultado.

Uso:
    python -m scripts.build_actors                # base completa
    python -m scripts.build_actors --limit 20     # teste rápido
    python -m scripts.build_actors --photos-dir fotos   # usa fotos locais

Modo offline: se você já tem imagens, crie uma pasta assim e use --photos-dir

    fotos/
      Tom Hanks/foto1.jpg
      Tom Hanks/foto2.jpg
      Fernanda Torres/foto1.jpg
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import face  # noqa: E402
from data.actors import ACTORS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
OUT_NPZ = ROOT / "data" / "actors.npz"
THUMB_DIR = ROOT / "static" / "actors"

API = "https://en.wikipedia.org/w/api.php"
# A Wikimedia exige um User-Agent identificável em requisições automatizadas.
UA = "Iaface/1.0 (projeto pessoal de estudo; contato via GitHub) python-urllib"

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def slugify(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _api(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    return json.loads(_get(f"{API}?{urllib.parse.urlencode(params)}"))


def photo_urls(actor: str, limit: int) -> list[str]:
    """URLs de fotos do ator, com a imagem principal do artigo em primeiro."""
    urls: list[str] = []

    try:
        data = _api(
            {
                "action": "query",
                "titles": actor,
                "prop": "pageimages",
                "piprop": "original",
                "redirects": "1",
            }
        )
        for page in data.get("query", {}).get("pages", []):
            original = page.get("original", {}).get("source")
            if original and original.lower().endswith(IMAGE_EXT):
                urls.append(original)
    except Exception as exc:  # rede instável não deve derrubar o build inteiro
        print(f"  ! imagem principal de {actor}: {exc}")

    if len(urls) < limit:
        try:
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
            for page in data.get("query", {}).get("pages", []):
                title = page.get("title", "").lower()
                if not title.endswith(IMAGE_EXT):
                    continue
                # Logos, ícones de idioma e afins entopem a lista de imagens.
                if any(w in title for w in ("logo", "icon", "commons", "wiki", "flag", "signature")):
                    continue
                for info in page.get("imageinfo", []):
                    url = info.get("thumburl") or info.get("url")
                    if url and url not in urls:
                        urls.append(url)
        except Exception as exc:
            print(f"  ! galeria de {actor}: {exc}")

    return urls[:limit]


def download(actor: str, urls: list[str]) -> list[Path]:
    folder = CACHE_DIR / slugify(actor)
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, url in enumerate(urls):
        ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
        if ext not in IMAGE_EXT:
            ext = ".jpg"
        dest = folder / f"{i:02d}{ext}"
        if not dest.exists():
            try:
                dest.write_bytes(_get(url))
            except Exception as exc:
                print(f"  ! download {url}: {exc}")
                continue
        paths.append(dest)
    return paths


def local_photos(photos_dir: Path, actor: str) -> list[Path]:
    folder = photos_dir / actor
    if not folder.is_dir():
        folder = photos_dir / slugify(actor)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def save_thumb(src: Path, actor: str) -> str | None:
    """Guarda um recorte quadrado do rosto para mostrar no resultado."""
    from PIL import Image

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    dest = THUMB_DIR / f"{slugify(actor)}.jpg"
    try:
        img = face.load_image(src.read_bytes())
        boxes, _ = face._models()[0].detect(img)
        if boxes is not None and len(boxes):
            x1, y1, x2, y2 = boxes[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            half = max(x2 - x1, y2 - y1) * 0.85
            img = img.crop(
                (
                    max(0, int(cx - half)),
                    max(0, int(cy - half)),
                    min(img.width, int(cx + half)),
                    min(img.height, int(cy + half)),
                )
            )
        img.thumbnail((320, 320), Image.LANCZOS)
        img.save(dest, "JPEG", quality=88)
        return dest.name
    except Exception as exc:
        print(f"  ! thumbnail de {actor}: {exc}")
        return None


def build_actor(actor: str, per_actor: int, photos_dir: Path | None) -> dict | None:
    if photos_dir is not None:
        paths = local_photos(photos_dir, actor)[:per_actor]
    else:
        paths = download(actor, photo_urls(actor, per_actor))

    if not paths:
        print(f"  - {actor}: sem fotos")
        return None

    vectors: list[np.ndarray] = []
    best_photo: Path | None = None
    for path in paths:
        try:
            vectors.append(face.embed_bytes(path.read_bytes()))
            best_photo = best_photo or path
        except face.NoFaceFound:
            continue
        except Exception as exc:
            print(f"  ! {actor} ({path.name}): {exc}")

    if not vectors:
        print(f"  - {actor}: nenhum rosto reconhecido nas fotos")
        return None

    mean = np.mean(vectors, axis=0)
    mean = mean / np.linalg.norm(mean)
    thumb = save_thumb(best_photo, actor) if best_photo else None
    print(f"  + {actor}: {len(vectors)} foto(s)")
    return {"name": actor, "vector": mean.astype(np.float32), "thumb": thumb or ""}


def main() -> int:
    parser = argparse.ArgumentParser(description="Monta a base de atores")
    parser.add_argument("--limit", type=int, default=0, help="usa apenas os N primeiros atores")
    parser.add_argument("--per-actor", type=int, default=4, help="fotos por ator (padrão: 4)")
    parser.add_argument("--workers", type=int, default=4, help="downloads em paralelo")
    parser.add_argument("--photos-dir", type=Path, help="usa fotos locais em vez da Wikipédia")
    args = parser.parse_args()

    actors = ACTORS[: args.limit] if args.limit else ACTORS
    print(f"Preparando {len(actors)} atores...")
    face.warmup()

    # O download é I/O, o embedding é CPU: baixar em paralelo e embutir em
    # seguida mantém o build rápido sem brigar pelo modelo.
    if args.photos_dir is None:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda a: download(a, photo_urls(a, args.per_actor)), actors))

    entries = [e for a in actors if (e := build_actor(a, args.per_actor, args.photos_dir))]

    if not entries:
        print("Nenhum ator processado — a base não foi gravada.")
        return 1

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_NPZ,
        names=np.array([e["name"] for e in entries]),
        thumbs=np.array([e["thumb"] for e in entries]),
        vectors=np.stack([e["vector"] for e in entries]),
    )
    print(f"\nBase salva em {OUT_NPZ} com {len(entries)} atores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
