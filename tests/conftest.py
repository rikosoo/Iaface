import urllib.error
import urllib.request
from pathlib import Path

import pytest

# Foto de rosto usada nos testes de ponta a ponta. Vem do repositório do
# facenet-pytorch para não guardar imagem de pessoa real neste projeto; fica
# em cache depois do primeiro download.
PHOTO_URL = (
    "https://raw.githubusercontent.com/timesler/facenet-pytorch/master"
    "/data/test_images/angelina_jolie/1.jpg"
)
PHOTO_CACHE = Path(__file__).parent / "fixtures" / "face.jpg"


@pytest.fixture(scope="session")
def photo() -> bytes:
    if PHOTO_CACHE.exists():
        return PHOTO_CACHE.read_bytes()

    PHOTO_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(PHOTO_URL, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"sem foto de teste (offline): {exc}")

    PHOTO_CACHE.write_bytes(data)
    return data
