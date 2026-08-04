"""Detecção de rosto e extração de embeddings faciais.

Usa MTCNN para localizar/alinhar o rosto e a InceptionResnetV1 treinada no
VGGFace2 para gerar um vetor de 512 dimensões que representa a face.
Rostos parecidos ficam próximos nesse espaço (similaridade de cosseno).
"""

from __future__ import annotations

import functools
import io

import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1

# Rostos menores que isso costumam gerar embeddings ruins.
MIN_FACE_PIXELS = 60


class NoFaceFound(Exception):
    """Nenhum rosto utilizável foi encontrado na imagem."""


@functools.lru_cache(maxsize=1)
def _models() -> tuple[MTCNN, InceptionResnetV1, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=MIN_FACE_PIXELS,
        post_process=True,
        select_largest=True,
        device=device,
    )
    encoder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return detector, encoder, device


def warmup() -> None:
    """Baixa/carrega os pesos antes do primeiro pedido do usuário."""
    _models()


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    # Fotos de celular guardam a rotação no EXIF; sem isso o rosto vem deitado.
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def embed_image(img: Image.Image) -> np.ndarray:
    """Retorna o embedding normalizado (512d) do maior rosto da imagem."""
    detector, encoder, device = _models()

    face = detector(img)
    if face is None:
        raise NoFaceFound("nenhum rosto detectado")

    with torch.no_grad():
        vec = encoder(face.unsqueeze(0).to(device))[0].cpu().numpy()

    norm = np.linalg.norm(vec)
    if norm == 0:
        raise NoFaceFound("rosto detectado, mas sem embedding válido")
    return (vec / norm).astype(np.float32)


def embed_bytes(data: bytes) -> np.ndarray:
    return embed_image(load_image(data))
