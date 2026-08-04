"""Detecção de rosto, medição de qualidade e extração do embedding facial.

O caminho é sempre o mesmo: encontrar o rosto (MTCNN), conferir se a foto tem
qualidade suficiente para valer a pena, alinhar o recorte e passar pela
InceptionResnetV1 treinada no VGGFace2, que devolve um vetor de 512 números.
Rostos parecidos ficam próximos nesse espaço.
"""

from __future__ import annotations

import functools
import io
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models.mtcnn import extract_face, fixed_image_standardization
from PIL import Image, ImageOps

from app import config, imaging

log = logging.getLogger(__name__)


class NoFaceFound(Exception):
    """Nenhum rosto utilizável foi encontrado na imagem."""


@dataclass(frozen=True)
class Quality:
    """O que dá para dizer sobre a foto antes de confiar no resultado."""

    face_pixels: int
    sharpness: float
    brightness: float
    detection_prob: float
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.warnings


@dataclass(frozen=True)
class DetectedFace:
    """Rosto localizado, já com tudo que as outras etapas precisam."""

    image: Image.Image  # imagem inteira (com rotação/EXIF corrigidos)
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 no espaço de `image`
    landmarks: np.ndarray  # 5 pontos: olhos, nariz, cantos da boca
    aligned: torch.Tensor  # recorte 160x160 normalizado para a rede
    crop: Image.Image  # mesmo recorte, sem normalizar (para a prévia)
    quality: Quality


@functools.lru_cache(maxsize=1)
def _models() -> tuple[MTCNN, InceptionResnetV1, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("carregando modelos em %s", device)
    detector = MTCNN(
        image_size=config.FACE_SIZE,
        margin=config.FACE_MARGIN,
        min_face_size=config.MIN_FACE_PIXELS,
        select_largest=True,
        keep_all=True,
        device=device,
    )
    encoder = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return detector, encoder, device


def warmup() -> None:
    """Carrega (e baixa, na primeira vez) os pesos antes do primeiro pedido."""
    _models()


def load_image(data: bytes) -> Image.Image:
    """Abre os bytes recebidos e deixa a imagem pronta para o detector."""
    img = Image.open(io.BytesIO(data))
    # Fotos de celular guardam a rotação no EXIF; sem corrigir, o rosto chega
    # deitado e o MTCNN simplesmente não acha nada.
    img = ImageOps.exif_transpose(img).convert("RGB")

    # Imagens muito grandes só deixam a detecção lenta, sem ganho de precisão.
    longest = max(img.size)
    if longest > config.MAX_WORKING_SIDE:
        scale = config.MAX_WORKING_SIDE / longest
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def _detect_once(img: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Roda o MTCNN e devolve a detecção mais confiável entre as grandes."""
    detector = _models()[0]
    boxes, probs, points = detector.detect(img, landmarks=True)
    if boxes is None or len(boxes) == 0:
        return None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Preferimos o rosto maior (é quem está posando), mas descartando detecções
    # de baixa confiança — costumam ser padrões do fundo.
    order = np.argsort(-areas)
    for i in order:
        if probs[i] is not None and probs[i] >= config.MIN_DETECTION_PROB:
            return boxes[i], float(probs[i]), points[i]
    best = int(np.argmax(probs))
    return boxes[best], float(probs[best]), points[best]


def _detect_with_retries(img: Image.Image) -> tuple[Image.Image, np.ndarray, float, np.ndarray]:
    """Tenta achar o rosto; se falhar, reprocessa a imagem e tenta de novo.

    Foto escura demais e foto rotacionada são os dois motivos mais comuns de
    "não achei rosto" — os dois têm conserto barato.
    """
    attempts: list[tuple[str, Image.Image]] = [
        ("original", img),
        ("contraste", ImageOps.autocontrast(img, cutoff=2)),
        ("girada 90°", img.rotate(90, expand=True)),
        ("girada -90°", img.rotate(-90, expand=True)),
        ("girada 180°", img.rotate(180, expand=True)),
    ]

    for label, candidate in attempts:
        found = _detect_once(candidate)
        if found is not None:
            box, prob, points = found
            if label != "original":
                log.info("rosto encontrado após ajuste: %s", label)
            # O recorte sai da imagem candidata, então é ela que segue adiante.
            return candidate, box, prob, points

    raise NoFaceFound("nenhum rosto detectado")


def _measure(crop: Image.Image, img: Image.Image, box: np.ndarray, prob: float) -> Quality:
    x1, _, x2, _ = (int(v) for v in box)
    face_px = max(0, min(x2, img.width) - max(0, x1))

    # A nitidez é medida no recorte alinhado (sempre 160x160): assim o número
    # significa a mesma coisa numa selfie de 12 MP e numa foto pequena.
    gray = imaging.grayscale(imaging.to_array(crop))
    sharp = imaging.sharpness(gray)
    bright = float(gray.mean())

    warnings: list[str] = []
    if face_px < config.SMALL_FACE_WARN:
        warnings.append("Seu rosto ficou pequeno na foto — chegue mais perto da câmera.")
    if sharp < config.SHARPNESS_WARN:
        warnings.append("A foto saiu meio desfocada — segure firme e tente de novo.")
    if bright < config.BRIGHTNESS_DARK:
        warnings.append("Está escuro demais — procure uma luz de frente para o rosto.")
    elif bright > config.BRIGHTNESS_BRIGHT:
        warnings.append("A foto está estourada de luz — evite luz forte atrás de você.")

    return Quality(
        face_pixels=face_px,
        sharpness=round(sharp, 1),
        brightness=round(bright, 1),
        detection_prob=round(prob, 4),
        warnings=warnings,
    )


def detect(img: Image.Image) -> DetectedFace:
    """Localiza o rosto e prepara tudo que as etapas seguintes consomem."""
    img, box, prob, points = _detect_with_retries(img)

    tensor = extract_face(img, box, image_size=config.FACE_SIZE, margin=config.FACE_MARGIN)
    crop = Image.fromarray(tensor.permute(1, 2, 0).byte().numpy())
    aligned = fixed_image_standardization(tensor)

    return DetectedFace(
        image=img,
        box=tuple(int(v) for v in box),
        landmarks=np.asarray(points, dtype=np.float32),
        aligned=aligned,
        crop=crop,
        quality=_measure(crop, img, box, prob),
    )


def embed(face: DetectedFace) -> np.ndarray:
    """Embedding normalizado (512d) do rosto.

    A média com a versão espelhada é um truque barato de test-time
    augmentation: rostos não são simétricos e a foto raramente está de frente,
    então juntar as duas leituras estabiliza a comparação.
    """
    _, encoder, device = _models()

    batch = torch.stack([face.aligned, torch.flip(face.aligned, dims=[2])]).to(device)
    with torch.no_grad():
        vectors = encoder(batch).cpu().numpy()

    mean = vectors.mean(axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        raise NoFaceFound("rosto detectado, mas sem embedding válido")
    return (mean / norm).astype(np.float32)


def embed_bytes(data: bytes) -> np.ndarray:
    """Atalho para quem só quer o vetor (usado ao montar a base de atores)."""
    return embed(detect(load_image(data)))
