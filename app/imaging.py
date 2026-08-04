"""Utilidades de imagem: conversão de cor e amostragem de regiões.

Nada aqui depende do torch — só numpy e Pillow —, o que deixa a análise de
cor testável sem carregar o modelo.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Matriz sRGB linear -> XYZ (D65) e branco de referência.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Converte RGB 0-255 (…, 3) para CIELAB. Lida com um pixel ou com um lote."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE_D65

    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def lab_to_hex(lab: np.ndarray) -> str:
    """Caminho inverso, só para mostrar a cor amostrada na tela."""
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200

    eps, kappa = 216 / 24389, 24389 / 27
    f = np.array([fx, fy, fz])
    xyz = np.where(f**3 > eps, f**3, (116 * f - 16) / kappa) * _WHITE_D65

    linear = xyz @ np.linalg.inv(_RGB_TO_XYZ).T
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    rgb = np.clip(np.round(srgb * 255), 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def sample_patch(arr: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray | None:
    """Pixels de um quadrado centrado em (cx, cy). None se cair fora da imagem."""
    h, w = arr.shape[:2]
    r = max(2, int(round(radius)))
    x0, x1 = int(round(cx)) - r, int(round(cx)) + r
    y0, y1 = int(round(cy)) - r, int(round(cy)) + r
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return arr[y0:y1, x0:x1].reshape(-1, 3)


def robust_color(pixels: np.ndarray, low: float = 0.25, high: float = 0.75) -> np.ndarray | None:
    """Cor representativa de um conjunto de pixels.

    Fica só com a faixa central de luminância antes de tirar a mediana: assim
    sombra em um lado do rosto e brilho especular no outro não puxam o
    resultado para os extremos.
    """
    if pixels is None or len(pixels) < 8:
        return None
    lum = pixels @ np.array([0.299, 0.587, 0.114])
    lo, hi = np.quantile(lum, [low, high])
    keep = pixels[(lum >= lo) & (lum <= hi)]
    if len(keep) < 4:
        keep = pixels
    return np.median(keep, axis=0)


def sharpness(gray: np.ndarray) -> float:
    """Variância do laplaciano — o medidor de foco mais simples que funciona."""
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(lap.var()) if lap.size else 0.0


def grayscale(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def gray_world_balance(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Correção simples de dominante de cor (luz amarela, luz de LED azulada).

    A análise de tom de pele é sensível à luz do ambiente; equilibrar os canais
    antes de medir evita classificar todo mundo como "quente" numa sala com
    lâmpada amarela.
    """
    arr = to_array(img).astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    if np.any(means < 1e-3):
        return img
    gain = means.mean() / means
    gain = 1.0 + (gain - 1.0) * strength
    return Image.fromarray(np.clip(arr * gain, 0, 255).astype(np.uint8))
