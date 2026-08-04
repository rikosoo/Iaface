import numpy as np
import pytest
from PIL import Image

from app import imaging


def test_lab_of_known_colors():
    # Referências clássicas do CIELAB: branco, preto e vermelho puro.
    assert imaging.srgb_to_lab(np.array([255, 255, 255])) == pytest.approx([100, 0, 0], abs=0.1)
    assert imaging.srgb_to_lab(np.array([0, 0, 0])) == pytest.approx([0, 0, 0], abs=0.1)
    assert imaging.srgb_to_lab(np.array([255, 0, 0])) == pytest.approx([53.24, 80.09, 67.20], abs=0.1)


def test_lab_roundtrip_returns_the_same_hex():
    for rgb in [(210, 170, 140), (60, 40, 30), (18, 22, 30), (240, 200, 210)]:
        lab = imaging.srgb_to_lab(np.array(rgb))
        assert imaging.lab_to_hex(lab) == "#{:02x}{:02x}{:02x}".format(*rgb)


def test_lab_handles_batches():
    batch = np.array([[[255, 255, 255], [0, 0, 0]]])
    assert imaging.srgb_to_lab(batch).shape == (1, 2, 3)


def test_robust_color_ignores_shadow_and_highlight():
    # Base marrom com alguns pixels pretos (sombra) e brancos (brilho).
    pixels = np.array([[150, 110, 90]] * 40 + [[0, 0, 0]] * 10 + [[255, 255, 255]] * 10)
    assert imaging.robust_color(pixels) == pytest.approx([150, 110, 90], abs=1)


def test_robust_color_needs_enough_pixels():
    assert imaging.robust_color(np.array([[1, 2, 3]])) is None
    assert imaging.robust_color(None) is None


def test_sample_patch_clamps_to_the_image():
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    assert imaging.sample_patch(arr, 0, 0, 5) is not None  # canto: recorta o que dá
    assert imaging.sample_patch(arr, -50, -50, 5) is None  # fora da imagem


def test_sharpness_separates_blur_from_detail():
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (60, 60)).astype(np.float32)
    flat = np.full((60, 60), 128.0, dtype=np.float32)
    assert imaging.sharpness(noisy) > imaging.sharpness(flat)
    assert imaging.sharpness(flat) == pytest.approx(0.0)


def test_gray_world_balance_neutralizes_a_color_cast():
    # Imagem cinza com dominante amarela forte.
    arr = np.full((30, 30, 3), 128, dtype=np.uint8)
    arr[:, :, 2] = 60
    balanced = imaging.to_array(imaging.gray_world_balance(Image.fromarray(arr)))
    before = 128 - 60
    after = float(balanced[..., 0].mean() - balanced[..., 2].mean())
    assert after < before
