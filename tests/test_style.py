import numpy as np
import pytest
from PIL import Image

from app import imaging, style


def synthetic_face(skin, hair, size=400):
    """Rosto de mentira: fundo, faixa de cabelo em cima e um oval de pele.

    Não precisa parecer gente — a análise de cor só amostra pontos fixos
    calculados a partir da caixa e dos landmarks que passamos na mão.
    """
    arr = np.full((size, size, 3), 200, dtype=np.uint8)
    arr[40:130, 100:300] = hair  # faixa de cabelo acima do rosto
    arr[130:340, 110:290] = skin  # área do rosto

    box = (110, 130, 290, 340)
    landmarks = np.array(
        [
            [160, 200],  # olho esquerdo
            [240, 200],  # olho direito
            [200, 250],  # nariz
            [170, 300],  # canto esquerdo da boca
            [230, 300],  # canto direito da boca
        ],
        dtype=np.float32,
    )
    return Image.fromarray(arr), box, landmarks


def analyze(skin, hair):
    img, box, landmarks = synthetic_face(skin, hair)
    profile = style.analyze(img, box, landmarks)
    assert profile is not None
    return profile


def test_reads_the_skin_tone_from_the_photo():
    profile = analyze(skin=(214, 172, 140), hair=(40, 30, 25))
    measured = np.array([int(profile.skin_hex[i : i + 2], 16) for i in (1, 3, 5)])
    # O balanço de branco desloca um pouco a cor; o tom tem que continuar
    # reconhecível como pele clara e avermelhada.
    assert measured[0] > measured[1] > measured[2]
    assert 150 < measured[0] < 255


def test_warm_light_skin_lands_on_spring():
    profile = analyze(skin=(232, 194, 152), hair=(150, 120, 70))
    assert profile.undertone == "quente"
    assert "Primavera" in profile.season


def test_cool_skin_with_dark_hair_lands_on_winter():
    profile = analyze(skin=(226, 186, 186), hair=(20, 18, 20))
    assert profile.undertone == "fria"
    assert "Inverno" in profile.season


def test_deep_skin_gets_a_deep_season():
    profile = analyze(skin=(92, 62, 44), hair=(24, 18, 14))
    assert profile.depth == "profunda"
    assert profile.season.startswith(("Outono", "Inverno"))


def test_contrast_between_hair_and_skin_changes_the_tip():
    high = analyze(skin=(236, 200, 170), hair=(15, 12, 10))
    low = analyze(skin=(150, 118, 96), hair=(120, 95, 78))
    assert high.contrast == "alto"
    assert low.contrast in {"baixo", "médio"}
    assert high.contrast_tip != low.contrast_tip


def test_every_profile_carries_usable_advice():
    profile = analyze(skin=(210, 165, 130), hair=(60, 45, 35))
    assert len(profile.palette) >= 6
    assert profile.avoid and profile.pieces and profile.metals
    for color in profile.palette + profile.avoid:
        assert set(color) == {"name", "hex"}
        assert color["hex"].startswith("#") and len(color["hex"]) == 7


def test_flags_low_confidence_when_the_hair_is_out_of_frame():
    # Caixa colada no topo: não sobra faixa de cabelo para medir.
    img = Image.fromarray(np.full((300, 300, 3), 190, dtype=np.uint8))
    landmarks = np.array([[110, 40], [190, 40], [150, 80], [120, 120], [180, 120]], dtype=np.float32)
    profile = style.analyze(img, (60, 5, 240, 260), landmarks)
    assert profile is not None
    assert profile.hair_hex is None
    assert any("cabelo" in n for n in profile.notes)


def test_returns_none_when_the_face_is_outside_the_image():
    img = Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8))
    landmarks = np.array([[900, 900]] * 5, dtype=np.float32)
    assert style.analyze(img, (800, 800, 1000, 1000), landmarks) is None


@pytest.mark.parametrize(
    "undertone,depth,contrast,expected",
    [
        ("quente", "clara", "médio", "primavera"),
        ("quente", "profunda", "médio", "outono"),
        ("quente", "média", "alto", "outono"),
        ("fria", "clara", "baixo", "verao"),
        ("fria", "clara", "alto", "inverno"),
        ("neutra", "média", "alto", "inverno"),
        ("neutra", "clara", "baixo", "primavera"),
        ("neutra", "média", "baixo", "verao"),
    ],
)
def test_season_rules(undertone, depth, contrast, expected):
    assert style._pick_season(undertone, depth, contrast) == expected


def test_palettes_are_well_formed():
    for key, palette in style.PALETTES.items():
        assert key in style.PIECES
        for _, hex_code in palette.colors + palette.avoid:
            assert len(hex_code) == 7
            imaging.srgb_to_lab(np.array([int(hex_code[i : i + 2], 16) for i in (1, 3, 5)]))
