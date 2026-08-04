"""Sugestão de cores e peças a partir das cores medidas no rosto.

Nada aqui é adivinhação do modelo de reconhecimento: são medidas de cor
tiradas da própria foto (pele nas bochechas e testa, cabelo acima da cabeça,
íris ao redor dos olhos), convertidas para CIELAB e encaixadas na análise
sazonal clássica — a mesma lógica de "cartela de cores" usada por
consultoria de imagem.

Três medidas decidem tudo:

- **subtom** (quente / frio / neutro): o ângulo de matiz da pele em Lab.
- **profundidade** (clara / média / profunda): o ângulo ITA°, padrão em
  dermatologia para classificar tom de pele.
- **contraste**: a diferença de luminosidade entre cabelo e pele.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from app import imaging

# --- Cartelas -------------------------------------------------------------

Color = tuple[str, str]  # (nome, hex)


@dataclass(frozen=True)
class Palette:
    season: str
    idea: str
    metals: str
    colors: list[Color]
    avoid: list[Color]


PALETTES: dict[str, Palette] = {
    "primavera": Palette(
        season="Primavera · paleta quente e luminosa",
        idea="Cores vivas e luminosas. Fuja de tons apagados: eles apagam você junto.",
        metals="Dourado, ouro rosé e cobre polido",
        colors=[
            ("Coral", "#ff6f5e"),
            ("Vermelho-tomate", "#e2452f"),
            ("Pêssego", "#ffb48a"),
            ("Amarelo-manteiga", "#ffd97d"),
            ("Verde-abacate", "#a7c957"),
            ("Turquesa", "#2fbfa8"),
            ("Azul-céu", "#6fb7e8"),
            ("Roxo-orquídea", "#b36bd4"),
        ],
        avoid=[("Preto puro", "#111318"), ("Cinza-chumbo", "#4a4f57"), ("Bordô escuro", "#4a1220")],
    ),
    "verao": Palette(
        season="Verão · paleta fria e suave",
        idea="Tons lavados, com um véu de cinza por cima. Cor forte demais rouba a cena do rosto.",
        metals="Prata, níquel e ouro branco",
        colors=[
            ("Framboesa", "#c2456b"),
            ("Rosa-antigo", "#d98ba5"),
            ("Lilás", "#b3a0d6"),
            ("Ameixa suave", "#7e5c8f"),
            ("Azul-serenity", "#8fb3de"),
            ("Azul-petróleo claro", "#4f7f8b"),
            ("Verde-sálvia", "#9bbfa8"),
            ("Cinza-pérola", "#c9ccd3"),
        ],
        avoid=[("Laranja", "#e8641c"), ("Mostarda", "#c99a2e"), ("Marrom-alaranjado", "#8a4a1e")],
    ),
    "outono": Palette(
        season="Outono · paleta quente e terrosa",
        idea="Cores terrosas e encorpadas, com fundo dourado. Tons pastel somem em você.",
        metals="Ouro velho, bronze e cobre fosco",
        colors=[
            ("Vermelho-tijolo", "#9e2b25"),
            ("Ferrugem", "#b5502a"),
            ("Terracota", "#c56a46"),
            ("Mostarda", "#c99a2e"),
            ("Caramelo", "#a9743f"),
            ("Verde-oliva", "#7d8c4a"),
            ("Verde-musgo", "#6b7a3a"),
            ("Roxo-berinjela", "#5c3a5a"),
        ],
        avoid=[("Rosa-bebê", "#f3b9cd"), ("Azul-gelo", "#cfe3f2"), ("Cinza frio", "#9aa3b2")],
    ),
    "inverno": Palette(
        season="Inverno · paleta fria e intensa",
        idea="Cores puras e saturadas, e o contraste alto que combina com você. É a única cartela em que preto e branco funcionam de verdade.",
        metals="Prata, platina e aço",
        colors=[
            ("Vermelho-rubi", "#c41e3a"),
            ("Roxo-ametista", "#7a2fa8"),
            ("Fúcsia", "#d6237e"),
            ("Azul-royal", "#1f4fd8"),
            ("Verde-esmeralda", "#0e8f6e"),
            ("Branco-neve", "#f5f7fa"),
            ("Grafite", "#3a4048"),
            ("Preto", "#111318"),
        ],
        avoid=[("Bege alaranjado", "#d8b48a"), ("Mostarda", "#c99a2e"), ("Salmão", "#f0a08a")],
    ),
}

# Peças concretas por cartela — a parte prática da recomendação.
PIECES: dict[str, list[str]] = {
    "primavera": [
        "Cachecol coral ou pêssego: puxa luz para o rosto sem pesar",
        "Camiseta turquesa ou verde-abacate no lugar da preta",
        "Jaqueta jeans clara ou caqui como peça neutra",
        "Óculos e relógio dourados",
    ],
    "verao": [
        "Cachecol lilás ou rosa-antigo em tecido leve",
        "Camisa azul-serenity — provavelmente sua melhor peça básica",
        "Troque o preto por azul-marinho e cinza-chumbo",
        "Acessórios prateados, sem brilho exagerado",
    ],
    "outono": [
        "Cachecol ferrugem ou mostarda em lã ou tricô",
        "Suéter verde-oliva ou caramelo",
        "Jaqueta de camurça ou couro marrom",
        "Troque o branco puro por off-white ou creme",
    ],
    "inverno": [
        "Cachecol vermelho-rubi ou roxo-ametista",
        "Camisa branca pura e casaco preto — o contraste é a seu favor",
        "Uma peça em azul-royal ou esmeralda para colorir o visual",
        "Acessórios prateados e armação de óculos escura",
    ],
}

CONTRAST_TIPS: dict[str, str] = {
    "alto": "Aposte em contraste: peça clara com peça escura, ou uma cor forte contra um neutro. Combinações tom sobre tom deixam o visual apagado em você.",
    "médio": "Você aguenta tanto contraste quanto tom sobre tom. Uma cor de destaque perto do rosto (gola, cachecol) já resolve o visual.",
    "baixo": "Tom sobre tom cai muito bem: variações do mesmo tom, sem preto contra branco. Contraste demais compete com o rosto em vez de valorizá-lo.",
}


# --- Análise --------------------------------------------------------------


@dataclass(frozen=True)
class StyleProfile:
    skin_hex: str
    hair_hex: str | None
    eyes_hex: str | None
    undertone: str
    depth: str
    contrast: str
    season: str
    idea: str
    metals: str
    contrast_tip: str
    palette: list[dict[str, str]]
    avoid: list[dict[str, str]]
    pieces: list[str]
    confidence: str
    notes: list[str] = field(default_factory=list)


def _skin_color(arr: np.ndarray, box, landmarks) -> np.ndarray | None:
    """Mediana da pele nas bochechas e na testa."""
    (eye_l, eye_r, nose, mouth_l, mouth_r) = landmarks
    width = box[2] - box[0]
    radius = max(3.0, width * 0.055)

    eye_mid = (eye_l + eye_r) / 2
    # A testa fica na direção oposta ao nariz, a partir da linha dos olhos.
    forehead = eye_mid + (eye_mid - nose) * 0.75

    spots = [
        (eye_l + mouth_l) / 2,  # bochecha esquerda
        (eye_r + mouth_r) / 2,  # bochecha direita
        forehead,
    ]

    samples = [
        color
        for spot in spots
        if (color := imaging.robust_color(imaging.sample_patch(arr, spot[0], spot[1], radius)))
        is not None
    ]
    return np.median(samples, axis=0) if samples else None


def _hair_color(arr: np.ndarray, box) -> tuple[np.ndarray | None, bool]:
    """Cor do cabelo na faixa acima do rosto.

    Devolve também se a medida é confiável: careca, touca, fundo escuro ou
    topo da cabeça cortado na foto caem todos aqui.
    """
    x1, y1, x2, _ = box
    width, height = x2 - x1, box[3] - y1

    # Se o topo da cabeça ficou cortado, ainda dá para medir a franja: usamos
    # o que sobrou da faixa, desde que não seja uma tira fina demais.
    top = max(0, int(y1 - height * 0.42))
    bottom = min(arr.shape[0], int(y1 + height * 0.08))
    inset = int(width * 0.22)
    left, right = max(0, x1 + inset), min(arr.shape[1], x2 - inset)

    if bottom - top < max(8, height * 0.08) or right - left < 6:
        return None, False

    band = arr[top:bottom, left:right].reshape(-1, 3)
    if len(band) < 40:
        return None, False

    # A faixa mistura cabelo e fundo. O cabelo é quase sempre a parte mais
    # escura, então ficamos com o terço inferior de luminância.
    lum = band @ np.array([0.299, 0.587, 0.114])
    hair = band[lum <= np.quantile(lum, 0.35)]
    if len(hair) < 20:
        return None, False

    color = np.median(hair, axis=0)

    # Se a faixa toda tem a mesma cor, provavelmente estamos medindo a parede.
    uniform = float(np.std(lum)) < 12.0
    return color, not uniform


def _eye_color(arr: np.ndarray, landmarks, box) -> np.ndarray | None:
    """Cor da íris: descarta o branco do olho e o preto da pupila/cílios."""
    radius = max(2.0, (box[2] - box[0]) * 0.035)
    samples = []
    for eye in landmarks[:2]:
        patch = imaging.sample_patch(arr, eye[0], eye[1], radius)
        if patch is None:
            continue
        color = imaging.robust_color(patch, low=0.30, high=0.65)
        if color is not None:
            samples.append(color)
    return np.median(samples, axis=0) if samples else None


def _classify_undertone(lab: np.ndarray) -> tuple[str, float]:
    """Ângulo de matiz da pele: mais amarelo = quente, mais rosa = frio."""
    hue = float(np.degrees(np.arctan2(lab[2], lab[1])))
    if hue >= 55:
        return "quente", hue
    if hue <= 45:
        return "fria", hue
    return "neutra", hue


def _classify_depth(lab: np.ndarray) -> tuple[str, float]:
    """Ângulo ITA° — quanto maior, mais clara a pele."""
    ita = float(np.degrees(np.arctan2(lab[0] - 50, lab[2]))) if lab[2] else 0.0
    if ita > 41:
        return "clara", ita
    if ita >= 10:
        return "média", ita
    return "profunda", ita


def _classify_contrast(skin_lab: np.ndarray, hair_lab: np.ndarray | None) -> tuple[str, float]:
    if hair_lab is None:
        return "médio", 0.0
    delta = abs(float(skin_lab[0] - hair_lab[0]))
    if delta >= 45:
        return "alto", delta
    if delta >= 25:
        return "médio", delta
    return "baixo", delta


def _pick_season(undertone: str, depth: str, contrast: str) -> str:
    if undertone == "quente":
        return "outono" if depth == "profunda" or contrast == "alto" else "primavera"
    if undertone == "fria":
        return "inverno" if depth == "profunda" or contrast == "alto" else "verao"
    # Subtom neutro: quem decide é o contraste, e depois a profundidade.
    if contrast == "alto":
        return "inverno"
    if depth == "profunda":
        return "outono"
    return "primavera" if depth == "clara" else "verao"


def analyze(img: Image.Image, box, landmarks) -> StyleProfile | None:
    """Perfil de cor a partir da foto. None se a pele não pôde ser medida."""
    # A luz do ambiente muda o tom medido; equilibrar antes evita classificar
    # todo mundo como "quente" numa sala de lâmpada amarela.
    balanced = imaging.gray_world_balance(img, strength=0.7)
    arr = imaging.to_array(balanced)
    landmarks = np.asarray(landmarks, dtype=np.float32)

    skin = _skin_color(arr, box, landmarks)
    if skin is None:
        return None

    hair, hair_ok = _hair_color(arr, box)
    eyes = _eye_color(arr, landmarks, box)

    skin_lab = imaging.srgb_to_lab(skin)
    hair_lab = imaging.srgb_to_lab(hair) if hair is not None and hair_ok else None

    undertone, hue = _classify_undertone(skin_lab)
    depth, _ = _classify_depth(skin_lab)
    contrast, _ = _classify_contrast(skin_lab, hair_lab)
    season = _pick_season(undertone, depth, contrast)
    palette = PALETTES[season]

    notes: list[str] = []
    if not hair_ok:
        notes.append("Não consegui medir bem o cabelo — enquadre a cabeça inteira para afinar o resultado.")
    if 45 < hue < 55:
        notes.append("Seu subtom ficou em cima do muro entre quente e frio: vale testar as duas cartelas.")

    confidence = "alta" if hair_ok and not notes else "média" if hair_ok or eyes is not None else "baixa"

    return StyleProfile(
        skin_hex=imaging.lab_to_hex(skin_lab),
        hair_hex=imaging.lab_to_hex(hair_lab) if hair_lab is not None else None,
        eyes_hex=imaging.lab_to_hex(imaging.srgb_to_lab(eyes)) if eyes is not None else None,
        undertone=undertone,
        depth=depth,
        contrast=contrast,
        season=palette.season,
        idea=palette.idea,
        metals=palette.metals,
        contrast_tip=CONTRAST_TIPS[contrast],
        palette=[{"name": n, "hex": h} for n, h in palette.colors],
        avoid=[{"name": n, "hex": h} for n, h in palette.avoid],
        pieces=PIECES[season],
        confidence=confidence,
        notes=notes,
    )
