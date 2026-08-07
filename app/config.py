"""Parâmetros ajustáveis do Iaface, reunidos em um lugar só."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
ACTORS_NPZ = DATA_DIR / "actors.npz"
CACHE_DIR = DATA_DIR / "cache"
THUMB_DIR = STATIC_DIR / "actors"
# Onde você joga suas próprias fotos para entrarem na base.
PHOTOS_DIR = ROOT / "fotos"

# --- Detecção -------------------------------------------------------------
# Rostos menores que isso geram embeddings instáveis.
MIN_FACE_PIXELS = 60
# Abaixo dessa confiança do MTCNN é mais provável ser um falso positivo.
MIN_DETECTION_PROB = 0.90
# Imagens gigantes só deixam a detecção lenta; 1600px de lado já é bastante.
MAX_WORKING_SIDE = 1600
# Tamanho do recorte alinhado que entra na rede.
FACE_SIZE = 160
FACE_MARGIN = 20

# --- Qualidade da foto ----------------------------------------------------
# Variância do laplaciano: valores baixos indicam foto tremida/desfocada.
SHARPNESS_WARN = 45.0
BRIGHTNESS_DARK = 60.0
BRIGHTNESS_BRIGHT = 205.0
SMALL_FACE_WARN = 110

# --- Comparação -----------------------------------------------------------
TOP_N = 3
# Faixas de cosseno para traduzir o número em algo legível. A base guarda a
# média de várias fotos por ator, o que puxa os valores um pouco para baixo em
# relação a uma comparação foto-a-foto.
SIMILARITY_BANDS: tuple[tuple[float, str], ...] = (
    (0.62, "muito parecido"),
    (0.52, "bastante parecido"),
    (0.42, "parecido"),
    (0.32, "alguns traços em comum"),
    (0.00, "pouca semelhança"),
)
# Faixa útil do cosseno, usada só para transformar o valor em porcentagem.
PERCENT_FLOOR = 0.10
PERCENT_CEIL = 0.75

# --- Upload ---------------------------------------------------------------
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
