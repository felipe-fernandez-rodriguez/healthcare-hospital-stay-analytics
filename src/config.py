"""Configuración central del proyecto: rutas relativas a la raíz y constantes."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

RAW_CSV: Path = DATA_RAW_DIR / "healthcare_raw.csv"
PROCESSED_CSV: Path = DATA_PROCESSED_DIR / "healthcare_processed.csv"
DATA_SUMMARY_TXT: Path = REPORTS_DIR / "data-summary.txt"
EXPLORATORY_TXT: Path = REPORTS_DIR / "exploratory-analysis.txt"

DATASET_ID: str = "Nicolybgs/healthcare_data"
DATASET_URL: str = "https://huggingface.co/datasets/Nicolybgs/healthcare_data"
DATASET_CONFIG: str | None = "default"  # única configuración expuesta (verificado)
DATASET_SPLIT: str = "train"  # único split expuesto (verificado)

# El límite SOLO aplica a los CSV exportados (primeras N filas, sin aleatoriedad).
# El DataFrame que se transforma, resume y explora siempre contiene todas las filas.
# GitHub avisa a 50 MB y bloquea a 100 MB por archivo.
CSV_MAX_ROWS: int = 100_000
