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

# ---- EA3: transformación y enriquecimiento (todos los entregables viven en la carpeta EA3/) ----
EA3_DIR: Path = PROJECT_ROOT / "EA3"
RESULTS_DIR: Path = EA3_DIR / "results"
ENRICHED_CSV: Path = EA3_DIR / "data" / "healthcare_enriched_sample.csv"  # máx. CSV_MAX_ROWS filas
ENRICHMENT_SUMMARY_TXT: Path = EA3_DIR / "reports" / "data-enrichment-summary.txt"
# Nombre de cada tabla de resultados -> archivo CSV (todas se calculan con el DataFrame COMPLETO).
RESULT_FILES: dict[str, Path] = {
    "department": RESULTS_DIR / "results_by_department.csv",
    "severity": RESULTS_DIR / "results_by_severity.csv",
    "staff_availability": RESULTS_DIR / "results_by_staff_availability.csv",
    "department_severity": RESULTS_DIR / "results_by_department_severity.csv",
    "stay_range": RESULTS_DIR / "results_by_stay_range.csv",
    "data_quality": RESULTS_DIR / "results_data_quality.csv",
    "complexity": RESULTS_DIR / "results_by_complexity_score.csv",
    "indicators": RESULTS_DIR / "results_by_department_indicators.csv",
}
