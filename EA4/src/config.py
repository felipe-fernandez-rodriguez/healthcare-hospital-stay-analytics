"""Rutas y constantes de EA4 (relativas a la carpeta EA4, independientes del directorio de trabajo)."""
from __future__ import annotations

from pathlib import Path

EA4_DIR: Path = Path(__file__).resolve().parents[1]
PROJECT_ROOT: Path = EA4_DIR.parent

SAMPLE_CSV: Path = EA4_DIR / "data" / "processed" / "healthcare_enriched_sample.csv"
TABLES_DIR: Path = EA4_DIR / "results" / "tables"
REPORT_TXT: Path = EA4_DIR / "reports" / "ea4-summary.txt"
METADATA_JSON: Path = TABLES_DIR / "run_metadata.json"
NOTEBOOK: Path = EA4_DIR / "notebooks" / "04-data-streamlit.ipynb"

SAMPLE_MAX_ROWS: int = 100_000  # límite del CSV de registros individuales (no de los cálculos)
MAX_FILE_MB: float = 50.0        # GitHub avisa a 50 MB por archivo

DATASET_ID: str = "Nicolybgs/healthcare_data"
DATASET_URL: str = "https://huggingface.co/datasets/Nicolybgs/healthcare_data"

REBUILD_COMMAND: str = (
    "jupyter nbconvert --to notebook --execute --inplace EA4/notebooks/04-data-streamlit.ipynb"
)

# Orden de categorías documentado en EA2/EA3.
SEVERITY_ORDER = ["Minor", "Moderate", "Extreme"]
STAFF_GROUP_ORDER = ["Bajo (0-3)", "Medio (4-6)", "Alto (7+)"]
STAY_RANGE_ORDER = ["Corta (1-3 días)", "Moderada (4-7 días)", "Prolongada (>7 días)"]
# Umbrales de stay_range de EA3: (0, 3], (3, 7], (7, inf) con estancia válida >= 1.
STAY_RANGE_BINS = [0, 3, 7, float("inf")]
NO_DATA = "(sin dato)"

# Dimensiones del cubo agregado (dataset completo) que alimenta los filtros del dashboard.
CUBE_DIMENSIONS = ["department", "severity_of_illness", "staff_availability_group", "type_of_admission"]

# Tabla lógica -> archivo CSV en EA4/results/tables. Las ocho primeras usan exactamente el código
# de EA3 (src/analysis/results.build_result_tables); las demás son métricas nuevas de EA4.
TABLE_FILES: dict[str, str] = {
    # EA3 (recalculadas con el mismo código sobre el dataset completo)
    "department": "results_by_department.csv",
    "severity": "results_by_severity.csv",
    "staff_availability": "results_by_staff_availability.csv",
    "department_severity": "results_by_department_severity.csv",
    "stay_range": "results_by_stay_range.csv",
    "complexity": "results_by_complexity_score.csv",
    "indicators": "results_by_department_indicators.csv",
    "data_quality": "results_data_quality.csv",
    # EA4
    "patient_days": "results_patient_days_by_department.csv",
    "severity_composition": "results_severity_expected_stay.csv",
    "staff_level": "results_by_staff_level.csv",
    "correlations": "results_spearman_correlations.csv",
    "eta_squared": "results_eta_squared.csv",
    "distribution": "results_distribution_summary.csv",
    "histograms": "results_histograms.csv",
    "pc2_groups": "results_pc2_group_metrics.csv",
    "filter_cube": "results_filter_cube.csv",
}
EA3_TABLE_KEYS = ["department", "severity", "staff_availability", "department_severity",
                  "stay_range", "complexity", "indicators", "data_quality"]
