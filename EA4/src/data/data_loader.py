"""Lectura y validación de los archivos que consume el dashboard de EA4.

La app NO descarga Hugging Face ni recalcula el enriquecimiento: lee las tablas agregadas que
generó el notebook 04 (dataset completo) y, solo para vistas de registros individuales, la muestra
sistemática. Si falta un archivo o una columna, se lanza :class:`DataUnavailableError` con la ruta
esperada y el comando para regenerarlo, en lugar de un error difícil de interpretar.

Prioridad de fuentes:
1. Tablas de EA4 (``EA4/results/tables``), calculadas con el dataset completo.
2. Tablas vigentes de EA3 (``EA3/results``) para las ocho tablas comunes, si faltan las de EA4
   (mismo código y mismas columnas; el notebook 04 verifica que coinciden).
3. Muestra de registros (``EA4/data/processed``) solo para gráficos que requieren filas individuales.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .. import config

# Columnas mínimas que la app necesita de cada tabla.
TABLE_SCHEMAS: dict[str, list[str]] = {
    "department": ["department", "registros", "estancia_dias_media", "estancia_dias_mediana",
                   "deposito_admision_media", "deposito_admision_mediana"],
    "severity": ["severity_of_illness", "registros", "estancia_dias_media", "estancia_dias_mediana",
                 "deposito_admision_media", "deposito_admision_mediana"],
    "staff_availability": ["staff_availability_group", "registros", "estancia_dias_media", "estancia_dias_mediana",
                           "deposito_admision_media", "deposito_admision_mediana"],
    "department_severity": ["department", "severity_of_illness", "registros", "estancia_dias_media",
                            "estancia_dias_mediana", "deposito_admision_media", "deposito_admision_mediana"],
    "stay_range": ["stay_range", "registros", "pct_total", "pct_clasificados"],
    "complexity": ["clinical_complexity_score", "registros", "estancia_dias_media", "deposito_admision_media"],
    "indicators": ["department", "registros", "deposito_por_dia_media", "deposito_por_dia_mediana",
                   "visitas_por_dia_media", "visitas_por_dia_mediana"],
    "data_quality": ["variable", "tipo_final", "registros", "nulos", "pct_nulos", "validos", "observaciones"],
    "patient_days": ["department", "admisiones", "pct_admisiones", "dias_paciente", "pct_dias_paciente"],
    "severity_composition": ["severity_of_illness", "registros", "pct_registros", "estancia_observada_media",
                             "estancia_esperada_composicion", "diferencia_dias"],
    "staff_level": ["staff_available", "registros", "estancia_dias_media", "deposito_admision_media"],
    "correlations": ["variable_x", "variable_y", "rho_spearman", "n_pares_validos"],
    "eta_squared": ["factor", "variable", "eta2", "n_validos", "n_grupos"],
    "distribution": ["variable", "validos", "media", "mediana", "desv_est", "q1", "q3", "atipicos", "atipicos_pct"],
    "histograms": ["variable", "limite_inferior", "limite_superior", "registros", "pct"],
    "pc2_groups": ["variable", "categoria", "registros", "estancia_dias_media", "deposito_admision_media"],
    "filter_cube": [*config.CUBE_DIMENSIONS, "stay_in_days", "registros", "depositos_validos",
                    "suma_deposito", "suma_deposito_cuadrado"],
}
# Columnas de la muestra que usa la app (se leen solo estas para ahorrar memoria).
SAMPLE_COLUMNS = ["department", "severity_of_illness", "staff_availability_group", "type_of_admission",
                  "stay_range", "stay_in_days", "admission_deposit", "deposit_per_stay_day", "visitors_per_day",
                  "stay_vs_dept_avg", "staff_available"]
EA3_RESULTS_DIR: Path = config.PROJECT_ROOT / "EA3" / "results"


class DataUnavailableError(FileNotFoundError):
    """Falta un archivo de resultados o no tiene la estructura esperada."""

    def __init__(self, path: Path, problem: str):
        self.path, self.problem = Path(path), problem
        super().__init__(
            f"{problem}: {_rel(path)}. Regenere los resultados desde la raíz del repositorio con:\n"
            f"    {config.REBUILD_COMMAND}"
        )


@dataclass(frozen=True)
class FileStatus:
    tabla: str
    archivo: str
    existe: bool
    filas: int | None
    tamano_kb: float | None
    fuente: str


def table_path(name: str, tables_dir: Path = config.TABLES_DIR) -> Path:
    if name not in config.TABLE_FILES:
        raise KeyError(f"Tabla desconocida: {name!r}. Disponibles: {sorted(config.TABLE_FILES)}")
    return Path(tables_dir) / config.TABLE_FILES[name]


def resolve_table_path(name: str, tables_dir: Path = config.TABLES_DIR) -> tuple[Path, str]:
    """Ruta a usar para ``name`` según la prioridad de fuentes y la etiqueta de su origen."""
    primary = table_path(name, tables_dir)
    if primary.exists():
        return primary, "EA4"
    if name in config.EA3_TABLE_KEYS and (EA3_RESULTS_DIR / config.TABLE_FILES[name]).exists():
        return EA3_RESULTS_DIR / config.TABLE_FILES[name], "EA3"
    return primary, "faltante"


def validate_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataUnavailableError(path, f"Faltan las columnas {missing}")


def load_table(name: str, tables_dir: Path = config.TABLES_DIR) -> pd.DataFrame:
    """Lee una tabla, valida sus columnas y conserva las etiquetas de grupo como texto."""
    path, source = resolve_table_path(name, tables_dir)
    if source == "faltante":
        raise DataUnavailableError(path, "No existe el archivo de resultados")
    try:
        df = pd.read_csv(path, keep_default_na=True)
    except Exception as exc:  # archivo corrupto o vacío
        raise DataUnavailableError(path, f"No se pudo leer el archivo ({exc})") from exc
    validate_columns(df, TABLE_SCHEMAS[name], path)
    for col in ("department", "severity_of_illness", "staff_availability_group", "type_of_admission",
                "clinical_complexity_score", "categoria", "staff_available", "stay_range"):
        if col in df.columns and col != "stay_in_days":
            df[col] = df[col].astype(object).where(df[col].notna(), config.NO_DATA).astype(str)
    df.attrs["fuente"] = source
    df.attrs["archivo"] = _rel(path)
    return df


def load_all_tables(tables_dir: Path = config.TABLES_DIR) -> dict[str, pd.DataFrame]:
    """Todas las tablas; propaga el primer :class:`DataUnavailableError`."""
    return {name: load_table(name, tables_dir) for name in config.TABLE_FILES}


def load_metadata(path: Path = config.METADATA_JSON) -> dict:
    """Metadatos de la última ejecución del notebook (dimensiones reales y fecha)."""
    if not Path(path).exists():
        raise DataUnavailableError(path, "No existen los metadatos de la ejecución")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_sample(path: Path = config.SAMPLE_CSV, columns: list[str] | None = None) -> pd.DataFrame:
    """Muestra sistemática de registros individuales (≤ 100.000 filas). Solo para vistas de muestra."""
    path = Path(path)
    if not path.exists():
        raise DataUnavailableError(path, "No existe la muestra de registros")
    cols = columns or SAMPLE_COLUMNS
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    validate_columns(df, cols, path)
    if len(df) > config.SAMPLE_MAX_ROWS:
        raise DataUnavailableError(path, f"La muestra supera {config.SAMPLE_MAX_ROWS} filas")
    return df


def file_status(tables_dir: Path = config.TABLES_DIR) -> pd.DataFrame:
    """Inventario de los archivos que necesita la app (existencia, filas, tamaño y fuente)."""
    rows = []
    for name in config.TABLE_FILES:
        path, source = resolve_table_path(name, tables_dir)
        exists = source != "faltante"
        rows.append(FileStatus(name, _rel(path), exists, _count_rows(path) if exists else None,
                               round(path.stat().st_size / 1024, 1) if exists else None, source))
    for label, path in (("muestra", config.SAMPLE_CSV), ("metadatos", config.METADATA_JSON)):
        exists = Path(path).exists()
        rows.append(FileStatus(label, _rel(path), exists,
                               _count_rows(path) if exists and path.suffix == ".csv" else None,
                               round(path.stat().st_size / 1024, 1) if exists else None, "EA4" if exists else "faltante"))
    return pd.DataFrame([r.__dict__ for r in rows])


def files_signature(tables_dir: Path = config.TABLES_DIR) -> tuple:
    """Marca de modificación de los archivos: invalida la caché de la app al regenerarlos."""
    paths = [resolve_table_path(n, tables_dir)[0] for n in config.TABLE_FILES] + [config.SAMPLE_CSV, config.METADATA_JSON]
    return tuple((str(p), p.stat().st_mtime_ns if p.exists() else 0) for p in paths)


def _count_rows(path: Path) -> int:
    with open(path, "rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    return p.relative_to(config.PROJECT_ROOT).as_posix() if p.is_relative_to(config.PROJECT_ROOT) else str(p)
