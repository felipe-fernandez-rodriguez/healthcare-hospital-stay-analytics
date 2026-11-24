"""Construcción y exportación de las tablas de EA4 (lo usa el notebook 04, no la app).

Reutiliza el código de EA1-EA3 del paquete ``src`` de la raíz (``build_result_tables`` y
``group_metrics``) y añade las métricas de :mod:`business_metrics`. Requiere que la raíz del
repositorio esté en ``sys.path``; la app de Streamlit no importa este módulo.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import descriptive as ds  # raíz del repositorio (EA2/EA3)
from src.analysis.results import build_result_tables  # EA3

from .. import config
from . import business_metrics as bm

# Variables de PC2 agrupadas por categorías y factores de la razón de varianza.
PC2_FACTORS = ["ward_facility_code", "age", "gender", "type_of_admission", "insurance", "health_conditions"]
ETA_FACTORS = ["department", "ward_facility_code", "age", "gender", "severity_of_illness", "type_of_admission",
               "insurance", "health_conditions", "staff_available", "staff_availability_group"]
SPEARMAN_PAIRS = [
    ("staff_available", "stay_in_days"), ("staff_available", "admission_deposit"),
    ("available_extra_rooms_in_hospital", "stay_in_days"), ("available_extra_rooms_in_hospital", "admission_deposit"),
    ("visitors_with_patient", "stay_in_days"), ("visitors_with_patient", "admission_deposit"),
    ("stay_in_days", "admission_deposit"),
]
DISTRIBUTION_COLUMNS = ["stay_in_days", "admission_deposit", "deposit_per_stay_day", "visitors_per_day", "stay_vs_dept_avg"]
HISTOGRAMS = [  # (columna, intervalos, entera)
    ("stay_in_days", None, True),
    ("admission_deposit", 40, False),
    ("deposit_per_stay_day", 40, False),
    ("visitors_per_day", 32, False),
    ("stay_vs_dept_avg", 46, False),
]
REQUIRED_COLUMNS = sorted({*PC2_FACTORS, *ETA_FACTORS, *DISTRIBUTION_COLUMNS, *config.CUBE_DIMENSIONS,
                           "available_extra_rooms_in_hospital", "visitors_with_patient", "stay_range",
                           "clinical_complexity_score", "dept_avg_stay", "dept_severity_avg_deposit"})


def check_required_columns(df: pd.DataFrame) -> list[str]:
    """Columnas requeridas que faltan (lista vacía si están todas)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def pc2_group_metrics(df: pd.DataFrame, factors: list[str] = PC2_FACTORS) -> pd.DataFrame:
    """``group_metrics`` de EA3 para cada variable de PC2, en formato largo (variable, categoría)."""
    parts = [ds.group_metrics(df, f).rename(columns={f: "categoria"}).assign(variable=f) for f in factors]
    out = pd.concat(parts, ignore_index=True)
    return out[["variable", "categoria", *[c for c in out.columns if c not in ("variable", "categoria")]]]


def build_ea4_tables(df: pd.DataFrame, enrichment_log: list[dict[str, Any]] | None = None) -> dict[str, pd.DataFrame]:
    """Todas las tablas de EA4 calculadas con el DataFrame enriquecido COMPLETO."""
    missing = check_required_columns(df)
    if missing:
        raise KeyError(f"Faltan columnas del enriquecimiento de EA3: {missing}")
    tables = dict(build_result_tables(df, enrichment_log))  # las 8 tablas de EA3, mismo código
    tables["patient_days"] = bm.patient_days_share(df, "department")
    tables["severity_composition"] = bm.expected_stay_by_composition(df, "severity_of_illness", "department")
    tables["staff_level"] = ds.group_metrics(df, "staff_available")
    tables["correlations"] = bm.spearman_table(df, SPEARMAN_PAIRS)
    tables["eta_squared"] = bm.eta_squared_table(df, ETA_FACTORS)
    tables["distribution"] = bm.distribution_summary(df, DISTRIBUTION_COLUMNS)
    tables["histograms"] = pd.concat(
        [bm.histogram_table(df[c], c, bins=b or 0, integer=i) for c, b, i in HISTOGRAMS], ignore_index=True)
    tables["pc2_groups"] = pc2_group_metrics(df)
    tables["filter_cube"] = bm.build_cube(df)
    return tables


def export_tables(tables: dict[str, pd.DataFrame], tables_dir: Path = config.TABLES_DIR) -> pd.DataFrame:
    """Escribe cada tabla en UTF-8 (crea la carpeta) y devuelve un inventario con filas y tamaño."""
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, table in tables.items():
        path = tables_dir / config.TABLE_FILES[key]
        table.to_csv(path, index=False, encoding="utf-8")
        rows.append({"tabla": key, "archivo": _rel(path), "filas": len(table), "columnas": table.shape[1],
                     "tamano_kb": round(path.stat().st_size / 1024, 1),
                     "origen": "EA3 (mismo código)" if key in config.EA3_TABLE_KEYS else "EA4"})
    return pd.DataFrame(rows)


def export_sample(df: pd.DataFrame, path: Path = config.SAMPLE_CSV, max_rows: int = config.SAMPLE_MAX_ROWS) -> dict[str, Any]:
    """Muestra sistemática determinista (una de cada ``paso`` filas, sin azar) con ≤ ``max_rows`` filas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, math.ceil(len(df) / max_rows))
    sample = df.iloc[::step].head(max_rows)
    sample.to_csv(path, index=False, encoding="utf-8")
    return {"archivo": _rel(path), "filas_dataset_completo": len(df), "filas_exportadas": len(sample),
            "metodo": f"sistemática: una de cada {step} filas en el orden original (sin azar)", "paso": step,
            "es_parcial": len(sample) < len(df), "columnas": sample.shape[1],
            "tamano_mb": round(path.stat().st_size / 1e6, 2)}


def quality_summary(df: pd.DataFrame, transform_log: list[dict[str, Any]], enrichment_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Controles de calidad de la ejecución, tomados de los registros de EA1-EA3 (no se recalculan)."""
    tlog = {r["paso"]: r for r in transform_log}
    elog = {r["paso"]: r for r in enrichment_log}
    score = elog.get("clinical_complexity_score", {})
    return {
        "duplicados_exactos": tlog.get("drop_exact_duplicates", {}).get("duplicados_eliminados"),
        "errores_conversion": tlog.get("convert_numeric_types", {}).get("valores_no_convertibles"),
        "patientid_negativos": int((df["patientid"] < 0).sum()) if "patientid" in df else None,
        "patientid_repetidos": int(df["patientid"].duplicated().sum()) if "patientid" in df else None,
        "categorias_sin_peso": score.get("categorias_admision_sin_peso"),
        "puntaje_calculado": score.get("registros_calculados"),
        "sin_rango_estancia": elog.get("stay_range", {}).get("sin_clasificar"),
        "nulos_por_columna": {c: int(v) for c, v in df.isna().sum().items() if v},
    }


def write_metadata(info: dict[str, Any], raw: pd.DataFrame, df: pd.DataFrame, sample_info: dict[str, Any],
                   inventory: pd.DataFrame, quality: dict[str, Any] | None = None,
                   path: Path = config.METADATA_JSON) -> Path:
    """Metadatos de la ejecución que muestra la app (fecha real de procesamiento y dimensiones)."""
    meta = {
        "dataset_id": info["dataset_id"], "dataset_url": config.DATASET_URL,
        "config": info["config"], "split": info["split"],
        "generado": datetime.now().astimezone().isoformat(timespec="seconds"),
        "filas_cargadas": len(raw), "columnas_originales": raw.shape[1],
        "filas_analizadas": len(df), "columnas_enriquecidas": df.shape[1],
        "muestra": sample_info,
        "calidad": quality or {},
        "tablas": inventory.to_dict(orient="records"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    return p.relative_to(config.PROJECT_ROOT).as_posix() if p.is_relative_to(config.PROJECT_ROOT) else str(p)
