"""Enriquecimiento (AE3): variables derivadas sobre el DataFrame ya transformado.

Sigue el mismo contrato que ``transformations.py``: cada paso es una función pura que
devuelve ``(DataFrame, registro)``. Ningún paso elimina filas ni sustituye resultados
indefinidos por cero: cuando un indicador no se puede calcular queda como nulo.
Se aplica DESPUÉS de ``transformations.apply_all`` (nombres en ``snake_case``).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data.transformations import Step, make_record

STAY, DEPOSIT, VISITORS = "stay_in_days", "admission_deposit", "visitors_with_patient"
DEPARTMENT, SEVERITY, ADMISSION = "department", "severity_of_illness", "type_of_admission"

# Reglas exploratorias del proyecto (NO son una escala clínica validada).
SEVERITY_WEIGHTS: dict[str, float] = {"minor": 1.0, "moderate": 2.0, "extreme": 3.0}
ADMISSION_WEIGHTS: dict[str, float] = {"elective": 1.0, "urgent": 1.25, "trauma": 1.5}
# Segmentación de la estancia definida por el proyecto: (0, 3], (3, 7], (7, inf); válida si >= 1.
STAY_BINS = [0, 3, 7, np.inf]
STAY_LABELS = ["Corta (1-3 días)", "Moderada (4-7 días)", "Prolongada (>7 días)"]

NEW_COLUMNS = [
    "deposit_per_stay_day", "visitors_per_day", "stay_range",
    "severity_weight", "admission_type_weight", "clinical_complexity_score",
    "dept_avg_stay", "stay_vs_dept_avg", "dept_severity_avg_deposit",
]


def _valid_stay(df: pd.DataFrame) -> pd.Series:
    """Estancia numérica y positiva; el resto de valores no admite cocientes."""
    stay = pd.to_numeric(df[STAY], errors="coerce")
    return stay.where(stay > 0)


def _ratio(df: pd.DataFrame, numerator: str, new_col: str, description: str) -> Step:
    num = pd.to_numeric(df[numerator], errors="coerce")
    stay = _valid_stay(df)
    out = df.copy()
    out[new_col] = (num / stay).astype("float64")  # denominador inválido -> NaN (nunca 0)
    calculados = int(out[new_col].notna().sum())
    return out, make_record(
        new_col, description, df, out,
        registros_calculados=calculados,
        registros_no_calculables=len(out) - calculados,
        denominadores_invalidos=int(stay.isna().sum()),
    )


def add_deposit_per_stay_day(df: pd.DataFrame) -> Step:
    """``deposit_per_stay_day = admission_deposit / stay_in_days`` (cociente descriptivo)."""
    return _ratio(df, DEPOSIT, "deposit_per_stay_day",
                  "admission_deposit / stay_in_days; nulo si la estancia es nula, 0 o negativa. "
                  "No es el costo diario de hospitalización.")


def add_visitors_per_day(df: pd.DataFrame) -> Step:
    """``visitors_per_day = visitors_with_patient / stay_in_days``."""
    return _ratio(df, VISITORS, "visitors_per_day",
                  "visitors_with_patient / stay_in_days; nulo si la estancia es nula, 0 o negativa. "
                  "No mide carga asistencial.")


def add_stay_range(df: pd.DataFrame) -> Step:
    """``stay_range``: Corta (1-3), Moderada (4-7) o Prolongada (>7); fuera de rango queda nulo."""
    out = df.copy()
    out["stay_range"] = pd.cut(_valid_stay(df).where(lambda s: s >= 1), bins=STAY_BINS, labels=STAY_LABELS, ordered=True)
    dist = {str(k): int(v) for k, v in out["stay_range"].value_counts().sort_index().items()}
    return out, make_record(
        "stay_range", "Segmentación de stay_in_days (umbrales del proyecto, no clínicos oficiales).",
        df, out, distribucion=dist, sin_clasificar=int(out["stay_range"].isna().sum()),
    )


def _norm(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def add_clinical_complexity_score(df: pd.DataFrame) -> Step:
    """Puntaje exploratorio ``severity_weight * admission_type_weight``.

    Las categorías sin peso definido (p. ej. ``Emergency``, que no figura en el mapeo)
    NO reciben puntaje: el indicador queda nulo y se reporta cuáles fueron.
    """
    out = df.copy()
    sev_norm, adm_norm = _norm(df[SEVERITY]), _norm(df[ADMISSION])
    out["severity_weight"] = sev_norm.map(SEVERITY_WEIGHTS).astype("float64")
    out["admission_type_weight"] = adm_norm.map(ADMISSION_WEIGHTS).astype("float64")
    out["clinical_complexity_score"] = out["severity_weight"] * out["admission_type_weight"]
    unknown_sev = df.loc[sev_norm.notna() & out["severity_weight"].isna(), SEVERITY].astype(str).value_counts()
    unknown_adm = df.loc[adm_norm.notna() & out["admission_type_weight"].isna(), ADMISSION].astype(str).value_counts()
    calc = int(out["clinical_complexity_score"].notna().sum())
    present = set(adm_norm.dropna().unique())
    return out, make_record(
        "clinical_complexity_score",
        "severity_weight (Minor 1, Moderate 2, Extreme 3) x admission_type_weight (Elective 1,0; Urgent 1,25; "
        "Trauma 1,5). Regla exploratoria del proyecto, no una escala clínica validada.",
        df, out,
        registros_calculados=calc, registros_no_calculables=len(out) - calc,
        categorias_severidad_sin_peso={str(k): int(v) for k, v in unknown_sev.items()},
        categorias_admision_sin_peso={str(k): int(v) for k, v in unknown_adm.items()},
        categorias_del_mapeo_ausentes_en_datos=sorted(set(ADMISSION_WEIGHTS) - present),
    )


def add_department_benchmark(df: pd.DataFrame) -> Step:
    """``dept_avg_stay`` (media de estancia válida por departamento) y ``stay_vs_dept_avg``."""
    out = df.copy()
    stay = _valid_stay(df)
    out["dept_avg_stay"] = stay.groupby(df[DEPARTMENT], observed=True).transform("mean").astype("float64")
    out["stay_vs_dept_avg"] = (stay - out["dept_avg_stay"]).astype("float64")
    return out, make_record(
        "department_benchmark",
        "dept_avg_stay = media de stay_in_days válida por departamento; stay_vs_dept_avg = stay_in_days - dept_avg_stay. "
        "Comparación relativa, no un juicio clínico ni de eficiencia.",
        df, out,
        registros_calculados=int(out["dept_avg_stay"].notna().sum()),
        departamentos_sin_valores_validos=int((stay.groupby(df[DEPARTMENT], observed=True).count() == 0).sum()),
    )


def add_dept_severity_avg_deposit(df: pd.DataFrame) -> Step:
    """``dept_severity_avg_deposit``: depósito medio de cada combinación departamento x severidad."""
    out = df.copy()
    deposit = pd.to_numeric(df[DEPOSIT], errors="coerce")
    out["dept_severity_avg_deposit"] = deposit.groupby([df[DEPARTMENT], df[SEVERITY]], observed=True).transform("mean").astype("float64")
    return out, make_record(
        "dept_severity_avg_deposit",
        "Media de admission_deposit por (department, severity_of_illness) calculada con los depósitos disponibles "
        "(los nulos se omiten del promedio, no de las filas).",
        df, out,
        registros_calculados=int(out["dept_severity_avg_deposit"].notna().sum()),
        combinaciones=int(out.drop_duplicates([DEPARTMENT, SEVERITY])["dept_severity_avg_deposit"].notna().sum()),
    )


ENRICHMENT_STEPS = (
    add_deposit_per_stay_day,
    add_visitors_per_day,
    add_stay_range,
    add_clinical_complexity_score,
    add_department_benchmark,
    add_dept_severity_avg_deposit,
)


def apply_enrichment(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Aplica todos los enriquecimientos; verifica que no cambie el número de filas."""
    log: list[dict[str, Any]] = []
    for step in ENRICHMENT_STEPS:
        before = len(df)
        df, rec = step(df)
        if len(df) != before:
            raise RuntimeError(f"{rec['paso']} cambió el número de filas ({before} -> {len(df)}).")
        log.append(rec)
    return df, log
