"""Tablas de resultados de AE3: construcción, exportación y hallazgos en texto.

Todas las tablas se calculan con el DataFrame COMPLETO (nunca con el CSV de muestra).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.analysis import descriptive as ds
from src.analysis.quality import quality_table

TABLE_QUESTIONS = {
    "department": "P1: estancia y depósito por departamento médico",
    "severity": "P2: estancia y depósito por severidad de la enfermedad",
    "staff_availability": "P3: estancia y depósito por grupo de disponibilidad de personal (umbrales de EA2)",
    "department_severity": "P4: cruce departamento x severidad",
    "stay_range": "P5 (apoyo): distribución de registros por rango de estancia",
    "complexity": "P5: estancia y depósito por puntaje exploratorio de complejidad",
    "indicators": "P5: depósito por día y visitas por día de estancia, por departamento",
    "data_quality": "Calidad: nulos, tipos y validez por variable",
}


def quality_notes(df: pd.DataFrame, enrichment_log: list[dict[str, Any]]) -> dict[str, str]:
    """Observaciones por variable, calculadas desde los datos y el registro de enriquecimiento."""
    notes: dict[str, str] = {}
    if "patientid" in df:
        notes["patientid"] = (f"Identificador, no cuantitativo. Repetido en {int(df['patientid'].duplicated().sum())} filas; "
                              f"{int((df['patientid'] < 0).sum())} valores negativos (semántica no documentada; se conserva).")
    if "health_conditions" in df and df["health_conditions"].isna().any():
        notes["health_conditions"] = "Nulos conservados sin imputar (significado no documentado)."
    for rec in enrichment_log:
        name = rec["paso"]
        if "registros_no_calculables" in rec:
            col = {"clinical_complexity_score": "clinical_complexity_score"}.get(name, name)
            msg = f"{rec['registros_calculados']} calculados; {rec['registros_no_calculables']} no calculables (nulo)."
            if rec.get("categorias_admision_sin_peso"):
                msg += f" Categorías de admisión sin peso: {rec['categorias_admision_sin_peso']}."
            if rec.get("denominadores_invalidos"):
                msg += f" Denominadores inválidos: {rec['denominadores_invalidos']}."
            if col in df.columns:
                notes[col] = msg
        if name == "stay_range":
            notes["stay_range"] = f"{rec['sin_clasificar']} registros sin clasificar (estancia nula o < 1)."
    return notes


def build_result_tables(df: pd.DataFrame, enrichment_log: list[dict[str, Any]] | None = None) -> dict[str, pd.DataFrame]:
    """Construye todas las tablas agregadas a partir del DataFrame enriquecido completo."""
    tables = {
        "department": ds.group_metrics(df, "department"),
        "severity": ds.group_metrics(df, "severity_of_illness"),
        "staff_availability": ds.group_metrics(df, "staff_availability_group"),
        "department_severity": ds.group_metrics(df, ["department", "severity_of_illness"]),
        "stay_range": ds.stay_range_table(df),
        "complexity": ds.group_metrics(df, "clinical_complexity_score"),
        "indicators": ds.group_metrics(df, "department", targets=("deposit_per_stay_day", "visitors_per_day")),
        "data_quality": quality_table(df, quality_notes(df, enrichment_log or [])),
    }
    return tables


def export_tables(tables: dict[str, pd.DataFrame], files: dict[str, Path] | None = None) -> dict[str, str]:
    """Escribe cada tabla en CSV UTF-8 (crea carpetas) y devuelve ``{tabla: ruta relativa}``."""
    files = files or config.RESULT_FILES
    paths: dict[str, str] = {}
    for key, table in tables.items():
        target = Path(files[key])
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, index=False, encoding="utf-8")
        paths[key] = target.resolve().relative_to(config.PROJECT_ROOT).as_posix() if target.resolve().is_relative_to(config.PROJECT_ROOT) else str(target)
    return paths


def key_findings(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Hallazgos descriptivos redactados desde las tablas (sin cifras escritas a mano)."""
    out: list[str] = []
    dept = tables["department"].sort_values("estancia_dias_mediana")
    lo, hi = dept.iloc[0], dept.iloc[-1]
    out.append(f"Estancia mediana por departamento: de {lo['estancia_dias_mediana']:.1f} días ({lo['department']}, n={int(lo['estancia_dias_validos'])}) "
               f"a {hi['estancia_dias_mediana']:.1f} días ({hi['department']}, n={int(hi['estancia_dias_validos'])}).")
    dep = tables["department"].sort_values("deposito_admision_media")
    out.append(f"Depósito medio por departamento: de {dep.iloc[0]['deposito_admision_media']:.2f} ({dep.iloc[0]['department']}) "
               f"a {dep.iloc[-1]['deposito_admision_media']:.2f} ({dep.iloc[-1]['department']}) (unidad monetaria no documentada).")
    sev = tables["severity"]
    out.append("Estancia media por severidad: " + "; ".join(
        f"{r['severity_of_illness']} {r['estancia_dias_media']:.2f} d (n={int(r['estancia_dias_validos'])})" for _, r in sev.iterrows()) + ".")
    out.append("Depósito medio por severidad: " + "; ".join(
        f"{r['severity_of_illness']} {r['deposito_admision_media']:.2f}" for _, r in sev.iterrows()) + ".")
    st = tables["staff_availability"]
    out.append("Estancia media por grupo de personal: " + "; ".join(
        f"{r['staff_availability_group']} {r['estancia_dias_media']:.2f} d (n={int(r['estancia_dias_validos'])})" for _, r in st.iterrows()) + ".")
    sr = tables["stay_range"]
    out.append("Rangos de estancia (% sobre el total): " + "; ".join(
        f"{r['stay_range']} {r['pct_total']:.2f} % (n={int(r['registros'])})" for _, r in sr.iterrows() if r["stay_range"] != "Total") + ".")
    ind = tables["indicators"].sort_values("deposito_por_dia_mediana")
    out.append(f"Depósito por día de estancia (mediana): de {ind.iloc[0]['deposito_por_dia_mediana']:.2f} ({ind.iloc[0]['department']}) "
               f"a {ind.iloc[-1]['deposito_por_dia_mediana']:.2f} ({ind.iloc[-1]['department']}); cociente descriptivo, no costo diario.")
    cx = tables["complexity"]
    n_scored = int(cx.loc[cx["clinical_complexity_score"] != ds.NO_DATA, "registros"].sum())
    out.append(f"Puntaje exploratorio de complejidad calculado para {n_scored} registros de {int(cx['registros'].sum())}; "
               "el resto queda sin puntaje (categorías sin peso definido).")
    return out
