"""Transformaciones del dataset de admisiones hospitalarias.

Cada paso es una función pura: recibe un DataFrame y devuelve ``(DataFrame, registro)``
sin modificar el original. El registro (dict) documenta las filas antes/después y el
resultado real de la operación, para mostrarlo en los notebooks y en los reportes.
Los notebooks 01 y 02 usan exactamente estas funciones (``apply_all``).
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

Step = tuple[pd.DataFrame, dict[str, Any]]

NUMERIC_COLUMNS = [
    "Available Extra Rooms in Hospital",
    "staff_available",
    "patientid",
    "Visitors with Patient",
    "Admission_Deposit",
    "Stay (in days)",
]
CATEGORICAL_COLUMNS = [
    "Department",
    "Ward_Facility_Code",
    "doctor_name",
    "Age",
    "gender",
    "Type of Admission",
    "Severity of Illness",
    "health_conditions",
    "Insurance",
]
SEVERITY_ORDER = ["Minor", "Moderate", "Extreme"]
# Grupos de personal: convención analítica propia (la ficha no define umbrales).
STAFF_BINS = [-1, 3, 6, float("inf")]
STAFF_LABELS = ["Bajo (0-3)", "Medio (4-6)", "Alto (7+)"]


def to_snake_case(name: str) -> str:
    """``"Stay (in days)"`` -> ``"stay_in_days"``."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.strip("_").lower()


def _record(name: str, description: str, before: pd.DataFrame, after: pd.DataFrame, **extra: Any) -> dict[str, Any]:
    return {
        "paso": name,
        "descripcion": description,
        "filas_antes": len(before),
        "filas_despues": len(after),
        "columnas_antes": before.shape[1],
        "columnas_despues": after.shape[1],
        **extra,
    }


def strip_text_values(df: pd.DataFrame) -> Step:
    """Quita espacios al inicio y al final de los valores de texto (nulos intactos)."""
    out = df.copy()
    changed: dict[str, int] = {}
    for col in out.select_dtypes(include=["object", "string"]).columns:
        stripped = out[col].str.strip()
        n = int((out[col].notna() & (out[col] != stripped)).sum())
        if n:
            changed[col] = n
        out[col] = stripped
    return out, _record(
        "strip_text_values",
        "Eliminación de espacios iniciales/finales en columnas de texto.",
        df, out, valores_modificados=sum(changed.values()), por_columna=changed,
    )


def convert_numeric_types(df: pd.DataFrame) -> Step:
    """Convierte las columnas numéricas con validación; reporta valores no convertibles."""
    out = df.copy()
    coerced: dict[str, int] = {}
    for col in NUMERIC_COLUMNS:
        if col not in out.columns:
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        n = int((converted.isna() & out[col].notna()).sum())
        if n:
            coerced[col] = n
        out[col] = converted
    for col in NUMERIC_COLUMNS:
        if col in out.columns and col != "Admission_Deposit" and out[col].notna().all():
            if (out[col] % 1 == 0).all():
                out[col] = out[col].astype("int64")
    negatives = {
        c: int((out[c] < 0).sum()) for c in NUMERIC_COLUMNS if c in out.columns and (out[c] < 0).any()
    }
    return out, _record(
        "convert_numeric_types",
        "Conversión validada a tipos numéricos (pd.to_numeric); valores no convertibles se reportan, no se eliminan.",
        df, out, valores_no_convertibles=sum(coerced.values()), no_convertibles_por_columna=coerced,
        valores_negativos_detectados=negatives,
    )


def drop_exact_duplicates(df: pd.DataFrame) -> Step:
    """Elimina filas idénticas en todas las columnas (``patientid`` no es único en el dataset)."""
    out = df.drop_duplicates().reset_index(drop=True)
    return out, _record(
        "drop_exact_duplicates",
        "Eliminación de registros exactamente duplicados (todas las columnas iguales).",
        df, out, duplicados_eliminados=len(df) - len(out),
    )


def normalize_column_names(df: pd.DataFrame) -> Step:
    """Renombra las columnas a ``snake_case`` conservando la correspondencia original."""
    mapping = {c: to_snake_case(c) for c in df.columns}
    out = df.rename(columns=mapping)
    return out, _record(
        "normalize_column_names",
        "Normalización de nombres de columnas a snake_case.",
        df, out, correspondencia=mapping,
    )


def standardize_categories(df: pd.DataFrame) -> Step:
    """Colapsa espacios internos y define tipos categóricos, con orden donde lo hay.

    No reescribe ni recodifica valores (p. ej. se conserva ``Asthama`` tal como está en
    la fuente) y los nulos de ``health_conditions`` se mantienen como nulos.
    ``severity_of_illness`` y ``age`` quedan como categorías ordenadas, no como números.
    """
    out = df.copy()
    changed = 0
    for col in [to_snake_case(c) for c in CATEGORICAL_COLUMNS]:
        if col not in out.columns:
            continue
        cleaned = out[col].astype("string").str.replace(r"\s+", " ", regex=True)
        changed += int((out[col].notna() & (out[col].astype("string") != cleaned)).sum())
        out[col] = cleaned.astype(object).where(cleaned.notna(), None)
    age_order = sorted(out["age"].dropna().unique(), key=lambda v: int(str(v).split("-")[0]))
    for col in [to_snake_case(c) for c in CATEGORICAL_COLUMNS]:
        if col == "severity_of_illness":
            out[col] = pd.Categorical(out[col], categories=SEVERITY_ORDER, ordered=True)
        elif col == "age":
            out[col] = pd.Categorical(out[col], categories=age_order, ordered=True)
        elif col in out.columns:
            out[col] = out[col].astype("category")
    unexpected = int(out["severity_of_illness"].isna().sum() - df["severity_of_illness"].isna().sum())
    return out, _record(
        "standardize_categories",
        "Espacios internos normalizados y columnas de texto convertidas a 'category' "
        "(severity_of_illness y age ordenadas); nulos conservados.",
        df, out, valores_modificados=changed, severidad_fuera_de_catalogo=unexpected,
    )


def add_staff_availability_group(df: pd.DataFrame) -> Step:
    """Enriquecimiento: agrupa ``staff_available`` en Bajo/Medio/Alto para comparar grupos."""
    out = df.copy()
    out["staff_availability_group"] = pd.cut(
        out["staff_available"], bins=STAFF_BINS, labels=STAFF_LABELS, ordered=True
    )
    return out, _record(
        "add_staff_availability_group",
        "Nueva columna staff_availability_group (Bajo 0-3, Medio 4-6, Alto 7+); umbrales definidos "
        "por el proyecto, no por la ficha del dataset.",
        df, out, distribucion={str(k): int(v) for k, v in out["staff_availability_group"].value_counts().sort_index().items()},
    )


STEPS = (
    strip_text_values,
    convert_numeric_types,
    drop_exact_duplicates,
    normalize_column_names,
    standardize_categories,
    add_staff_availability_group,
)


def apply_all(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Aplica todas las transformaciones en orden sobre TODAS las filas."""
    log: list[dict[str, Any]] = []
    for step in STEPS:
        df, rec = step(df)
        log.append(rec)
    return df, log


def describe_log(log: list[dict[str, Any]]) -> list[str]:
    """Texto breve por paso (para los reportes ``.txt``)."""
    return [
        f"{r['paso']}: {r['descripcion']} Filas {r['filas_antes']} -> {r['filas_despues']}; "
        f"columnas {r['columnas_antes']} -> {r['columnas_despues']}."
        for r in log
    ]
