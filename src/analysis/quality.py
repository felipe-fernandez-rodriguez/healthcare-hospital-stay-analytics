"""Controles de calidad y comparación antes/después (EA3)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.transformations import to_snake_case


def compare_dataframes(original: pd.DataFrame, final: pd.DataFrame) -> dict[str, Any]:
    """Compara el DataFrame original con el final (filas, columnas, tipos, nulos, duplicados).

    Las columnas se emparejan por la correspondencia ``snake_case`` de sus nombres originales.
    """
    mapping = {c: to_snake_case(c) for c in original.columns}
    renamed = {o: n for o, n in mapping.items() if o != n}
    new_cols = [c for c in final.columns if c not in mapping.values()]
    rows = []
    for orig, new in mapping.items():
        if new not in final.columns:
            continue
        rows.append({
            "columna_original": orig,
            "columna_final": new,
            "tipo_antes": str(original[orig].dtype),
            "tipo_despues": str(final[new].dtype),
            "nulos_antes": int(original[orig].isna().sum()),
            "pct_nulos_antes": round(original[orig].isna().mean() * 100, 2),
            "nulos_despues": int(final[new].isna().sum()),
            "pct_nulos_despues": round(final[new].isna().mean() * 100, 2),
        })
    for c in new_cols:
        rows.append({
            "columna_original": "(nueva)", "columna_final": c, "tipo_antes": "-", "tipo_despues": str(final[c].dtype),
            "nulos_antes": None, "pct_nulos_antes": None,
            "nulos_despues": int(final[c].isna().sum()), "pct_nulos_despues": round(final[c].isna().mean() * 100, 2),
        })
    dims = pd.DataFrame(
        {"antes": [len(original), original.shape[1], int(original.duplicated().sum())],
         "despues": [len(final), final.shape[1], int(final.duplicated().sum())]},
        index=["filas", "columnas", "duplicados_exactos"],
    )
    return {
        "dimensiones": dims,
        "columnas": pd.DataFrame(rows),
        "columnas_renombradas": renamed,
        "columnas_nuevas": new_cols,
        "columnas_eliminadas": [o for o, n in mapping.items() if n not in final.columns],
    }


def quality_table(df: pd.DataFrame, notes: dict[str, str] | None = None) -> pd.DataFrame:
    """Tabla de calidad por variable: tipo final, registros, nulos, % y valores válidos."""
    notes = notes or {}
    n = df.isna().sum()
    return pd.DataFrame({
        "variable": df.columns,
        "tipo_final": [str(t) for t in df.dtypes],
        "registros": len(df),
        "nulos": n.values,
        "pct_nulos": (n / len(df) * 100).round(2).values,
        "validos": (len(df) - n).values,
        "observaciones": [notes.get(c, "Sin observaciones") for c in df.columns],
    })
