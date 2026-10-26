"""Funciones de análisis descriptivo reutilizadas por el notebook 02 y su reporte TXT."""
from __future__ import annotations

import pandas as pd


def null_table(df: pd.DataFrame) -> pd.DataFrame:
    """Nulos por columna (cantidad y porcentaje)."""
    n = df.isna().sum()
    return pd.DataFrame({"nulos": n, "porcentaje": (n / len(df) * 100).round(2)})


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos descriptivos de las columnas numéricas (incluye mediana)."""
    num = df.select_dtypes(include="number")
    stats = num.describe().T
    stats["mediana"] = num.median()
    return stats.round(3)


def category_frequency(df: pd.DataFrame, column: str, dropna: bool = False) -> pd.DataFrame:
    """Frecuencia absoluta y porcentual de una variable categórica (nulos visibles por defecto)."""
    counts = df[column].value_counts(dropna=dropna, sort=False)
    out = pd.DataFrame({"registros": counts, "porcentaje": (counts / len(df) * 100).round(2)})
    out.index = out.index.astype(object).map(lambda v: "(nulo)" if pd.isna(v) else v)
    out.index.name = column
    return out


def group_summary(df: pd.DataFrame, by: str, targets: list[str]) -> pd.DataFrame:
    """Registros, media, mediana, desviación, mín. y máx. de ``targets`` por grupo de ``by``."""
    g = df.groupby(by, observed=True, dropna=False)[targets].agg(["mean", "median", "std", "min", "max"])
    g.insert(0, ("registros", ""), df.groupby(by, observed=True, dropna=False).size())
    g.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in g.columns]
    return g.round(2)
