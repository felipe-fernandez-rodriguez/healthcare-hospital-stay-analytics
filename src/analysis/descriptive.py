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


_STAT_NAMES = {"mean": "media", "median": "mediana", "std": "desv_est", "min": "minimo", "max": "maximo"}
_TARGET_NAMES = {
    "stay_in_days": "estancia_dias",
    "admission_deposit": "deposito_admision",
    "deposit_per_stay_day": "deposito_por_dia",
    "visitors_per_day": "visitas_por_dia",
}
NO_DATA = "(sin dato)"


def group_metrics(
    df: pd.DataFrame,
    by: str | list[str],
    targets: tuple[str, ...] = ("stay_in_days", "admission_deposit"),
) -> pd.DataFrame:
    """Tabla de resultados por grupo con nombres comprensibles y tamaños de muestra.

    Reutiliza :func:`group_summary` y agrega, por variable, la cantidad de valores válidos
    (no nulos) que respaldan cada estadístico. Los grupos con clave nula se conservan como
    ``(sin dato)``. Las unidades van en los nombres: ``estancia_dias_*`` (días) y
    ``deposito_admision_*`` (unidad monetaria no documentada).
    """
    keys = [by] if isinstance(by, str) else list(by)
    table = group_summary(df, keys if len(keys) > 1 else keys[0], list(targets))
    grouped = df.groupby(keys if len(keys) > 1 else keys[0], observed=True, dropna=False)
    for t in targets:
        table[f"{t}_validos"] = grouped[t].count()
    table = table.rename(columns={"registros": "registros"})
    table.columns = [_rename(c) for c in table.columns]
    ordered = ["registros"]
    for t in targets:
        n = _TARGET_NAMES.get(t, t)
        ordered += [f"{n}_validos", *[f"{n}_{s}" for s in _STAT_NAMES.values()]]
    table = table[ordered].reset_index()
    for k in keys:
        labels = table[k].map(lambda v: f"{v:g}") if pd.api.types.is_numeric_dtype(table[k]) else table[k].astype(object)
        table[k] = labels.where(table[k].notna(), NO_DATA)  # etiquetas homogéneas (texto)
    return table


def _rename(col: str) -> str:
    for raw, nice in _TARGET_NAMES.items():
        if col.startswith(raw):
            rest = col[len(raw):]
            if rest == "_validos":
                return f"{nice}_validos"
            return f"{nice}_{_STAT_NAMES.get(rest.lstrip('_'), rest.lstrip('_'))}"
    return col


def stay_range_table(df: pd.DataFrame, column: str = "stay_range") -> pd.DataFrame:
    """Registros y porcentajes por rango de estancia, con denominadores explícitos.

    ``pct_total`` usa todos los registros; ``pct_clasificados`` solo los que recibieron rango
    (queda vacío en la fila de registros sin clasificar).
    """
    total = len(df)
    counts = df[column].value_counts(sort=False)
    classified = int(counts.sum())
    nan = float("nan")
    rows = [
        (str(label), int(n), round(n / total * 100, 2), round(n / classified * 100, 2) if classified else nan)
        for label, n in counts.items()
    ]
    rows.append(("Sin clasificar", total - classified, round((total - classified) / total * 100, 2), nan))
    rows.append(("Total", total, 100.0, nan))
    return pd.DataFrame(rows, columns=[column, "registros", "pct_total", "pct_clasificados"])
