"""Métricas de negocio de EA4 (solo pandas y numpy, sin descargas).

Dos familias de funciones:

* **Sobre registros** (``df`` enriquecido completo): días-paciente, estancia esperada por
  composición, razón de varianza (η²), Spearman, resumen de distribución e histogramas. Las usa
  el notebook 04 para generar las tablas oficiales. Las fórmulas de días-paciente, composición
  y η² son las documentadas en el Anexo A del informe del proyecto.
* **Sobre el cubo agregado** (``build_cube``): conteos por departamento × severidad × grupo de
  personal × tipo de admisión × valor de estancia, calculados con el dataset completo. Permiten
  que el dashboard aplique filtros y recalcule medias, medianas y desviaciones **exactas** sin
  cargar los 500.000 registros. El notebook comprueba que ambas familias dan el mismo resultado.

Regla de validez: una estancia es válida si es numérica y mayor que 0 (misma regla que
``src/data/enrichment.py`` de EA3); los nulos nunca se tratan como cero.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import CUBE_DIMENSIONS, NO_DATA, SEVERITY_ORDER, STAFF_GROUP_ORDER, STAY_RANGE_BINS, STAY_RANGE_ORDER

STAY, DEPOSIT = "stay_in_days", "admission_deposit"


# =========================================================================== utilidades
def valid_stay(series: pd.Series) -> pd.Series:
    """Estancia numérica y positiva; el resto queda como nulo."""
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s > 0)


def group_labels(series: pd.Series) -> pd.Series:
    """Etiquetas de grupo en texto, con los nulos como ``(sin dato)`` (no se descartan)."""
    labels = series.map(lambda v: f"{v:g}") if pd.api.types.is_numeric_dtype(series) else series.astype(object)
    return labels.where(series.notna(), NO_DATA)


def order_categories(values: Iterable[str], column: str) -> list[str]:
    """Orden de presentación documentado (severidad, personal, rango) o alfabético."""
    values = list(dict.fromkeys(str(v) for v in values))
    known = {"severity_of_illness": SEVERITY_ORDER, "staff_availability_group": STAFF_GROUP_ORDER,
             "stay_range": STAY_RANGE_ORDER}.get(column)
    if known:
        return [v for v in known if v in values] + sorted(v for v in values if v not in known)
    try:
        return sorted(values, key=float)
    except ValueError:
        return sorted(values, key=lambda v: (v == NO_DATA, v))


def stay_range_of(stay: pd.Series) -> pd.Series:
    """Rango de estancia con los umbrales de EA3; fuera de rango o inválido queda nulo."""
    s = valid_stay(stay)
    return pd.cut(s.where(s >= 1), bins=STAY_RANGE_BINS, labels=STAY_RANGE_ORDER).astype(object)


# =========================================================================== métricas sobre registros
def patient_days_share(df: pd.DataFrame, by: str = "department") -> pd.DataFrame:
    """Admisiones y días-paciente por grupo, con sus participaciones.

    ``dias_paciente`` = suma de las estancias válidas del grupo.
    ``pct_dias_paciente`` = dias_paciente del grupo / dias_paciente totales × 100.
    ``pct_admisiones`` usa como denominador todas las admisiones del DataFrame.
    """
    stay = valid_stay(df[STAY])
    key = group_labels(df[by])
    g = stay.groupby(key)
    out = pd.DataFrame({
        "admisiones": key.value_counts(),
        "estancia_validos": g.count(),
        "dias_paciente": g.sum(),
        "estancia_media": g.mean(),
        "estancia_desv_est": g.std(),
    })
    out["pct_admisiones"] = out["admisiones"] / out["admisiones"].sum() * 100
    out["pct_dias_paciente"] = out["dias_paciente"] / out["dias_paciente"].sum() * 100
    out["cv_estancia_pct"] = out["estancia_desv_est"] / out["estancia_media"] * 100
    out = out.reindex(order_categories(out.index, by))
    out.index.name = by
    cols = ["admisiones", "pct_admisiones", "estancia_validos", "dias_paciente", "pct_dias_paciente",
            "estancia_media", "estancia_desv_est", "cv_estancia_pct"]
    return out[cols].reset_index()


def expected_stay_by_composition(df: pd.DataFrame, factor: str = "severity_of_illness",
                                 composition: str = "department") -> pd.DataFrame:
    """Estancia observada frente a la esperada si solo importara la composición por departamento.

    Para cada registro con estancia válida, la estancia esperada es la media de su departamento
    (calculada con todas las estancias válidas). La esperada del grupo es la media de esos valores:
    Σ_d n_{g,d} · media_d / n_g. ``diferencia_dias`` = observada − esperada. Es una comparación
    descriptiva: no aísla un efecto causal del factor.
    """
    stay = valid_stay(df[STAY])
    comp = group_labels(df[composition])
    comp_mean = stay.groupby(comp).mean()
    expected = comp.map(comp_mean).where(stay.notna())
    key = group_labels(df[factor])
    out = pd.DataFrame({
        "registros": key.value_counts(),
        "estancia_validos": stay.groupby(key).count(),
        "estancia_observada_media": stay.groupby(key).mean(),
        "estancia_esperada_composicion": expected.groupby(key).mean(),
    })
    out["pct_registros"] = out["registros"] / out["registros"].sum() * 100
    out["diferencia_dias"] = out["estancia_observada_media"] - out["estancia_esperada_composicion"]
    out["cobertura_pct"] = expected.notna().groupby(key).sum() / out["estancia_validos"] * 100
    out = out.reindex(order_categories(out.index, factor))
    out.index.name = factor
    return out[["registros", "pct_registros", "estancia_validos", "estancia_observada_media",
                "estancia_esperada_composicion", "diferencia_dias", "cobertura_pct"]].reset_index()


def eta_squared(df: pd.DataFrame, factor: str, target: str) -> dict:
    """Razón de varianza η² = SS_entre / SS_total de ``target`` según ``factor``.

    SS_entre = Σ_g n_g (media_g − media)²; SS_total = Σ_i (y_i − media)². Solo entran los registros
    con ``target`` válido; los nulos del factor forman su propio grupo ``(sin dato)``.
    Vale 0 si todos los grupos tienen la misma media y 1 si el factor separa por completo los valores.
    """
    y = valid_stay(df[target]) if target == STAY else pd.to_numeric(df[target], errors="coerce")
    mask = y.notna()
    y = y[mask]
    key = group_labels(df.loc[mask, factor])
    grand = y.mean()
    ss_total = float(((y - grand) ** 2).sum())
    g = y.groupby(key)
    ss_between = float((g.size() * (g.mean() - grand) ** 2).sum())
    return {"factor": factor, "variable": target,
            "eta2": ss_between / ss_total if ss_total > 0 else float("nan"),
            "n_validos": int(mask.sum()), "n_grupos": int(key.nunique())}


def eta_squared_table(df: pd.DataFrame, factors: Iterable[str], targets: Iterable[str] = (STAY, DEPOSIT)) -> pd.DataFrame:
    rows = [eta_squared(df, f, t) for t in targets for f in factors]
    out = pd.DataFrame(rows)
    out["eta2_pct"] = out["eta2"] * 100
    return out.sort_values(["variable", "eta2"], ascending=[False, False]).reset_index(drop=True)


def spearman(df: pd.DataFrame, x: str, y: str) -> dict:
    """ρ de Spearman con pares válidos (ambos valores no nulos); los empates reciben rango promedio."""
    a, b = pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce")
    mask = a.notna() & b.notna()
    rho = a[mask].rank().corr(b[mask].rank()) if mask.sum() > 2 else float("nan")
    return {"variable_x": x, "variable_y": y, "rho_spearman": float(rho), "n_pares_validos": int(mask.sum()),
            "metodo": "Pearson sobre rangos promedio (empates), pares completos"}


def spearman_table(df: pd.DataFrame, pairs: Iterable[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([spearman(df, x, y) for x, y in pairs])


def distribution_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Medidas de posición y dispersión, y atípicos por la regla de Tukey (1,5 × RIC), sin eliminarlos."""
    rows = []
    for col in columns:
        s = pd.to_numeric(df[col], errors="coerce")
        v = s.dropna()
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((v < lo) | (v > hi)).sum())
        rows.append({
            "variable": col, "registros": len(s), "validos": len(v), "nulos": int(s.isna().sum()),
            # El CV solo tiene sentido en variables no negativas con media positiva (no en diferencias).
            "media": v.mean(), "mediana": v.median(), "desv_est": v.std(),
            "cv_pct": v.std() / v.mean() * 100 if v.min() >= 0 and v.mean() > 0 else float("nan"),
            "minimo": v.min(), "q1": q1, "q3": q3, "ric": iqr, "maximo": v.max(),
            "limite_inferior_tukey": lo, "limite_superior_tukey": hi,
            "atipicos": n_out, "atipicos_pct": n_out / len(v) * 100 if len(v) else float("nan"),
        })
    return pd.DataFrame(rows)


def histogram_table(series: pd.Series, variable: str, bins: int = 40, integer: bool = False) -> pd.DataFrame:
    """Frecuencias por intervalo con todos los valores válidos (``pct`` sobre los válidos).

    ``integer=True`` usa un intervalo por valor entero (p. ej. días de estancia), incluidos los ceros.
    En el resto, ``np.histogram`` con intervalos de igual ancho entre el mínimo y el máximo
    (el último intervalo incluye el máximo).
    """
    v = pd.to_numeric(series, errors="coerce").dropna()
    if integer:
        values = np.arange(int(v.min()), int(v.max()) + 1)
        counts = v.astype(int).value_counts().reindex(values, fill_value=0).to_numpy()
        left, right = values - 0.5, values + 0.5
    else:
        counts, edges = np.histogram(v, bins=bins)
        left, right = edges[:-1], edges[1:]
    out = pd.DataFrame({"variable": variable, "limite_inferior": left, "limite_superior": right, "registros": counts})
    out["pct"] = out["registros"] / out["registros"].sum() * 100
    return out


# =========================================================================== cubo agregado
def build_cube(df: pd.DataFrame, dims: Iterable[str] = CUBE_DIMENSIONS) -> pd.DataFrame:
    """Conteos y sumas por combinación de dimensiones y valor de estancia (dataset completo).

    Cada fila: dimensiones, ``stay_in_days`` (nulo si inválida), ``registros``, ``depositos_validos``,
    ``suma_deposito`` y ``suma_deposito_cuadrado``. Con esto se recalculan exactamente conteos,
    medias, medianas y cuantiles de estancia, y media y desviación del depósito, bajo cualquier filtro
    de esas dimensiones. La mediana del depósito NO es recuperable desde el cubo.
    """
    dims = list(dims)
    stay = valid_stay(df[STAY])
    if not (stay.dropna() % 1 == 0).all():
        raise ValueError("El cubo requiere estancias enteras para calcular medianas exactas.")
    d = pd.DataFrame({c: group_labels(df[c]) for c in dims})
    d[STAY] = stay
    dep = pd.to_numeric(df[DEPOSIT], errors="coerce")
    d["dep"], d["dep_ok"], d["dep_sq"] = dep, dep.notna().astype("int64"), dep ** 2
    g = d.groupby(dims + [STAY], dropna=False)
    cube = pd.DataFrame({"registros": g.size(), "depositos_validos": g["dep_ok"].sum(),
                         "suma_deposito": g["dep"].sum(), "suma_deposito_cuadrado": g["dep_sq"].sum()}).reset_index()
    return cube


def filter_cube(cube: pd.DataFrame, selections: dict[str, list[str]] | None = None,
                stay_ranges: list[str] | None = None) -> pd.DataFrame:
    """Filtra el cubo. Una lista vacía o ``None`` no filtra esa dimensión."""
    out = cube
    for col, values in (selections or {}).items():
        if values:
            out = out[out[col].astype(str).isin([str(v) for v in values])]
    if stay_ranges:
        out = out[stay_range_of(out[STAY]).isin(stay_ranges)]
    return out


def _value_at(values: np.ndarray, cum: np.ndarray, pos: int) -> float:
    return float(values[np.searchsorted(cum, pos, side="right")])


def weighted_quantile(values: np.ndarray, counts: np.ndarray, q: float) -> float:
    """Cuantil con interpolación lineal de pandas/numpy sobre datos expresados como (valor, frecuencia)."""
    order = np.argsort(values)
    values, counts = np.asarray(values, float)[order], np.asarray(counts, float)[order]
    n = counts.sum()
    if n == 0:
        return float("nan")
    cum = np.cumsum(counts)
    pos = q * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    v_lo, v_hi = _value_at(values, cum, lo), _value_at(values, cum, hi)
    return v_lo + (v_hi - v_lo) * (pos - lo)


def _stay_stats(group: pd.DataFrame) -> dict:
    ok = group[group[STAY].notna()]
    w, x = ok["registros"].to_numpy(float), ok[STAY].to_numpy(float)
    n = w.sum()
    mean = (w * x).sum() / n if n else float("nan")
    var = (w * (x - mean) ** 2).sum() / (n - 1) if n > 1 else float("nan")
    dep_n = group["depositos_validos"].sum()
    dep_mean = group["suma_deposito"].sum() / dep_n if dep_n else float("nan")
    dep_var = ((group["suma_deposito_cuadrado"].sum() - dep_n * dep_mean ** 2) / (dep_n - 1)) if dep_n > 1 else float("nan")
    return {
        "registros": int(group["registros"].sum()),
        "estancia_dias_validos": int(n),
        "estancia_dias_media": mean,
        "estancia_dias_mediana": weighted_quantile(x, w, 0.5),
        "estancia_dias_desv_est": math.sqrt(var) if var == var else float("nan"),
        "estancia_dias_minimo": float(x.min()) if n else float("nan"),
        "estancia_dias_maximo": float(x.max()) if n else float("nan"),
        "dias_paciente": float((w * x).sum()),
        "deposito_admision_validos": int(dep_n),
        "deposito_admision_media": dep_mean,
        "deposito_admision_desv_est": math.sqrt(max(dep_var, 0.0)) if dep_var == dep_var else float("nan"),
    }


def cube_group_metrics(cube: pd.DataFrame, by: str | list[str] | None = None) -> pd.DataFrame:
    """Métricas por grupo desde el cubo; participaciones respecto del universo filtrado."""
    if by is None:
        out = pd.DataFrame([_stay_stats(cube)])
    else:
        keys = [by] if isinstance(by, str) else list(by)
        rows = [{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **_stay_stats(g)}
                for k, g in cube.groupby(keys, sort=False)]
        out = pd.DataFrame(rows)
        for k in keys:
            out[k] = pd.Categorical(out[k], categories=order_categories(out[k], k), ordered=True)
        out = out.sort_values(keys).reset_index(drop=True)
        for k in keys:
            out[k] = out[k].astype(str)
    total_reg, total_dp = out["registros"].sum(), out["dias_paciente"].sum()
    out["pct_registros"] = out["registros"] / total_reg * 100 if total_reg else float("nan")
    out["pct_dias_paciente"] = out["dias_paciente"] / total_dp * 100 if total_dp else float("nan")
    return out


def cube_box_stats(cube: pd.DataFrame, by: str) -> pd.DataFrame:
    """Cuartiles y bigotes de Tukey de la estancia por grupo (exactos, dataset completo)."""
    rows = []
    for k, g in cube[cube[STAY].notna()].groupby(by, sort=False):
        agg = g.groupby(STAY)["registros"].sum()
        x, w = agg.index.to_numpy(float), agg.to_numpy(float)
        q1, med, q3 = (weighted_quantile(x, w, q) for q in (0.25, 0.5, 0.75))
        iqr = q3 - q1
        inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
        rows.append({by: str(k), "n": int(w.sum()), "q1": q1, "mediana": med, "q3": q3,
                     "bigote_inferior": float(inside.min()), "bigote_superior": float(inside.max()),
                     "media": float((x * w).sum() / w.sum())})
    out = pd.DataFrame(rows)
    out[by] = pd.Categorical(out[by], categories=order_categories(out[by], by), ordered=True)
    return out.sort_values(by).assign(**{by: lambda d: d[by].astype(str)}).reset_index(drop=True)


def cube_stay_distribution(cube: pd.DataFrame) -> pd.DataFrame:
    """Registros por valor de estancia (días) dentro del universo filtrado."""
    agg = cube[cube[STAY].notna()].groupby(STAY)["registros"].sum()
    out = agg.rename("registros").reset_index()
    out["pct"] = out["registros"] / out["registros"].sum() * 100 if len(out) else []
    return out


def cube_stay_range_table(cube: pd.DataFrame) -> pd.DataFrame:
    """Misma tabla que ``stay_range_table`` de EA3 (denominadores explícitos), desde el cubo filtrado."""
    total = int(cube["registros"].sum())
    ranges = stay_range_of(cube[STAY])
    counts = cube.groupby(ranges)["registros"].sum().reindex(STAY_RANGE_ORDER, fill_value=0)
    classified = int(counts.sum())
    nan = float("nan")
    rows = [(lab, int(n), n / total * 100 if total else nan, n / classified * 100 if classified else nan)
            for lab, n in counts.items()]
    rows.append(("Sin clasificar", total - classified, (total - classified) / total * 100 if total else nan, nan))
    rows.append(("Total", total, 100.0 if total else nan, nan))
    return pd.DataFrame(rows, columns=["stay_range", "registros", "pct_total", "pct_clasificados"])


def cube_expected_stay(cube: pd.DataFrame, factor: str = "severity_of_illness", composition: str = "department") -> pd.DataFrame:
    """Misma definición que :func:`expected_stay_by_composition`, calculada desde el cubo filtrado."""
    ok = cube[cube[STAY].notna()].copy()
    ok["dp"] = ok["registros"] * ok[STAY]
    comp = ok.groupby(composition)[["dp", "registros"]].sum()
    comp_mean = comp["dp"] / comp["registros"]
    ok["esperada_dp"] = ok["registros"] * ok[composition].map(comp_mean)
    g = ok.groupby(factor)[["registros", "dp", "esperada_dp"]].sum()
    out = pd.DataFrame({
        "estancia_validos": g["registros"],
        "estancia_observada_media": g["dp"] / g["registros"],
        "estancia_esperada_composicion": g["esperada_dp"] / g["registros"],
    })
    out["diferencia_dias"] = out["estancia_observada_media"] - out["estancia_esperada_composicion"]
    out = out.reindex(order_categories(out.index, factor))
    out.index.name = factor
    return out.reset_index()


def cube_kpis(cube: pd.DataFrame) -> dict:
    """Indicadores generales del universo (filtrado o completo)."""
    s = _stay_stats(cube)
    s["departamentos"] = int(cube.loc[cube["registros"] > 0, "department"].nunique()) if "department" in cube else 0
    return s
