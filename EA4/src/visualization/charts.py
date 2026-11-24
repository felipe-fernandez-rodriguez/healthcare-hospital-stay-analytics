"""Figuras de Plotly del dashboard de EA4 (se ven igual en Jupyter y en Streamlit).

Cada función recibe una tabla ya calculada (dataset completo, cubo filtrado o muestra) y devuelve
una figura con título, subtítulo que identifica el universo de datos, ejes con unidades y formato
numérico en español. Ninguna función calcula métricas de negocio: solo las dibuja.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import NO_DATA, SEVERITY_ORDER

# Un color fijo por departamento (identidad) en todas las páginas; paleta apta para daltonismo.
DEPT_COLORS = {"gynecology": "#0072B2", "radiotherapy": "#E69F00", "anesthesia": "#009E73",
               "TB & Chest disease": "#CC79A7", "surgery": "#D55E00"}
SEVERITY_COLORS = {"Minor": "#9ecae1", "Moderate": "#4292c6", "Extreme": "#08519c"}
PRIMARY, ACCENT, MUTED = "#0072B2", "#D55E00", "#9ecae1"
TEXT, TEXT_2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
UNIVERSE_FULL = "Dataset completo"
UNIVERSE_SAMPLE = "Muestra sistemática de registros (no es el dataset completo)"


def dept_color(name: str) -> str:
    return DEPT_COLORS.get(str(name), "#7f7f7f")


def style(fig: go.Figure, title: str, subtitle: str = "", height: int = 380, legend: bool = False) -> go.Figure:
    """Estilo común (como ``_estilo`` del código de referencia): títulos, fuente, separadores en español."""
    text = f"{title}<br><sup style='color:{TEXT_2}'>{subtitle}</sup>" if subtitle else title
    fig.update_layout(
        title={"text": text, "x": 0, "xanchor": "left", "font": {"size": 15, "color": TEXT}},
        height=height, margin={"l": 10, "r": 10, "t": 70 if subtitle else 50, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": TEXT_2},
        separators=",.", showlegend=legend, hoverlabel={"bgcolor": "white", "font_size": 12},
        legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0, "title": None},
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    if legend:
        fig.update_layout(margin={"b": 70})
    return fig


# ------------------------------------------------------------------ distribuciones
def stay_distribution(dist: pd.DataFrame, subtitle: str) -> go.Figure:
    """Barras por valor de estancia (``stay_in_days``, ``registros``, ``pct``)."""
    fig = go.Figure(go.Bar(x=dist["stay_in_days"], y=dist["registros"], marker={"color": PRIMARY}, width=0.85,
                           customdata=dist["pct"],
                           hovertemplate="%{x} días<br>%{y:,.0f} registros (%{customdata:.2f} %)<extra></extra>"))
    fig.update_xaxes(title_text="Estancia (días)", dtick=4)
    fig.update_yaxes(title_text="Registros", rangemode="tozero", tickformat=",.0f")
    return style(fig, "Distribución de la estancia hospitalaria", subtitle, 360)


def histogram(hist: pd.DataFrame, variable: str, x_title: str, title: str, subtitle: str,
              marker: float | None = None, marker_label: str = "") -> go.Figure:
    """Histograma a partir de intervalos precalculados (``limite_inferior``, ``limite_superior``, ``registros``)."""
    h = hist[hist["variable"] == variable]
    centers = (h["limite_inferior"] + h["limite_superior"]) / 2
    widths = h["limite_superior"] - h["limite_inferior"]
    fig = go.Figure(go.Bar(x=centers, y=h["registros"], width=widths, marker={"color": PRIMARY, "line": {"width": 0.5, "color": "white"}},
                           customdata=np.stack([h["limite_inferior"], h["limite_superior"], h["pct"]], axis=-1),
                           hovertemplate="[%{customdata[0]:,.2f}; %{customdata[1]:,.2f}]<br>%{y:,.0f} registros"
                                         " (%{customdata[2]:.2f} %)<extra></extra>"))
    if marker is not None:
        fig.add_vline(x=marker, line_dash="dot", line_color=ACCENT, annotation_text=marker_label,
                      annotation_position="top right", annotation_font={"size": 11, "color": ACCENT})
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text="Registros", rangemode="tozero", tickformat=",.0f")
    return style(fig, title, subtitle, 340)


# ------------------------------------------------------------------ P1
def dept_bars(dept: pd.DataFrame, column: str, title: str, x_title: str, subtitle: str, decimals: int = 2) -> go.Figure:
    d = dept.sort_values(column)
    fig = go.Figure(go.Bar(
        x=d[column], y=d["department"], orientation="h", width=0.6, cliponaxis=False,
        marker={"color": [dept_color(x) for x in d["department"]]},
        text=[f"{v:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".") for v in d[column]],
        textposition="outside", customdata=d["registros"],
        hovertemplate="<b>%{y}</b><br>%{x:,." + str(decimals) + "f}<br>n = %{customdata:,.0f}<extra></extra>"))
    fig.update_xaxes(title_text=x_title, range=[0, d[column].max() * 1.18], tickformat=",.0f")
    fig.update_yaxes(title_text=None, gridcolor="rgba(0,0,0,0)")
    return style(fig, title, subtitle, 60 * len(d) + 120)


def admissions_vs_patient_days(dept: pd.DataFrame, subtitle: str) -> go.Figure:
    d = dept.sort_values("registros", ascending=False)
    fig = go.Figure([
        go.Bar(name="% de admisiones", x=d["department"], y=d["pct_registros"], marker_color=MUTED,
               text=[f"{v:.1f} %".replace(".", ",") for v in d["pct_registros"]], textposition="outside",
               hovertemplate="%{x}<br>%{y:.2f} % de las admisiones<extra></extra>"),
        go.Bar(name="% de días-paciente", x=d["department"], y=d["pct_dias_paciente"], marker_color=PRIMARY,
               text=[f"{v:.1f} %".replace(".", ",") for v in d["pct_dias_paciente"]], textposition="outside",
               hovertemplate="%{x}<br>%{y:.2f} % de los días-paciente<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(title_text="% del total", ticksuffix=" %", range=[0, max(d["pct_registros"].max(), d["pct_dias_paciente"].max()) * 1.2])
    fig.update_xaxes(title_text="Departamento")
    return style(fig, "Participación en admisiones frente a participación en días-paciente", subtitle, 400, legend=True)


def box_from_stats(box: pd.DataFrame, by: str, title: str, subtitle: str, y_title: str = "Estancia (días)") -> go.Figure:
    """Diagrama de caja con cuartiles y bigotes precalculados (exactos sobre el dataset completo)."""
    if by == "department":  # ordenado por mediana para comparar de un vistazo
        box = box.sort_values(["mediana", "media"])
    colors =[dept_color(x) if by == "department" else SEVERITY_COLORS.get(x, PRIMARY) for x in box[by]]
    fig = go.Figure()
    for (_, r), color in zip(box.iterrows(), colors):
        fig.add_trace(go.Box(x=[r[by]], q1=[r["q1"]], median=[r["mediana"]], q3=[r["q3"]],
                             lowerfence=[r["bigote_inferior"]], upperfence=[r["bigote_superior"]], mean=[r["media"]],
                             name=str(r[by]), marker_color=color, boxmean=True,
                             hovertemplate=f"<b>{r[by]}</b> (n = {int(r['n']):,})<extra></extra>".replace(",", ".")))
    fig.update_yaxes(title_text=y_title, rangemode="tozero")
    fig.update_xaxes(title_text=None)
    return style(fig, title, subtitle, 380)


# ------------------------------------------------------------------ P2
def severity_observed_expected(comp: pd.DataFrame, subtitle: str) -> go.Figure:
    order = [s for s in SEVERITY_ORDER if s in set(comp["severity_of_illness"])]
    c = comp.set_index("severity_of_illness").reindex(order)
    fig = go.Figure([
        go.Bar(name="Estancia media observada", x=order, y=c["estancia_observada_media"], marker_color=PRIMARY,
               text=[f"{v:.2f}".replace(".", ",") for v in c["estancia_observada_media"]], textposition="outside",
               hovertemplate="%{x}<br>Observada: %{y:.2f} días<extra></extra>"),
        go.Bar(name="Estancia esperada por composición de departamentos", x=order, y=c["estancia_esperada_composicion"],
               marker_color=MUTED, text=[f"{v:.2f}".replace(".", ",") for v in c["estancia_esperada_composicion"]],
               textposition="outside", hovertemplate="%{x}<br>Esperada: %{y:.2f} días<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(title_text="Estancia media (días)", rangemode="tozero",
                     range=[0, max(c["estancia_observada_media"].max(), c["estancia_esperada_composicion"].max()) * 1.2])
    fig.update_xaxes(title_text="Severidad de la enfermedad")
    return style(fig, "Estancia observada y esperada por la mezcla de departamentos", subtitle, 400, legend=True)


def category_dots(df: pd.DataFrame, by: str, column: str, title: str, y_title: str, subtitle: str,
                  order: list[str] | None = None, decimals: int = 2) -> go.Figure:
    """Puntos con etiqueta por categoría (útil cuando las diferencias son pequeñas y la escala no parte de cero)."""
    d = df.set_index(by).reindex(order) if order else df.set_index(by)
    fig = go.Figure(go.Scatter(x=d.index.astype(str), y=d[column], mode="markers+text", marker={"size": 14, "color": PRIMARY},
                               text=[f"{v:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".") for v in d[column]],
                               textposition="top center", customdata=d["registros"],
                               hovertemplate="%{x}<br>%{y:,." + str(decimals) + "f}<br>n = %{customdata:,.0f}<extra></extra>"))
    span = d[column].max() - d[column].min()
    pad = span * 0.6 if span else abs(d[column].max()) * 0.05 + 1
    tick = ",.2f" if span + 2 * pad < 2 else ",.1f" if span + 2 * pad < 20 else ",.0f"  # evita etiquetas repetidas
    fig.update_yaxes(title_text=y_title, range=[d[column].min() - pad, d[column].max() + pad], tickformat=tick)
    fig.update_xaxes(title_text=None, type="category")
    return style(fig, title, subtitle + " · eje vertical ampliado: no parte de cero", 340)


# ------------------------------------------------------------------ P3
def staff_level_bars(level: pd.DataFrame, subtitle: str) -> go.Figure:
    d = level.assign(_x=level["staff_available"].astype(float)).sort_values("_x")
    fig = go.Figure(go.Bar(x=d["staff_available"].astype(str), y=d["estancia_dias_media"], marker_color=PRIMARY, width=0.7,
                           customdata=np.stack([d["registros"], d["estancia_dias_mediana"]], axis=-1),
                           hovertemplate="Personal disponible: %{x}<br>Estancia media: %{y:.2f} días<br>Mediana: "
                                         "%{customdata[1]:.0f} días<br>n = %{customdata[0]:,.0f}<extra></extra>"))
    fig.update_xaxes(title_text="Personal disponible registrado (staff_available)", type="category")
    fig.update_yaxes(title_text="Estancia media (días)", rangemode="tozero")
    return style(fig, "Estancia media según el personal disponible (11 valores originales)", subtitle, 340)


def correlation_bars(corr: pd.DataFrame, subtitle: str) -> go.Figure:
    d = corr.assign(par=corr["variable_x"] + " · " + corr["variable_y"]).sort_values("rho_spearman")
    colors = [ACCENT if v < 0 else PRIMARY for v in d["rho_spearman"]]
    fig = go.Figure(go.Bar(x=d["rho_spearman"], y=d["par"], orientation="h", marker_color=colors, width=0.6,
                           text=[f"{v:.3f}".replace(".", ",") for v in d["rho_spearman"]], textposition="outside",
                           cliponaxis=False, customdata=d["n_pares_validos"],
                           hovertemplate="<b>%{y}</b><br>ρ = %{x:.4f}<br>n = %{customdata:,.0f} pares válidos<extra></extra>"))
    lim = max(0.2, d["rho_spearman"].abs().max() * 1.4)
    fig.add_vrect(x0=-0.1, x1=0.1, fillcolor=GRID, opacity=0.5, line_width=0, layer="below",
                  annotation_text="|ρ| < 0,1", annotation_position="top left", annotation_font={"size": 10, "color": TEXT_2})
    fig.update_xaxes(title_text="ρ de Spearman", range=[-lim, lim], zeroline=True, zerolinecolor=TEXT_2)
    fig.update_yaxes(title_text=None, gridcolor="rgba(0,0,0,0)")
    return style(fig, "Correlaciones de Spearman", subtitle, 50 * len(d) + 120)


# ------------------------------------------------------------------ P4
def heatmap(cross: pd.DataFrame, column: str, title: str, colorbar_title: str, subtitle: str, decimals: int = 2) -> go.Figure:
    depts = sorted(cross["department"].unique(), key=lambda d: cross.loc[cross["department"] == d, "registros"].sum(), reverse=True)
    sevs = [s for s in SEVERITY_ORDER if s in set(cross["severity_of_illness"])]
    z = cross.pivot_table(index="department", columns="severity_of_illness", values=column, aggfunc="first").reindex(index=depts, columns=sevs)
    n = cross.pivot_table(index="department", columns="severity_of_illness", values="registros", aggfunc="first").reindex(index=depts, columns=sevs)
    text = [[("sin datos" if pd.isna(v) else f"{v:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", "."))
             for v in row] for row in z.to_numpy()]
    fig = go.Figure(go.Heatmap(z=z.to_numpy(), x=sevs, y=depts, colorscale="Blues", xgap=2, ygap=2, text=text,
                               texttemplate="%{text}", customdata=n.fillna(0).to_numpy(), hoverongaps=False,
                               hovertemplate="<b>%{y}</b> · %{x}<br>" + colorbar_title + ": %{z:,." + str(decimals) +
                                             "f}<br>n = %{customdata:,.0f}<extra></extra>",
                               colorbar={"title": {"text": colorbar_title}, "thickness": 12, "len": 0.85}))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", title_text="Departamento")
    fig.update_xaxes(title_text="Severidad de la enfermedad")
    return style(fig, title, subtitle, 70 * len(depts) + 140)


# ------------------------------------------------------------------ P5
def stay_range_bars(sr: pd.DataFrame, subtitle: str) -> go.Figure:
    d = sr[~sr["stay_range"].isin(["Total"])]
    fig = go.Figure(go.Bar(x=d["stay_range"], y=d["pct_total"], marker_color=PRIMARY, width=0.6, cliponaxis=False,
                           text=[f"{v:.2f} %".replace(".", ",") for v in d["pct_total"]], textposition="outside",
                           customdata=d["registros"], hovertemplate="%{x}<br>%{y:.2f} % del total<br>n = %{customdata:,.0f}<extra></extra>"))
    fig.update_yaxes(title_text="% del total de registros", ticksuffix=" %", range=[0, max(d["pct_total"].max() * 1.18, 1)])
    fig.update_xaxes(title_text="Rango de estancia (umbrales del proyecto)")
    return style(fig, "Registros por rango de estancia", subtitle, 340)


def complexity_bars(cx: pd.DataFrame, subtitle: str) -> go.Figure:
    d = cx.copy()
    d["_orden"] = pd.to_numeric(d["clinical_complexity_score"], errors="coerce").fillna(np.inf)
    d = d.sort_values("_orden")
    labels = ["Sin puntaje" if v == NO_DATA else str(v).replace(".", ",") for v in d["clinical_complexity_score"]]
    colors = [MUTED if v == NO_DATA else PRIMARY for v in d["clinical_complexity_score"]]
    fig = go.Figure(go.Bar(x=labels, y=d["estancia_dias_media"], marker_color=colors, width=0.6,
                           customdata=np.stack([d["registros"], d["deposito_admision_media"]], axis=-1),
                           hovertemplate="Puntaje %{x}<br>Estancia media: %{y:.2f} días<br>Depósito medio: "
                                         "%{customdata[1]:,.2f}<br>n = %{customdata[0]:,.0f}<extra></extra>"))
    fig.update_xaxes(title_text="Puntaje exploratorio de complejidad (no validado clínicamente)", type="category")
    fig.update_yaxes(title_text="Estancia media (días)", rangemode="tozero")
    return style(fig, "Estancia media por puntaje exploratorio de complejidad", subtitle, 340)


def sample_box(sample: pd.DataFrame, column: str, y_title: str, title: str) -> go.Figure:
    """Diagrama de caja con registros individuales de la MUESTRA (identificada en el subtítulo)."""
    fig = go.Figure()
    order = sample.groupby("department")[column].median().sort_values().index
    for dept in order:
        values = sample.loc[sample["department"] == dept, column].dropna()
        fig.add_trace(go.Box(y=values, name=dept, marker_color=dept_color(dept), boxpoints=False,
                             hovertemplate=f"<b>{dept}</b> (n muestra = {len(values):,})<extra></extra>".replace(",", ".")))
    fig.update_yaxes(title_text=y_title, rangemode="tozero")
    return style(fig, title, f"{UNIVERSE_SAMPLE}: {len(sample):,} registros".replace(",", "."), 380)


# ------------------------------------------------------------------ PC2 y PC3
def eta_bars(eta: pd.DataFrame, variable: str, subtitle: str) -> go.Figure:
    d = eta[eta["variable"] == variable].sort_values("eta2")
    fig = go.Figure(go.Bar(x=d["eta2"] * 100, y=d["factor"], orientation="h", width=0.6, cliponaxis=False,
                           marker_color=[PRIMARY if v >= 0.10 else MUTED for v in d["eta2"]],
                           text=[f"{v * 100:.2f} %".replace(".", ",") for v in d["eta2"]], textposition="outside",
                           customdata=np.stack([d["n_grupos"], d["n_validos"]], axis=-1),
                           hovertemplate="<b>%{y}</b><br>η² = %{x:.3f} %<br>%{customdata[0]} grupos · n = %{customdata[1]:,.0f}<extra></extra>"))
    name = "la estancia" if variable == "stay_in_days" else "el depósito"
    fig.update_xaxes(title_text="η² (% de la variación total)", ticksuffix=" %", range=[0, max(d["eta2"].max() * 100 * 1.25, 1)])
    fig.update_yaxes(title_text=None, gridcolor="rgba(0,0,0,0)")
    return style(fig, f"Razón de varianza (η²) de {name} por factor", subtitle + " · barras oscuras: η² ≥ 10 %", 40 * len(d) + 130)


def pc2_bars(pc2: pd.DataFrame, variable: str, column: str, y_title: str, subtitle: str, max_categories: int = 12) -> go.Figure:
    d = pc2[pc2["variable"] == variable].copy()
    note = ""
    if len(d) > max_categories:  # se muestran las categorías con más registros
        d, note = d.sort_values("registros", ascending=False).head(max_categories), f" · {max_categories} categorías con más registros"
    try:
        d = d.assign(_k=d["categoria"].map(lambda v: float(str(v).split("-")[0]))).sort_values("_k")
    except ValueError:
        d = d.sort_values("registros", ascending=False)
    fig = go.Figure(go.Bar(x=d["categoria"].astype(str), y=d[column], marker_color=PRIMARY, width=0.65,
                           customdata=d["registros"],
                           hovertemplate="%{x}<br>%{y:,.2f}<br>n = %{customdata:,.0f}<extra></extra>"))
    fig.update_xaxes(title_text=variable, type="category")
    fig.update_yaxes(title_text=y_title, rangemode="tozero", tickformat=",.0f")
    return style(fig, f"{y_title} por {variable}", subtitle + note, 340)


def quality_nulls(quality: pd.DataFrame, subtitle: str) -> go.Figure:
    d = quality.sort_values(["pct_nulos", "variable"], ascending=[True, False])
    fig = go.Figure(go.Bar(x=d["pct_nulos"], y=d["variable"], orientation="h", width=0.65, cliponaxis=False,
                           marker_color=[ACCENT if v > 0 else GRID for v in d["pct_nulos"]],
                           text=[f"{v:.2f} %".replace(".", ",") if v > 0 else "0" for v in d["pct_nulos"]],
                           textposition="outside", customdata=d["nulos"],
                           hovertemplate="<b>%{y}</b><br>%{x:.2f} % nulos (%{customdata:,.0f} registros)<extra></extra>"))
    fig.update_xaxes(title_text="% de registros con valor nulo", ticksuffix=" %", range=[0, max(d["pct_nulos"].max() * 1.25, 1)])
    fig.update_yaxes(title_text=None, gridcolor="rgba(0,0,0,0)")
    return style(fig, "Valores faltantes por variable", subtitle, 22 * len(d) + 130)
