"""Componentes visuales reutilizables de Streamlit (tarjetas, notas, avisos e interpretaciones)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.pregunta {border-left: 4px solid #0072B2; background: rgba(0,114,178,.07); padding: .8rem 1rem;
           border-radius: 6px; margin: .2rem 0 1rem; font-size: 1.02rem; line-height: 1.45;}
.pregunta span {display:block; font-size:.72rem; font-weight:700; letter-spacing:.06em;
                text-transform:uppercase; color:#005a8c; margin-bottom:.2rem;}
.universo {font-size:.85rem; color:#5f5e5a; border:1px dashed rgba(128,128,128,.45); border-radius:6px;
           padding:.35rem .7rem; margin:.2rem 0 .8rem;}
.universo.muestra {border-color:#D55E00; color:#8a3a00; background:rgba(213,94,0,.06);}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 8px; padding: .55rem .8rem;}
</style>
"""


def esc(text: str) -> str:
    """Streamlit interpreta ``$...$`` como LaTeX: se escapa el símbolo."""
    return str(text).replace("$", r"\$")


def question_box(qid: str, question: str) -> None:
    st.markdown(f'<div class="pregunta"><span>{qid}</span>{question}</div>', unsafe_allow_html=True)


def universe_note(text: str, sample: bool = False) -> None:
    """Aviso visible del universo de datos de la vista (dataset completo, filtrado o muestra)."""
    css = "universo muestra" if sample else "universo"
    icon = "🧪" if sample else "🗂️"
    st.markdown(f'<div class="{css}">{icon} {text}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, help_text: str | None = None) -> None:
    col.metric(label, value, help=help_text)


def interpretation(item: dict, expanded: bool = True) -> None:
    """Las cinco partes de la interpretación de un visual."""
    with st.expander("Interpretación", expanded=expanded):
        st.markdown(f"**Métrica y cálculo.** {esc(item['metrica'])}")
        st.markdown(f"**Patrón observado.** {esc(item['patron'])}")
        st.markdown(f"**Respuesta a la pregunta.** {esc(item['respuesta'])}")
        st.markdown(f"**Revisión operativa que podría apoyar.** {esc(item['decision'])}")
        st.markdown(f"**Limitaciones.** {esc(item['limitacion'])}")


def chart(fig) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


COLUMN_LABELS = {
    "department": "Departamento", "severity_of_illness": "Severidad", "staff_availability_group": "Grupo de personal",
    "staff_available": "Personal disponible", "stay_range": "Rango de estancia", "categoria": "Categoría",
    "clinical_complexity_score": "Puntaje de complejidad", "variable": "Variable", "registros": "Registros",
    "admisiones": "Admisiones", "pct_registros": "% de registros", "estancia_dias_validos": "Estancias válidas",
    "estancia_dias_media": "Estancia media (días)", "estancia_dias_mediana": "Estancia mediana (días)",
    "estancia_dias_desv_est": "Desv. estándar estancia (días)", "dias_paciente": "Días-paciente",
    "pct_dias_paciente": "% de días-paciente", "deposito_admision_media": "Depósito medio",
    "deposito_admision_mediana": "Depósito mediano", "deposito_admision_desv_est": "Desv. estándar depósito",
    "estancia_esperada_composicion": "Estancia esperada por composición (días)",
    "diferencia_dias": "Observada − esperada (días)", "deposito_por_dia_media": "Depósito por día (media)",
    "deposito_por_dia_mediana": "Depósito por día (mediana)", "visitas_por_dia_media": "Visitas por día (media)",
    "visitas_por_dia_mediana": "Visitas por día (mediana)", "pct_total": "% del total",
    "pct_clasificados": "% de clasificados", "validos": "Válidos", "nulos": "Nulos", "pct_nulos": "% de nulos",
    "tipo_final": "Tipo", "observaciones": "Observaciones", "media": "Media", "mediana": "Mediana",
    "desv_est": "Desv. estándar", "cv_pct": "CV (%)", "minimo": "Mínimo", "maximo": "Máximo", "q1": "Q1", "q3": "Q3",
    "ric": "RIC", "limite_inferior_tukey": "Valla inferior (Tukey)", "limite_superior_tukey": "Valla superior (Tukey)",
    "atipicos": "Atípicos", "atipicos_pct": "% de atípicos",
}


def table(df: pd.DataFrame, formats: dict[str, str] | None = None, height: int | None = None) -> None:
    """Tabla con etiquetas legibles y números en el formato regional del navegador.

    ``formats`` (columna -> «%d» o «%.Nf») fija los decimales; el resto de columnas decimales usa 2.
    """
    view = df.copy()
    cfg = {}
    for col in view.columns:
        label = COLUMN_LABELS.get(col, col)
        if pd.api.types.is_numeric_dtype(view[col]):
            fmt = (formats or {}).get(col, "%.2f" if pd.api.types.is_float_dtype(view[col]) else "%d")
            decimals = 0 if fmt == "%d" else int(fmt.split(".")[1][0])
            view[col] = view[col].round(decimals)
            cfg[col] = st.column_config.NumberColumn(label, format="localized")
        else:
            cfg[col] = st.column_config.Column(label)
    kwargs = {"height": height} if height else {}
    st.dataframe(view, hide_index=True, width="stretch", column_config=cfg, **kwargs)


def missing_data_panel(error: Exception, status: pd.DataFrame | None = None) -> None:
    """Explica qué archivo falta y cómo regenerarlo, sin inventar datos."""
    st.error("No se encontraron los resultados que necesita el dashboard.")
    st.markdown(esc(str(error)).replace("\n", "  \n"))
    st.markdown("Para regenerar todos los archivos (descarga el dataset completo desde Hugging Face):")
    st.code(config.REBUILD_COMMAND, language="bash")
    if status is not None:
        st.markdown("**Estado de los archivos esperados**")
        st.dataframe(status, hide_index=True, width="stretch")
