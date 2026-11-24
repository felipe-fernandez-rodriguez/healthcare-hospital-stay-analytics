"""Dashboard de Streamlit de EA4: estancia hospitalaria y depósito de admisión.

La clase :class:`HospitalDashboard` (patrón ``DashboardStreamlit`` del código de referencia):

1. recibe las tablas que generó el notebook 04 con el dataset completo (no recalcula EA3);
2. aplica los filtros sobre el **cubo agregado** del dataset completo, de modo que las vistas
   filtradas siguen siendo exactas y nunca se mezclan con cifras de la muestra;
3. construye cada visual con :mod:`charts` y redacta su interpretación con :mod:`insights`;
4. compone la app en ``render()``, que se llama desde ``EA4/app.py``.

Uso en Jupyter::

    dash = HospitalDashboard.from_files()
    dash.official_insights()["P1"]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .. import config
from ..data import data_loader as dl
from ..formatting import days, num, pct
from ..metrics import business_metrics as bm
from ..metrics import insights as ins
from . import charts as ch

PAGES: dict[str, str] = {
    "inicio": "🏠 Inicio y contexto",
    "pc1": "📊 Indicadores generales · PC1",
    "p1": "🏥 P1 · Departamentos",
    "p2": "🩺 P2 · Severidad",
    "p3": "👥 P3 · Personal disponible",
    "p4": "🧩 P4 · Departamento × severidad",
    "p5": "🧮 P5 · Indicadores derivados",
    "pc2": "🔗 PC2 · Otras variables",
    "pc3": "🧪 PC3 · Calidad de los datos",
    "fin": "✅ Conclusiones",
}
QUESTION_PAGE = {"PC1": "pc1", "P1": "p1", "P2": "p2", "P3": "p3", "P4": "p4", "P5": "p5", "PC2": "pc2", "PC3": "pc3"}
FILTERS = {"department": "Departamento", "severity_of_illness": "Severidad de la enfermedad",
           "staff_availability_group": "Disponibilidad de personal", "type_of_admission": "Tipo de admisión"}
INTEREST_VARS = ["stay_in_days", "admission_deposit", "department", "severity_of_illness", "staff_available",
                 "staff_availability_group"]
METRIC_LABELS = {"estancia_dias_media": "Estancia media (días)", "estancia_dias_mediana": "Estancia mediana (días)",
                 "deposito_admision_media": "Depósito medio", "registros": "Registros"}
FMT_2 = "%.2f"


@dataclass
class Filters:
    """Selección de la barra lateral; una lista vacía significa «todas las categorías»."""

    selections: dict[str, list[str]] = field(default_factory=dict)
    stay_ranges: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return any(self.selections.values()) or bool(self.stay_ranges)

    def describe(self) -> str:
        parts = [f"{FILTERS[k]}: {', '.join(v)}" for k, v in self.selections.items() if v]
        if self.stay_ranges:
            parts.append(f"Rango de estancia: {', '.join(self.stay_ranges)}")
        return "; ".join(parts) if parts else "sin filtros"


class HospitalDashboard:
    """Dashboard de métricas hospitalarias: vistas calculadas (testeables) + app de Streamlit."""

    TITLE = "Estancia hospitalaria y factores operativos en admisiones médicas"

    def __init__(self, tables: dict[str, pd.DataFrame], metadata: dict | None = None, sample: pd.DataFrame | None = None):
        missing = [k for k in config.TABLE_FILES if k not in tables]
        if missing:
            raise KeyError(f"Faltan tablas para el dashboard: {missing}")
        self.t = tables
        self.meta = metadata or {}
        self.sample = sample
        self.cube = tables["filter_cube"]
        self.total_records = int(self.cube["registros"].sum())

    @classmethod
    def from_files(cls, with_sample: bool = True) -> "HospitalDashboard":
        """Lee los archivos de EA4 (lanza ``DataUnavailableError`` si falta alguna tabla)."""
        tables = dl.load_all_tables()
        try:
            meta = dl.load_metadata()
        except dl.DataUnavailableError:
            meta = {}
        sample = None
        if with_sample:
            try:
                sample = dl.load_sample()
            except dl.DataUnavailableError:
                sample = None
        return cls(tables, meta, sample)

    # ================================================================== vistas (sin Streamlit)
    def filter_options(self) -> dict[str, list[str]]:
        return {c: bm.order_categories(self.cube.loc[self.cube[c] != config.NO_DATA, c].unique(), c) for c in FILTERS}

    def filtered_cube(self, f: Filters | None = None) -> pd.DataFrame:
        f = f or Filters()
        return bm.filter_cube(self.cube, f.selections, f.stay_ranges)

    def filtered_sample(self, f: Filters | None = None) -> pd.DataFrame | None:
        """Muestra con los mismos filtros (para vistas de registros individuales, siempre rotuladas)."""
        if self.sample is None:
            return None
        s = self.sample
        f = f or Filters()
        for col, values in f.selections.items():
            if values:
                s = s[s[col].astype(str).isin(values)]
        if f.stay_ranges:
            s = s[s["stay_range"].astype(str).isin(f.stay_ranges)]
        return s

    def dept_view(self, cube: pd.DataFrame, filtered: bool) -> pd.DataFrame:
        """Métricas por departamento; la mediana del depósito solo existe en la tabla oficial (sin filtros)."""
        d = bm.cube_group_metrics(cube, "department")
        official = self.t["department"].set_index("department")["deposito_admision_mediana"]
        d["deposito_admision_mediana"] = float("nan") if filtered else d["department"].map(official)
        return d

    def severity_view(self, cube: pd.DataFrame, filtered: bool) -> pd.DataFrame:
        d = bm.cube_group_metrics(cube, "severity_of_illness")
        official = self.t["severity"].set_index("severity_of_illness")["deposito_admision_mediana"]
        d["deposito_admision_mediana"] = float("nan") if filtered else d["severity_of_illness"].map(official)
        return d

    def official_insights(self) -> dict[str, dict]:
        """Interpretaciones del dataset completo (sin filtros).

        Usan las mismas vistas que las páginas sin filtros (cubo completo, sin redondeo intermedio),
        de modo que las cifras de «Conclusiones» coinciden con las de cada pregunta.
        """
        cube = self.cube
        return {
            "PC1": ins.pc1_distribution(self.t["distribution"]),
            "P1": ins.p1_departments(self.dept_view(cube, filtered=False)),
            "P2": ins.p2_severity(self.severity_view(cube, filtered=False), bm.cube_expected_stay(cube)),
            "P3": ins.p3_staff(bm.cube_group_metrics(cube, "staff_availability_group"), self.t["correlations"]),
            "P4": ins.p4_cross(bm.cube_group_metrics(cube, ["department", "severity_of_illness"])),
            "P5": ins.p5_indicators(bm.cube_stay_range_table(cube), self.t["complexity"], self.t["indicators"]),
            "PC2": ins.pc2_associations(self.t["eta_squared"], self.t["correlations"]),
            "PC3": ins.pc3_quality(self.t["data_quality"]),
        }

    def universe_text(self, cube: pd.DataFrame, f: Filters) -> str:
        n = int(cube["registros"].sum())
        if not f.active:
            return f"Dataset completo: {num(n)} registros válidos (calculado sobre todos los registros, sin filtros)."
        return (f"Dataset completo filtrado ({f.describe()}): {num(n)} de {num(self.total_records)} registros "
                f"({pct(n / self.total_records * 100)}). Cifras exactas calculadas desde el cubo agregado, no desde la muestra.")

    # ================================================================== app
    def render(self) -> None:
        """Arma la app (la configuración de página la fija ``EA4/app.py``)."""
        import streamlit as st

        from . import components as ui

        st.markdown(ui.CSS, unsafe_allow_html=True)
        self._ui, self._st = ui, st
        options = self.filter_options()
        with st.sidebar:
            st.header("Navegación")
            page = st.radio("Sección", list(PAGES), format_func=PAGES.get, key="pagina", label_visibility="collapsed")
            st.divider()
            st.header("Filtros")
            st.caption("Recalculan, con el dataset completo, las vistas de PC1 y P1 a P5 que tienen esas dimensiones. "
                       "Las tablas que no las tienen se muestran sin filtrar y lo indican.")
            selections = {c: st.multiselect(label, options[c], default=[], placeholder="Todas", key=f"f_{c}")
                          for c, label in FILTERS.items()}
            ranges = st.multiselect("Rango de estancia", config.STAY_RANGE_ORDER, default=[], placeholder="Todos", key="f_stay")
            st.divider()
            st.caption(f"Datos: [{config.DATASET_ID}]({config.DATASET_URL}) (Hugging Face). Análisis descriptivo; "
                       "escenario institucional hipotético.")
        f = Filters(selections, ranges)
        cube = self.filtered_cube(f)
        st.title(self.TITLE)
        if page not in ("inicio", "pc2", "pc3", "fin") and cube["registros"].sum() == 0:
            st.warning("Ningún registro cumple los filtros seleccionados. Amplíe la selección en la barra lateral.")
            return
        getattr(self, f"_page_{page}")(f, cube)

    # ------------------------------------------------------------------ páginas
    def _page_inicio(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("Pregunta principal de negocio", ins.MAIN_QUESTION)
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.subheader("Necesidad analítica")
            st.markdown(
                "En un **escenario hipotético**, la Dirección de Gestión Operativa y Control de Camas de un centro "
                "hospitalario necesita caracterizar la **duración de la estancia** y el **depósito de admisión** según "
                "factores clínicos y operativos, para orientar revisiones sobre la carga de camas por servicio.")
            st.subheader("Método")
            st.markdown(
                "1. **EA1-EA3:** descarga del dataset completo desde Hugging Face, transformaciones de calidad y "
                "variables derivadas (código de `src/`).\n"
                "2. **Notebook 04:** reutiliza ese código, calcula las métricas con todos los registros y exporta "
                "tablas agregadas, un cubo para los filtros y una muestra sistemática.\n"
                "3. **Esta app:** lee esas tablas (no descarga datos) y presenta cada pregunta con su métrica, su "
                "visual y su interpretación.")
        with c2:
            st.subheader("Datos")
            gen = self.meta.get("generado")
            n_an = self.meta.get("filas_analizadas", self.total_records)
            st.markdown(
                f"- **Dataset:** [{config.DATASET_ID}]({config.DATASET_URL})\n"
                f"- **Configuración / split:** {self.meta.get('config', 'n/d')} / {self.meta.get('split', 'n/d')}\n"
                f"- **Registros analizados:** {num(n_an)}\n"
                f"- **Columnas tras el enriquecimiento:** {self.meta.get('columnas_enriquecidas', 'n/d')}\n"
                f"- **Último procesamiento:** {gen.replace('T', ' ') if gen else 'no disponible (faltan los metadatos)'}\n"
                f"- **Muestra para vistas de registros:** {num(self.meta.get('muestra', {}).get('filas_exportadas'))} filas "
                f"({self.meta.get('muestra', {}).get('metodo', 'n/d')})")
            st.warning("Análisis **descriptivo**: las diferencias entre grupos no demuestran causalidad, eficiencia "
                       "clínica ni efectos operativos, y no se generalizan a otros hospitales.")
        st.subheader("Preguntas de negocio")
        cols = st.columns(2)
        for i, (qid, question) in enumerate(ins.QUESTIONS.items()):
            with cols[i % 2].container(border=True):
                st.markdown(f"**{qid}** · {question}")
                st.button(f"Ir a {PAGES[QUESTION_PAGE[qid]]}", key=f"go_{qid}", on_click=_go_to, args=(QUESTION_PAGE[qid],))
        with st.expander("Archivos que usa la app y prioridad de fuentes"):
            st.markdown("1. Tablas de EA4 calculadas con el dataset completo (`EA4/results/tables`).\n"
                        "2. Tablas vigentes de EA3 para las ocho tablas comunes, si faltan las de EA4.\n"
                        "3. Muestra sistemática (`EA4/data/processed`) solo para vistas de registros individuales, siempre rotuladas.")
            st.dataframe(dl.file_status(), hide_index=True, width="stretch")

    def _page_pc1(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("PC1", ins.QUESTIONS["PC1"])
        ui.universe_note(self.universe_text(cube, f))
        k = bm.cube_kpis(cube)
        q = self.t["data_quality"].set_index("variable")
        null_interest = q.loc[[v for v in INTEREST_VARS if v in q.index], "pct_nulos"].max()
        c = st.columns(6)
        ui.kpi(c[0], "Admisiones analizadas", num(k["registros"]), "Registros del universo de la vista.")
        ui.kpi(c[1], "Estancia media", days(k["estancia_dias_media"], 2), "Media de stay_in_days (estancias válidas).")
        ui.kpi(c[2], "Estancia mediana", days(k["estancia_dias_mediana"], 0), "Percentil 50 de stay_in_days.")
        ui.kpi(c[3], "Depósito medio", num(k["deposito_admision_media"], 2),
               "Media de admission_deposit; unidad monetaria no documentada. No es ingreso ni costo.")
        ui.kpi(c[4], "Departamentos", num(k["departamentos"]), "Departamentos con registros en la vista.")
        ui.kpi(c[5], "Nulos en variables de interés", pct(null_interest, 2),
               "Máximo % de nulos en estancia, depósito, departamento, severidad y personal (dataset completo, sin filtros).")
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(ch.stay_distribution(bm.cube_stay_distribution(cube), self.universe_short(f)))
        with c2:
            dist = self.t["distribution"].set_index("variable")
            ui.chart(ch.histogram(self.t["histograms"], "admission_deposit", "Depósito de admisión (unidad no documentada)",
                                  "Distribución del depósito de admisión", "Dataset completo · no cambia con los filtros",
                                  marker=dist.loc["admission_deposit", "mediana"], marker_label="mediana"))
        st.markdown("**Resumen de distribución y atípicos (regla de Tukey) — dataset completo, sin filtros**")
        cols = ["variable", "validos", "nulos", "media", "mediana", "desv_est", "cv_pct", "minimo", "q1", "q3", "ric",
                "maximo", "limite_inferior_tukey", "limite_superior_tukey", "atipicos", "atipicos_pct"]
        ui.table(self.t["distribution"][cols], {c: FMT_2 for c in cols[3:] if c != "atipicos"} | {"atipicos": "%d"})
        ui.interpretation(self.official_insights()["PC1"])

    def universe_short(self, f: Filters) -> str:
        return ch.UNIVERSE_FULL + (" filtrado · " + f.describe() if f.active else " · sin filtros")

    def _page_p1(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("P1", ins.QUESTIONS["P1"])
        ui.universe_note(self.universe_text(cube, f))
        d = self.dept_view(cube, f.active)
        sub = self.universe_short(f)
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(ch.dept_bars(d, "estancia_dias_media", "Estancia media por departamento", "Estancia media (días)", sub))
        with c2:
            ui.chart(ch.dept_bars(d, "deposito_admision_media", "Depósito medio por departamento",
                                  "Depósito medio (unidad no documentada)", sub, decimals=0))
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(ch.admissions_vs_patient_days(d, sub))
        with c2:
            ui.chart(ch.box_from_stats(bm.cube_box_stats(cube, "department"), "department",
                                       "Distribución de la estancia por departamento", sub + " · cuartiles exactos"))
        cols = ["department", "registros", "pct_registros", "estancia_dias_media", "estancia_dias_mediana",
                "estancia_dias_desv_est", "dias_paciente", "pct_dias_paciente", "deposito_admision_media", "deposito_admision_mediana"]
        ui.table(d[cols], {"registros": "%d", "dias_paciente": "%d"} | {c: FMT_2 for c in cols[2:] if c not in ("dias_paciente",)})
        st.caption("Días-paciente = suma de las estancias válidas del grupo; participación = días-paciente del departamento / "
                   "total del universo × 100. " + ("La mediana del depósito no se puede recalcular con filtros (requiere registros "
                                                   "individuales) y se muestra vacía." if f.active else
                                                   "La mediana del depósito proviene de results_by_department.csv."))
        ui.interpretation(ins.p1_departments(d))

    def _page_p2(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("P2", ins.QUESTIONS["P2"])
        ui.universe_note(self.universe_text(cube, f))
        sev = self.severity_view(cube, f.active)
        comp = bm.cube_expected_stay(cube, "severity_of_illness", "department")
        sub = self.universe_short(f)
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(ch.severity_observed_expected(comp, sub))
        with c2:
            ui.chart(ch.category_dots(sev, "severity_of_illness", "deposito_admision_media", "Depósito medio por severidad",
                                      "Depósito medio (unidad no documentada)", sub, order=config.SEVERITY_ORDER))
        ui.chart(ch.box_from_stats(bm.cube_box_stats(cube, "severity_of_illness"), "severity_of_illness",
                                   "Distribución de la estancia por severidad", sub + " · cuartiles exactos"))
        merged = sev.merge(comp[["severity_of_illness", "estancia_esperada_composicion", "diferencia_dias"]], on="severity_of_illness")
        cols = ["severity_of_illness", "registros", "pct_registros", "estancia_dias_media", "estancia_dias_mediana",
                "estancia_esperada_composicion", "diferencia_dias", "deposito_admision_media", "deposito_admision_mediana"]
        ui.table(merged[cols], {"registros": "%d"} | {c: FMT_2 for c in cols[2:]})
        st.caption("Estancia esperada por composición = Σ_d (registros del departamento d en la severidad × estancia media de d) "
                   "/ registros de la severidad (definición del Anexo A del informe). % de registros sobre el universo de la vista.")
        ui.interpretation(ins.p2_severity(sev, comp))

    def _page_p3(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("P3", ins.QUESTIONS["P3"])
        ui.universe_note(self.universe_text(cube, f))
        g = bm.cube_group_metrics(cube, "staff_availability_group")
        corr = self.t["correlations"].set_index(["variable_x", "variable_y"])
        r_stay, r_dep = corr.loc[("staff_available", "stay_in_days")], corr.loc[("staff_available", "admission_deposit")]
        c = st.columns(3)
        ui.kpi(c[0], "ρ personal – estancia", num(r_stay["rho_spearman"], 3), "Spearman, pares válidos, dataset completo.")
        ui.kpi(c[1], "ρ personal – depósito", num(r_dep["rho_spearman"], 3), "Spearman, pares válidos, dataset completo.")
        ui.kpi(c[2], "Pares válidos", num(r_stay["n_pares_validos"]), "Registros con ambos valores no nulos; empates con rango promedio.")
        st.caption("Las correlaciones y la tabla por los 11 valores de personal se calculan con el dataset completo y no cambian con los filtros. "
                   "No se reporta valor p: con 500.000 pares incluso correlaciones despreciables resultan «significativas».")
        sub = self.universe_short(f)
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(ch.category_dots(g, "staff_availability_group", "estancia_dias_media", "Estancia media por grupo de personal",
                                      "Estancia media (días)", sub, order=config.STAFF_GROUP_ORDER))
        with c2:
            ui.chart(ch.category_dots(g, "staff_availability_group", "deposito_admision_media", "Depósito medio por grupo de personal",
                                      "Depósito medio (unidad no documentada)", sub, order=config.STAFF_GROUP_ORDER))
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(ch.staff_level_bars(self.t["staff_level"], "Dataset completo · no cambia con los filtros"))
        with c2:
            staff_pairs = self.t["correlations"][self.t["correlations"]["variable_x"] == "staff_available"]
            ui.chart(ch.correlation_bars(staff_pairs, "Dataset completo · no cambia con los filtros"))
        cols = ["staff_availability_group", "registros", "pct_registros", "estancia_dias_media", "estancia_dias_mediana",
                "deposito_admision_media", "deposito_admision_desv_est"]
        ui.table(g[cols], {"registros": "%d"} | {c: FMT_2 for c in cols[2:]})
        ui.interpretation(ins.p3_staff(g, self.t["correlations"]))

    def _page_p4(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("P4", ins.QUESTIONS["P4"])
        ui.universe_note(self.universe_text(cube, f))
        cross = bm.cube_group_metrics(cube, ["department", "severity_of_illness"])
        metric = st.radio("Métrica del mapa de calor", list(METRIC_LABELS), format_func=METRIC_LABELS.get, horizontal=True, key="p4_metric")
        ui.chart(ch.heatmap(cross, metric, f"{METRIC_LABELS[metric]} por departamento y severidad", METRIC_LABELS[metric],
                            self.universe_short(f) + " · celdas sin registros: «sin datos» (no cero)",
                            decimals=0 if metric == "registros" else 2))
        cols = ["department", "severity_of_illness", "registros", "estancia_dias_media", "estancia_dias_mediana",
                "deposito_admision_media", "deposito_admision_desv_est"]
        ui.table(cross[cols], {"registros": "%d"} | {c: FMT_2 for c in cols[3:]})
        if not f.active:
            st.caption("La mediana del depósito por combinación (sin filtros) está en results_by_department_severity.csv; "
                       "se muestra en la página P5 como dept_severity_avg_deposit (media).")
        ui.interpretation(ins.p4_cross(cross))

    def _page_p5(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("P5", ins.QUESTIONS["P5"])
        ui.universe_note(self.universe_text(cube, f))
        sr = bm.cube_stay_range_table(cube)
        tabs = st.tabs(["Rango de estancia", "Puntaje de complejidad", "Depósito y visitas por día",
                        "Frente al promedio del departamento", "Depósito por departamento × severidad"])
        with tabs[0]:
            ui.chart(ch.stay_range_bars(sr, self.universe_short(f)))
            ui.table(sr, {"registros": "%d", "pct_total": FMT_2, "pct_clasificados": FMT_2})
            st.caption("pct_total: denominador = todos los registros del universo; pct_clasificados: solo los que recibieron rango.")
        with tabs[1]:
            ui.universe_note("Dataset completo · no cambia con los filtros (el puntaje no está en el cubo).")
            ui.chart(ch.complexity_bars(self.t["complexity"], "Dataset completo"))
            cols = ["clinical_complexity_score", "registros", "estancia_dias_media", "estancia_dias_mediana", "deposito_admision_media"]
            ui.table(self.t["complexity"][cols], {"registros": "%d"} | {c: FMT_2 for c in cols[2:]})
            st.caption("Puntaje = peso de severidad (Minor 1, Moderate 2, Extreme 3) × peso de admisión (Elective 1,0; Urgent 1,25; "
                       "Trauma 1,5). Emergency no tiene peso: esos registros quedan «sin dato». Exploratorio, no clínico.")
        with tabs[2]:
            ui.universe_note("Dataset completo · histogramas con todos los registros; no cambian con los filtros.")
            c1, c2 = st.columns(2)
            dist = self.t["distribution"].set_index("variable")
            with c1:
                ui.chart(ch.histogram(self.t["histograms"], "deposit_per_stay_day", "Depósito por día de estancia",
                                      "Distribución del depósito por día de estancia", "Dataset completo · no es el costo diario",
                                      dist.loc["deposit_per_stay_day", "mediana"], "mediana"))
            with c2:
                ui.chart(ch.histogram(self.t["histograms"], "visitors_per_day", "Visitantes por día de estancia",
                                      "Distribución de visitantes por día", "Dataset completo · no mide carga asistencial",
                                      dist.loc["visitors_per_day", "mediana"], "mediana"))
            cols = ["department", "registros", "deposito_por_dia_media", "deposito_por_dia_mediana", "visitas_por_dia_media",
                    "visitas_por_dia_mediana"]
            ui.table(self.t["indicators"][cols], {"registros": "%d"} | {c: FMT_2 for c in cols[2:]})
            sample = self.filtered_sample(f)
            if sample is not None and len(sample):
                ui.universe_note(f"Vista de registros individuales: {ch.UNIVERSE_SAMPLE.lower()} con los filtros aplicados "
                                 f"({num(len(sample))} registros).", sample=True)
                ui.chart(ch.sample_box(sample, "deposit_per_stay_day", "Depósito por día de estancia",
                                       "Depósito por día de estancia por departamento"))
            else:
                st.info("La muestra de registros no está disponible o no tiene registros con estos filtros.")
        with tabs[3]:
            ui.universe_note("Dataset completo · no cambia con los filtros.")
            dist = self.t["distribution"].set_index("variable")
            ui.chart(ch.histogram(self.t["histograms"], "stay_vs_dept_avg", "Estancia − media de su departamento (días)",
                                  "Desviación de la estancia frente al promedio departamental", "Dataset completo",
                                  0.0, "0 = igual al promedio"))
            bench = self.t["patient_days"][["department", "admisiones", "estancia_media"]].rename(
                columns={"estancia_media": "dept_avg_stay (días)"})
            ui.table(bench, {"admisiones": "%d", "dept_avg_stay (días)": FMT_2})
            st.caption(f"stay_vs_dept_avg = stay_in_days − dept_avg_stay. Mediana {num(dist.loc['stay_vs_dept_avg', 'mediana'], 2)} días; "
                       "valores positivos: estancia superior al promedio de su departamento (no indica ineficiencia).")
        with tabs[4]:
            ui.universe_note("Dataset completo · no cambia con los filtros (valores de dept_severity_avg_deposit).")
            ui.chart(ch.heatmap(self.t["department_severity"], "deposito_admision_media",
                                "Depósito medio por departamento y severidad (dept_severity_avg_deposit)", "Depósito medio",
                                "Dataset completo · unidad monetaria no documentada", decimals=0))
        ui.interpretation(ins.p5_indicators(sr, self.t["complexity"], self.t["indicators"]))

    def _page_pc2(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("PC2", ins.QUESTIONS["PC2"])
        ui.universe_note("Dataset completo · esta página no cambia con los filtros: η², correlaciones y agrupaciones usan todos los registros.")
        c1, c2 = st.columns(2)
        with c1:
            ui.chart(ch.eta_bars(self.t["eta_squared"], "stay_in_days", "Dataset completo"))
        with c2:
            ui.chart(ch.eta_bars(self.t["eta_squared"], "admission_deposit", "Dataset completo"))
        st.caption("η² = Σ_g n_g (media_g − media)² / Σ_i (y_i − media)², con los registros de valor válido; los nulos del factor "
                   "forman su propio grupo. Los η² de factores distintos no se suman: los factores están relacionados entre sí.")
        ui.chart(ch.correlation_bars(self.t["correlations"], "Dataset completo · Spearman con pares válidos"))
        st.markdown("**Estancia y depósito por categoría**")
        c1, c2 = st.columns(2)
        variables = list(dict.fromkeys(self.t["pc2_groups"]["variable"]))
        var = c1.selectbox("Variable", variables, key="pc2_var")
        metric = c2.selectbox("Métrica", ["estancia_dias_media", "estancia_dias_mediana", "deposito_admision_media"],
                              format_func=METRIC_LABELS.get, key="pc2_metric")
        ui.chart(ch.pc2_bars(self.t["pc2_groups"], var, metric, METRIC_LABELS[metric], "Dataset completo"))
        sub = self.t["pc2_groups"][self.t["pc2_groups"]["variable"] == var]
        cols = ["categoria", "registros", "estancia_dias_media", "estancia_dias_mediana", "deposito_admision_media", "deposito_admision_mediana"]
        ui.table(sub[cols], {"registros": "%d"} | {c: FMT_2 for c in cols[2:]})
        ui.interpretation(ins.pc2_associations(self.t["eta_squared"], self.t["correlations"]))

    def _page_pc3(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("PC3", ins.QUESTIONS["PC3"])
        ui.universe_note("Dataset completo · no cambia con los filtros.")
        q = self.t["data_quality"]
        c1, c2 = st.columns([1, 1.2])
        with c1:
            ui.chart(ch.quality_nulls(q, "Dataset completo enriquecido"))
        with c2:
            st.markdown("**Advertencias de calidad**")
            cal = self.meta.get("calidad", {})
            warnings = pd.DataFrame([
                ("Duplicados exactos", num(cal.get("duplicados_exactos")), "Filas idénticas en todas las columnas (se eliminarían; no hubo)."),
                ("Errores de conversión numérica", num(cal.get("errores_conversion")), "Valores no convertibles a número (se reportan, no se eliminan)."),
                ("patientid negativos", num(cal.get("patientid_negativos")), "Semántica no documentada; se conservan."),
                ("patientid repetidos", num(cal.get("patientid_repetidos")), "El identificador no es único: no se usa como clave."),
                ("Categorías sin peso en el puntaje", str(cal.get("categorias_sin_peso", "n/d")), "Emergency no está en el mapeo: puntaje nulo."),
                ("Registros sin rango de estancia", num(cal.get("sin_rango_estancia")), "Estancia nula o menor que 1."),
            ], columns=["control", "resultado", "tratamiento"])
            st.dataframe(warnings, hide_index=True, width="stretch")
            st.caption("Distinción: los nulos de health_conditions son faltantes sin significado documentado; los del puntaje son "
                       "indicadores no calculables por regla; ninguno se convierte en error ni se imputa.")
        ui.table(q, {"registros": "%d", "nulos": "%d", "validos": "%d", "pct_nulos": FMT_2})
        st.info("Limitaciones de la fuente: la ficha de Hugging Face no documenta licencia, procedencia, carácter real o "
                "sintético, periodo ni definiciones de las columnas (p. ej., la moneda del depósito).")
        ui.interpretation(ins.pc3_quality(q))

    def _page_fin(self, f: Filters, cube: pd.DataFrame) -> None:
        st, ui = self._st, self._ui
        ui.question_box("Conclusiones", ins.MAIN_QUESTION)
        ui.universe_note("Conclusiones con el dataset completo (sin filtros): se redactan desde las tablas oficiales.")
        insights = self.official_insights()
        st.subheader("Respuestas a P1-P5 y resultados complementarios")
        for qid in ["P1", "P2", "P3", "P4", "P5", "PC1", "PC2", "PC3"]:
            item = insights[qid]
            with st.container(border=True):
                st.markdown(f"**{qid} · {ui.esc(ins.QUESTIONS[qid])}**")
                st.markdown(f"**Observación.** {ui.esc(item['patron'])}")
                st.markdown(f"**Respuesta.** {ui.esc(item['respuesta'])}")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Recomendaciones (para validar)")
            for qid in ["P1", "P2", "P3", "P4", "P5", "PC3"]:
                st.markdown(f"- **{qid}:** {ui.esc(insights[qid]['decision'])}")
            st.subheader("Hipótesis (no verificadas con estos datos)")
            st.markdown("- Las diferencias de estancia asociadas a la edad, la sala y el género podrían deberse a cómo se "
                        "distribuyen los pacientes entre departamentos (los factores están entrelazados).\n"
                        "- La falta de relación con el personal disponible podría reflejar cómo se registró la variable.")
        with c2:
            st.subheader("Limitaciones")
            st.markdown("- Análisis descriptivo: sin causalidad ni juicios de eficiencia.\n"
                        "- Ficha del dataset sin licencia, procedencia, periodo ni definiciones de variables.\n"
                        "- El depósito no es ingreso, beneficio ni costo total; su moneda no está documentada.\n"
                        "- Umbrales de personal y de estancia, y pesos del puntaje, son reglas del proyecto.\n"
                        "- No hay fechas ni número de camas: no se calculan ocupación ni giro de cama.\n"
                        "- Los resultados no se generalizan a otros hospitales.")


def _go_to(page: str) -> None:
    """Callback de los accesos directos del inicio (cambia la sección antes del siguiente ciclo)."""
    import streamlit as st

    st.session_state["pagina"] = page
