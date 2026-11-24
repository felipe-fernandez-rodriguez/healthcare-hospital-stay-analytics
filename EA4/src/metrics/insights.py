"""Interpretaciones redactadas con las cifras de las tablas (ninguna cifra escrita a mano).

Cada función devuelve un diccionario con las cinco partes que acompañan a cada visual:
``metrica`` (qué se muestra y cómo se calcula), ``patron`` (qué se observa), ``respuesta``
(cómo responde la pregunta), ``decision`` (qué revisión operativa podría apoyar) y
``limitacion``. Las afirmaciones cualitativas se derivan de comparaciones entre las cifras
calculadas, de modo que cambian si cambian los datos o los filtros.
"""
from __future__ import annotations

import pandas as pd

from ..config import NO_DATA
from ..formatting import days, num, pct

QUESTIONS = {
    "P1": "¿Cómo varían la estancia y el depósito entre departamentos?",
    "P2": "¿Qué diferencias se observan según la severidad de la enfermedad?",
    "P3": "¿Cómo se distribuyen la estancia y el depósito según la disponibilidad de personal?",
    "P4": "¿Qué patrones aparecen al cruzar departamento y severidad?",
    "P5": "¿Qué utilidad aportan los indicadores derivados de EA3?",
    "PC1": "¿Cómo se distribuyen la estancia y el depósito en el conjunto de admisiones?",
    "PC2": "¿Qué otras variables se relacionan descriptivamente con la estancia y el depósito?",
    "PC3": "¿Qué limitaciones de calidad condicionan la interpretación?",
}
MAIN_QUESTION = ("¿Cómo varían la duración de la estancia hospitalaria y el valor del depósito de admisión según el "
                 "departamento médico, el nivel de severidad de la enfermedad y los recursos de personal disponibles "
                 "en los registros del dataset seleccionado?")
DEPOSIT_NOTE = "el depósito de admisión registrado (unidad monetaria no documentada) no equivale a ingresos ni a costos"
NEGLIGIBLE_RHO = 0.1  # convención usual para una asociación monótona despreciable (|ρ| < 0,1)


def _spread_pct(values: pd.Series) -> float:
    """Diferencia entre el máximo y el mínimo, como % de la media de los valores."""
    values = values.dropna()
    return (values.max() - values.min()) / values.mean() * 100 if len(values) and values.mean() else float("nan")


def _composition_clause(residual: float, total: float) -> str:
    """Qué parte de la diferencia entre severidades queda tras descontar la mezcla de departamentos."""
    if not total:
        return "."
    share = residual / total
    if share < 0.25:
        return f"; la diferencia residual ({days(residual, 2)} de {days(total, 2)}) indica que la mezcla de departamentos explica la mayor parte."
    if share < 0.75:
        return f"; la mezcla de departamentos explica solo una parte de la diferencia (residual {days(residual, 2)} de {days(total, 2)})."
    return f"; la mezcla de departamentos no explica la diferencia (residual {days(residual, 2)} de {days(total, 2)})."


def _not_enough(group: str) -> dict:
    msg = f"Con los filtros actuales hay menos de dos categorías de {group}: no es posible comparar."
    return {"metrica": "—", "patron": msg, "respuesta": msg, "decision": "Amplíe los filtros para comparar grupos.",
            "limitacion": "—"}


def p1_departments(dept: pd.DataFrame) -> dict:
    """``dept``: métricas por departamento (cubo o tabla oficial) con días-paciente y participaciones."""
    if len(dept) < 2:
        return _not_enough("departamento")
    d = dept.sort_values("estancia_dias_mediana")
    lo, hi = d.iloc[0], d.iloc[-1]
    big = dept.sort_values("registros").iloc[-1]
    ratio = hi["estancia_dias_mediana"] / lo["estancia_dias_mediana"]
    dep_spread = _spread_pct(dept["deposito_admision_media"])
    stay_spread = _spread_pct(dept["estancia_dias_media"])
    heavier = big["pct_dias_paciente"] < big["pct_registros"]
    return {
        "metrica": "Admisiones, estancia media y mediana (días), días-paciente (suma de estancias válidas) y su "
                   "participación (días-paciente del departamento / total × 100), y depósito medio por departamento.",
        "patron": (f"La estancia mediana va de {days(lo['estancia_dias_mediana'], 0)} en {lo['department']} "
                   f"(n = {num(lo['registros'])}) a {days(hi['estancia_dias_mediana'], 0)} en {hi['department']} "
                   f"(n = {num(hi['registros'])}). {big['department']} concentra {pct(big['pct_registros'])} de las "
                   f"admisiones y {pct(big['pct_dias_paciente'])} de los días-paciente. El depósito medio varía "
                   f"{pct(dep_spread)} entre departamentos, frente a {pct(stay_spread)} de la estancia media."),
        "respuesta": (f"La estancia mediana del departamento con mayor estancia es {num(ratio, 1)} veces la del de menor "
                      f"estancia; el depósito medio "
                      + ("cambia mucho menos que la estancia." if dep_spread < stay_spread else "cambia tanto o más que la estancia.")),
        "decision": ("Revisar la carga de camas por servicio con los días-paciente además de las admisiones: "
                     + (f"{big['department']} pesa menos en días-paciente que en admisiones, y los servicios de estancia "
                        "larga pesan más de lo que indica su volumen." if heavier else
                        "en este universo las participaciones en admisiones y en días-paciente son similares.")),
        "limitacion": f"Descriptivo: no mide eficiencia ni causas de la estancia; {DEPOSIT_NOTE}.",
    }


def p2_severity(sev: pd.DataFrame, composition: pd.DataFrame) -> dict:
    """``sev``: métricas por severidad; ``composition``: observada frente a esperada por departamento."""
    if len(sev) < 2:
        return _not_enough("severidad")
    comp = composition.set_index("severity_of_illness")
    medians = sev["estancia_dias_mediana"].unique()
    stay_range = sev["estancia_dias_media"].max() - sev["estancia_dias_media"].min()
    max_diff = comp["diferencia_dias"].abs().max()
    parts = "; ".join(f"{r['severity_of_illness']} {days(r['estancia_dias_media'], 2)} (n = {num(r['registros'])})"
                      for _, r in sev.iterrows())
    return {
        "metrica": ("Estancia media y mediana y depósito medio por severidad; estancia esperada por composición = "
                    "Σ (registros del departamento d en la severidad × estancia media de d) / registros de la severidad."),
        "patron": (f"Estancia media: {parts}. "
                   + (f"La mediana es {days(medians[0], 0)} en todas las categorías. " if len(medians) == 1 else "")
                   + f"Depósito medio de {num(sev['deposito_admision_media'].min(), 2)} a "
                   f"{num(sev['deposito_admision_media'].max(), 2)}."),
        "respuesta": (f"Entre severidades la estancia media difiere como máximo {days(stay_range, 2)}; al descontar la "
                      f"mezcla de departamentos, la diferencia máxima entre estancia observada y esperada es de "
                      f"{days(max_diff, 2)}."),
        "decision": ("No usar la severidad registrada como indicador de la estancia esperada sin validar su definición"
                     + _composition_clause(max_diff, stay_range)),
        "limitacion": "El orden Minor < Moderate < Extreme se asume por el nombre; la estandarización no demuestra causalidad.",
    }


def p3_staff(groups: pd.DataFrame, correlations: pd.DataFrame | None) -> dict:
    """``groups``: métricas por grupo de personal; ``correlations``: tabla de Spearman (dataset completo)."""
    if len(groups) < 2:
        return _not_enough("disponibilidad de personal")
    stay_range = groups["estancia_dias_media"].max() - groups["estancia_dias_media"].min()
    text_rho, negligible = "", None
    if correlations is not None and len(correlations):
        c = correlations.set_index(["variable_x", "variable_y"])
        r1 = c.loc[("staff_available", "stay_in_days")]
        r2 = c.loc[("staff_available", "admission_deposit")]
        negligible = max(abs(r1["rho_spearman"]), abs(r2["rho_spearman"])) < NEGLIGIBLE_RHO
        text_rho = (f" En el dataset completo, ρ de Spearman con la estancia = {num(r1['rho_spearman'], 3)} y con el "
                    f"depósito = {num(r2['rho_spearman'], 3)} (n = {num(r1['n_pares_validos'])} pares válidos).")
    verdict = ("asociación monótona despreciable (|ρ| < 0,1)" if negligible
               else "asociación monótona no despreciable" if negligible is False else "sin correlación disponible")
    return {
        "metrica": ("Registros, estancia media y mediana y depósito medio por grupo de personal (Bajo 0-3, Medio 4-6, "
                    "Alto 7+, umbrales de EA2), y ρ de Spearman entre staff_available y cada variable."),
        "patron": (f"La estancia media va de {days(groups['estancia_dias_media'].min(), 2)} a "
                   f"{days(groups['estancia_dias_media'].max(), 2)} entre grupos y el depósito medio varía "
                   f"{pct(_spread_pct(groups['deposito_admision_media']), 2)}.{text_rho}"),
        "respuesta": f"Diferencia máxima de estancia media entre grupos: {days(stay_range, 2)}; {verdict}.",
        "decision": ("Verificar qué mide staff_available (momento, unidad, servicio) antes de usarlo en la planeación de "
                     "personal o camas."),
        "limitacion": "Una correlación cercana a cero no prueba ausencia de efecto; la variable no está documentada.",
    }


def p4_cross(cross: pd.DataFrame) -> dict:
    """``cross``: métricas por departamento × severidad."""
    if cross["department"].nunique() < 2 or cross["severity_of_illness"].nunique() < 2:
        return _not_enough("departamento o severidad")
    within = cross.groupby("department")["estancia_dias_media"].agg(lambda s: s.max() - s.min())
    between = cross.groupby("severity_of_illness")["estancia_dias_media"].agg(lambda s: s.max() - s.min())
    small = cross.sort_values("registros").iloc[0]
    dept_dominates = within.max() < between.max()
    return {
        "metrica": "Estancia media (días) y registros de cada combinación departamento × severidad; las celdas sin datos se muestran como no disponibles.",
        "patron": (f"Dentro de un departamento, la estancia media cambia como máximo {days(within.max(), 2)} entre "
                   f"severidades; dentro de una misma severidad, la diferencia entre departamentos llega a "
                   f"{days(between.max(), 2)}. La celda más pequeña es {small['department']} × "
                   f"{small['severity_of_illness']} con {num(small['registros'])} registros."),
        "respuesta": ("Las diferencias entre departamentos superan a las de severidad dentro de cada departamento."
                      if dept_dominates else
                      "Las diferencias por severidad dentro de un departamento son comparables o mayores que las diferencias entre departamentos."),
        "decision": "Comparar estancias entre servicios dentro de la misma severidad para no confundir el efecto de la mezcla de pacientes.",
        "limitacion": "Las celdas pequeñas tienen estimaciones menos estables; el cruce no controla otros factores.",
    }


def p5_indicators(stay_range: pd.DataFrame, complexity: pd.DataFrame, indicators: pd.DataFrame) -> dict:
    sr = stay_range.set_index("stay_range")
    labels = [i for i in sr.index if i not in ("Sin clasificar", "Total")]
    top_label = max(labels, key=lambda lab: sr.loc[lab, "pct_total"])
    scored = complexity[complexity["clinical_complexity_score"] != NO_DATA].copy()
    scored["score"] = scored["clinical_complexity_score"].astype(float)
    scored = scored.sort_values("score")
    monotonic = scored["estancia_dias_media"].is_monotonic_increasing
    unscored = complexity.loc[complexity["clinical_complexity_score"] == NO_DATA, "registros"].sum()
    total = complexity["registros"].sum()
    ind = indicators.sort_values("deposito_por_dia_mediana")
    return {
        "metrica": ("stay_range (Corta 1-3, Moderada 4-7, Prolongada >7 días), clinical_complexity_score "
                    "(severidad × tipo de admisión, exploratorio), deposit_per_stay_day = depósito / estancia, "
                    "visitors_per_day = visitantes / estancia y stay_vs_dept_avg = estancia − media del departamento."),
        "patron": ("Rangos: " + "; ".join(f"{lab} {pct(sr.loc[lab, 'pct_total'], 2)} (n = {num(sr.loc[lab, 'registros'])})"
                                          for lab in labels)
                   + f". El puntaje se calculó para {num(total - unscored)} de {num(total)} registros "
                   f"({pct((total - unscored) / total * 100, 2)}); su estancia media va de "
                   f"{days(scored['estancia_dias_media'].min(), 2)} a {days(scored['estancia_dias_media'].max(), 2)}. "
                   f"La mediana del depósito por día va de {num(ind.iloc[0]['deposito_por_dia_mediana'], 2)} "
                   f"({ind.iloc[0]['department']}) a {num(ind.iloc[-1]['deposito_por_dia_mediana'], 2)} "
                   f"({ind.iloc[-1]['department']})."),
        "respuesta": (f"«{top_label}» concentra {pct(sr.loc[top_label, 'pct_total'])} de los registros"
                      + (", por lo que con estos umbrales la segmentación discrimina poco" if sr.loc[top_label, 'pct_total'] > 80 else "")
                      + "; la estancia media "
                      + ("aumenta con el puntaje de complejidad." if monotonic else "no sigue un orden monótono con el puntaje de complejidad.")
                      + " El depósito por día refleja sobre todo la duración de la estancia."),
        "decision": "Revisar los umbrales de stay_range con criterios de la institución y no usar el puntaje como predictor.",
        "limitacion": ("Umbrales y pesos son reglas del proyecto; Emergency no tiene peso en el puntaje; el depósito por "
                       "día no es el costo diario y las visitas por día no miden carga asistencial."),
    }


def pc1_distribution(distribution: pd.DataFrame) -> dict:
    d = distribution.set_index("variable")
    s, p = d.loc["stay_in_days"], d.loc["admission_deposit"]
    skew = "asimétrica a la derecha (media mayor que la mediana)" if s["media"] > s["mediana"] else "no asimétrica a la derecha"
    return {
        "metrica": "Media, mediana, desviación, cuartiles y atípicos por la regla de Tukey (fuera de Q1 − 1,5 RIC y Q3 + 1,5 RIC), con todos los registros válidos.",
        "patron": (f"Estancia: media {days(s['media'], 2)}, mediana {days(s['mediana'], 0)}, desviación "
                   f"{days(s['desv_est'], 2)}; {pct(s['atipicos_pct'])} de valores fuera de las vallas de Tukey. "
                   f"Depósito: media {num(p['media'], 2)}, mediana {num(p['mediana'], 2)}; {pct(p['atipicos_pct'])} de atípicos."),
        "respuesta": f"La distribución de la estancia es {skew}; el rango intercuartílico del depósito es {num(p['ric'], 2)}.",
        "decision": "Reportar la estancia con la mediana y su distribución, no solo con la media.",
        "limitacion": "Los atípicos se conservan: no hay evidencia de que sean errores.",
    }


def pc2_associations(eta: pd.DataFrame, correlations: pd.DataFrame) -> dict:
    e = eta[eta["variable"] == "stay_in_days"].sort_values("eta2", ascending=False)
    ed = eta[eta["variable"] == "admission_deposit"].sort_values("eta2", ascending=False)
    top = "; ".join(f"{r['factor']} {pct(r['eta2'] * 100)}" for _, r in e.head(4).iterrows())
    strongest = correlations.reindex(correlations["rho_spearman"].abs().sort_values(ascending=False).index).iloc[0]
    return {
        "metrica": ("Razón de varianza η² = Σ_g n_g (media_g − media)² / Σ_i (y_i − media)² para cada factor, y ρ de "
                    "Spearman con pares válidos entre variables numéricas."),
        "patron": (f"Factores con mayor η² para la estancia: {top}. Para el depósito, el máximo es "
                   f"{ed.iloc[0]['factor']} con {pct(ed.iloc[0]['eta2'] * 100, 2)}. La correlación de mayor magnitud es "
                   f"{strongest['variable_x']} – {strongest['variable_y']} (ρ = {num(strongest['rho_spearman'], 3)})."),
        "respuesta": (f"{e.iloc[0]['factor']} es el factor con mayor η² para la estancia ({pct(e.iloc[0]['eta2'] * 100)}); "
                      f"para el depósito ningún factor supera {pct(ed.iloc[0]['eta2'] * 100, 2)}. Los η² de factores "
                      "distintos pueden solaparse porque los factores están relacionados entre sí."),
        "decision": "Analizar en conjunto los factores con mayor η² (cruces) antes de atribuir la estancia a uno solo.",
        "limitacion": "η² y ρ son medidas de asociación, no de causalidad.",
    }


def pc3_quality(quality: pd.DataFrame) -> dict:
    with_nulls = quality[quality["nulos"] > 0].sort_values("pct_nulos", ascending=False)
    lines = "; ".join(f"{r['variable']} {pct(r['pct_nulos'], 2)}" for _, r in with_nulls.iterrows()) or "ninguna"
    complete = int((quality["nulos"] == 0).sum())
    return {
        "metrica": "Nulos y porcentaje de nulos por variable (sobre todos los registros) y observaciones de conversión e indicadores no calculables.",
        "patron": f"Variables con nulos: {lines}. Variables completas: {complete} de {len(quality)}.",
        "respuesta": (f"{len(with_nulls)} variables tienen nulos; los nulos se conservan y no se imputan, y los indicadores "
                      "que no pueden calcularse quedan vacíos en lugar de cero."),
        "decision": "Solicitar a la fuente la definición de las variables y del valor nulo antes de usar estas columnas en decisiones.",
        "limitacion": "La ficha del dataset no documenta licencia, procedencia, periodo ni definiciones de columnas.",
    }
