"""Pruebas de EA3: enriquecimiento, tablas de resultados, calidad y reporte.

Usan DataFrames pequeños con respuestas calculables a mano; no requieren Internet.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.analysis import descriptive as ds
from src.analysis.quality import compare_dataframes
from src.analysis.results import build_result_tables, export_tables, key_findings
from src.data import enrichment as en
from src.data import transformations as tr
from src.data.dataset_io import export_csv_head
from src.utils.reporter import DatasetSummarizer


def small_df() -> pd.DataFrame:
    """6 filas ya normalizadas (como tras ``transformations.apply_all``)."""
    return pd.DataFrame(
        {
            "department": ["a", "a", "b", "b", "b", "c"],
            "severity_of_illness": ["Minor", "Extreme", "Moderate", "Moderate", "Minor", "Minor"],
            "type_of_admission": ["Trauma", "Urgent", "Emergency", "Trauma", "Trauma", "Urgent"],
            "stay_in_days": [2, 10, 4, 8, 0, np.nan],
            "admission_deposit": [100.0, 500.0, 400.0, 800.0, 300.0, 250.0],
            "visitors_with_patient": [4, 5, 8, 2, 1, 3],
            "staff_available": [0, 3, 4, 6, 7, 10],
        }
    )


# --- indicadores de cociente ---------------------------------------------
def test_ratios_exact_values_and_invalid_denominators_stay_null():
    out, rec = en.add_deposit_per_stay_day(small_df())
    assert out["deposit_per_stay_day"].iloc[:4].tolist() == [50.0, 50.0, 100.0, 100.0]
    assert out["deposit_per_stay_day"].iloc[4:].isna().all()  # estancia 0 y nula: sin resultado, no 0 ni inf
    assert rec["registros_calculados"] == 4 and rec["registros_no_calculables"] == 2 and rec["denominadores_invalidos"] == 2
    out2, _ = en.add_visitors_per_day(small_df())
    assert out2["visitors_per_day"].iloc[0] == 2.0 and out2["visitors_per_day"].iloc[1] == 0.5
    assert not np.isinf(out2["visitors_per_day"]).any()


def test_negative_stay_is_invalid():
    df = small_df()
    df.loc[0, "stay_in_days"] = -3
    out, rec = en.add_deposit_per_stay_day(df)
    assert math.isnan(out.loc[0, "deposit_per_stay_day"]) and rec["registros_no_calculables"] == 3


# --- rango de estancia -----------------------------------------------------
def test_stay_range_boundaries_and_unclassified():
    df = pd.DataFrame({"stay_in_days": [1, 3, 4, 7, 8, 0, -1, np.nan]})
    out, rec = en.add_stay_range(df)
    got = out["stay_range"].astype(object).tolist()
    assert got[:5] == [en.STAY_LABELS[0], en.STAY_LABELS[0], en.STAY_LABELS[1], en.STAY_LABELS[1], en.STAY_LABELS[2]]
    assert all(pd.isna(v) for v in got[5:]) and rec["sin_clasificar"] == 3


# --- puntaje de complejidad -------------------------------------------------
def test_complexity_score_values_and_unknown_categories():
    out, rec = en.add_clinical_complexity_score(small_df())
    assert out["clinical_complexity_score"].iloc[:2].tolist() == [1.5, 3.75]  # Minor*Trauma, Extreme*Urgent
    assert math.isnan(out["clinical_complexity_score"].iloc[2])  # Emergency no tiene peso: no se inventa
    assert out["clinical_complexity_score"].iloc[3] == 3.0  # Moderate*Trauma
    assert rec["categorias_admision_sin_peso"] == {"Emergency": 1}
    assert rec["registros_calculados"] == 5 and rec["registros_no_calculables"] == 1
    assert "elective" in rec["categorias_del_mapeo_ausentes_en_datos"]


def test_complexity_normalizes_labels_and_flags_unknown_severity():
    df = pd.DataFrame({"severity_of_illness": [" minor ", "Critical"], "type_of_admission": ["TRAUMA", "Trauma"]})
    out, rec = en.add_clinical_complexity_score(df)
    assert out["clinical_complexity_score"].iloc[0] == 1.5 and math.isnan(out["clinical_complexity_score"].iloc[1])
    assert rec["categorias_severidad_sin_peso"] == {"Critical": 1}


# --- benchmarks -------------------------------------------------------------
def test_department_benchmark_matches_manual_means():
    out, rec = en.add_department_benchmark(small_df())
    # a: (2+10)/2 = 6 ; b: estancias válidas 4 y 8 -> 6 (el 0 se excluye) ; c: sin estancia válida -> nulo
    assert out["dept_avg_stay"].tolist()[:5] == [6.0] * 5 and math.isnan(out["dept_avg_stay"].iloc[5])
    assert out["stay_vs_dept_avg"].iloc[:4].tolist() == [-4.0, 4.0, -2.0, 2.0]
    assert math.isnan(out["stay_vs_dept_avg"].iloc[4])  # estancia 0 inválida
    assert rec["departamentos_sin_valores_validos"] == 1


def test_dept_severity_avg_deposit_values_and_row_count():
    df = small_df()
    out, rec = en.add_dept_severity_avg_deposit(df)
    assert len(out) == len(df)
    assert out["dept_severity_avg_deposit"].tolist() == [100.0, 500.0, 600.0, 600.0, 300.0, 250.0]
    assert rec["combinaciones"] == 5


def test_apply_enrichment_preserves_rows_and_only_adds_columns():
    df = small_df()
    out, log = en.apply_enrichment(df)
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns) + en.NEW_COLUMNS
    assert out[list(df.columns)].equals(df)  # no altera las columnas existentes
    assert [r["filas_despues"] for r in log] == [len(df)] * len(en.ENRICHMENT_STEPS)


def test_enrichment_after_transformations_pipeline():
    base = pd.DataFrame({
        "Department": ["gynecology", "surgery"], "Severity of Illness": ["Minor", "Extreme"], "Type of Admission": ["Trauma", "Urgent"],
        "Stay (in days)": [10, 5], "Admission_Deposit": [1000.0, 500.0], "Visitors with Patient": [2, 5],
        "staff_available": [1, 8], "patientid": [1, 1], "Age": ["21-30", "31-40"], "gender": ["Female", "Male"],
        "Ward_Facility_Code": ["A", "B"], "doctor_name": ["x", "y"], "health_conditions": [None, "Diabetes"], "Insurance": ["Yes", "No"],
        "Available Extra Rooms in Hospital": [3, 4],
    })
    transformed, _ = tr.apply_all(base)
    out, _ = en.apply_enrichment(transformed)
    assert out["deposit_per_stay_day"].tolist() == [100.0, 100.0]
    assert out["health_conditions"].isna().sum() == 1  # nulo conservado
    assert len(out) == 2  # patientid repetido no elimina filas


# --- tablas de resultados ---------------------------------------------------
def enriched() -> pd.DataFrame:
    df = small_df()
    df["staff_availability_group"] = pd.cut(df["staff_available"], bins=tr.STAFF_BINS, labels=tr.STAFF_LABELS, ordered=True)
    return en.apply_enrichment(df)[0]


def test_group_metrics_counts_and_stats():
    t = ds.group_metrics(enriched(), "department").set_index("department")
    assert t.loc["a", "registros"] == 2 and t.loc["a", "estancia_dias_media"] == 6.0 and t.loc["a", "deposito_admision_media"] == 300.0
    assert t.loc["b", "registros"] == 3 and t.loc["b", "estancia_dias_validos"] == 3  # 0 cuenta como valor no nulo
    assert t.loc["c", "estancia_dias_validos"] == 0 and t.loc["c", "deposito_admision_validos"] == 1
    assert t["registros"].sum() == 6
    assert t.loc["b", "estancia_dias_mediana"] == 4.0


def test_group_metrics_keeps_null_groups_and_two_keys():
    df = enriched()
    out = ds.group_metrics(df, ["department", "severity_of_illness"])
    assert out["registros"].sum() == len(df)
    nulls = ds.group_metrics(df.assign(department=pd.Series([None] * 6, dtype="object")), "department")
    assert nulls.loc[0, "department"] == ds.NO_DATA and nulls.loc[0, "registros"] == 6


def test_stay_range_table_percentages_and_denominators():
    t = ds.stay_range_table(enriched()).set_index("stay_range")
    assert t.loc["Total", "registros"] == 6
    assert t.loc["Sin clasificar", "registros"] == 2  # estancia 0 y nula
    assert t.loc[en.STAY_LABELS[0], "pct_total"] == round(1 / 6 * 100, 2)
    assert t.loc[en.STAY_LABELS[0], "pct_clasificados"] == 25.0  # 1 de 4 clasificados
    assert t.drop("Total")["registros"].sum() == 6


def test_result_tables_consistency_and_csv_roundtrip(tmp_path):
    df = enriched()
    tables = build_result_tables(df, [])
    for key in ("department", "severity", "staff_availability", "department_severity", "complexity"):
        assert tables[key]["registros"].sum() == len(df)
    files = {k: tmp_path / "res" / f"{k}.csv" for k in tables}
    paths = export_tables(tables, files)  # crea la carpeta
    assert set(paths) == set(tables)
    for key, table in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(files[key]), table, check_dtype=False, check_exact=False)
    assert "estancia" in " ".join(key_findings(tables)).lower()


def test_config_result_files_are_inside_results_dir():
    assert all(p.parent == config.RESULTS_DIR for p in config.RESULT_FILES.values())
    assert config.ENRICHED_CSV.name == "healthcare_enriched_sample.csv"


# --- calidad / comparación --------------------------------------------------
def test_compare_dataframes_reports_renames_new_columns_and_nulls():
    raw = pd.DataFrame({"Stay (in days)": [1, 2], "health_conditions": [None, "x"]})
    final, _ = tr.normalize_column_names(raw)
    final = final.assign(nueva=1)
    cmp = compare_dataframes(raw, final)
    assert cmp["columnas_renombradas"] == {"Stay (in days)": "stay_in_days"}
    assert cmp["columnas_nuevas"] == ["nueva"] and cmp["columnas_eliminadas"] == []
    assert cmp["dimensiones"].loc["columnas"].tolist() == [2, 3]
    row = cmp["columnas"].set_index("columna_final").loc["health_conditions"]
    assert row["nulos_antes"] == 1 and row["pct_nulos_antes"] == 50.0


def test_exact_duplicates_detected_after_enrichment():
    df = enriched()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert int(dup.duplicated().sum()) == 1
    assert len(tr.drop_exact_duplicates(dup)[0]) == len(df)


# --- exportación y reporte --------------------------------------------------
def test_enriched_export_is_first_rows_and_capped(tmp_path):
    df = pd.DataFrame({"x": range(250_000)})
    info = export_csv_head(df, tmp_path / "data" / "processed" / "e.csv")
    saved = pd.read_csv(tmp_path / "data" / "processed" / "e.csv")
    assert len(saved) == config.CSV_MAX_ROWS == info["filas_exportadas"] and info["es_parcial"]
    assert saved["x"].tolist() == list(range(config.CSV_MAX_ROWS))  # primeras filas, orden estable
    assert info["filas_dataset_completo"] == 250_000


def test_summary_report_extra_sections_utf8(tmp_path):
    df = enriched()
    out = DatasetSummarizer(title="RESUMEN EA3").save(
        df, tmp_path / "rep" / "ae3.txt",
        dataset_info={"dataset_id": "a/b", "config": "default", "split": "train", "muestreo": "dataset completo"},
        extra_sections=[("Fórmulas", ["- deposit_per_stay_day: depósito / estancia"]), ("Tabla", pd.DataFrame({"a": [1]}))],
    )
    text = out.read_text(encoding="utf-8")
    for expected in ["RESUMEN EA3", "Filas: 6", f"Columnas: {df.shape[1]}", "clinical_complexity_score", "FÓRMULAS", "deposit_per_stay_day: depósito"]:
        assert expected in text


def test_department_indicators_table_values():
    tables = build_result_tables(enriched(), [])
    t = tables["indicators"].set_index("department")
    assert t.loc["a", "deposito_por_dia_media"] == 50.0 and t.loc["b", "deposito_por_dia_validos"] == 2  # estancia 0 -> nulo
    assert t.loc["c", "deposito_por_dia_validos"] == 0 and t.loc["a", "visitas_por_dia_media"] == 1.25  # (2.0 + 0.5) / 2
    assert t["registros"].sum() == 6
