"""Pruebas de EA4: métricas, cubo de filtros, carga de resultados y app de Streamlit.

* Unitarias: DataFrames pequeños cuyos resultados se calculan a mano (sin Internet).
* Integración: usan los archivos reales de EA4 si existen (se omiten si aún no se generaron,
  p. ej. en el job de pruebas de GitHub Actions antes de ejecutar el notebook 04).
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from EA4.src import config as ea4  # noqa: E402
from EA4.src.data import data_loader as dl  # noqa: E402
from EA4.src.metrics import business_metrics as bm  # noqa: E402
from EA4.src.metrics import insights as ins  # noqa: E402
from EA4.src.metrics import table_builder as tb  # noqa: E402
from EA4.src.visualization import charts as ch  # noqa: E402
from EA4.src.visualization.dashboard_streamlit import Filters, HospitalDashboard  # noqa: E402
from src.data import enrichment as en  # noqa: E402
from src.data import transformations as tr  # noqa: E402

REAL_FILES = all((ea4.TABLES_DIR / f).exists() for f in ea4.TABLE_FILES.values()) and ea4.METADATA_JSON.exists()
needs_files = pytest.mark.skipif(not REAL_FILES, reason="Los resultados de EA4 aún no se han generado (ejecute el notebook 04).")


# ======================================================================== datos de prueba
def small() -> pd.DataFrame:
    """Registros ya enriquecidos mínimos (6 filas) con respuestas calculables a mano."""
    return pd.DataFrame({
        "department": ["a", "a", "a", "b", "b", "c"],
        "severity_of_illness": ["Minor", "Extreme", "Minor", "Minor", "Extreme", "Minor"],
        "staff_availability_group": ["Bajo (0-3)"] * 3 + ["Alto (7+)"] * 3,
        "type_of_admission": ["Trauma"] * 6,
        "stay_in_days": [2.0, 4.0, 6.0, 10.0, 20.0, np.nan],
        "admission_deposit": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        "staff_available": [0, 1, 2, 8, 9, 10],
    })


def raw_like(n_rep: int = 4) -> pd.DataFrame:
    """Registros con la estructura original del dataset (para pasar por EA1-EA3 de verdad)."""
    base = pd.DataFrame({
        "Available Extra Rooms in Hospital": [3, 4, 2, 5, 1, 6, 2, 3, 4, 5, 2, 3],
        "Department": ["gynecology", "gynecology", "surgery", "surgery", "anesthesia", "anesthesia",
                       "radiotherapy", "radiotherapy", "TB & Chest disease", "gynecology", "surgery", "anesthesia"],
        "Ward_Facility_Code": list("ABCDEFABCDEF"),
        "doctor_name": ["Dr X"] * 12,
        "staff_available": [0, 3, 4, 6, 7, 10, 1, 5, 8, 2, 9, 4],
        "patientid": list(range(1, 13)),
        "Age": ["21-30", "31-40", "41-50", "51-60", "61-70", "0-10", "21-30", "31-40", "41-50", "11-20", "71-80", "21-30"],
        "gender": ["Female", "Male", "Other", "Female", "Male", "Female", "Female", "Male", "Other", "Female", "Male", "Female"],
        "Type of Admission": ["Trauma", "Emergency", "Urgent", "Trauma", "Emergency", "Urgent"] * 2,
        "Severity of Illness": ["Minor", "Moderate", "Extreme"] * 4,
        "health_conditions": ["Diabetes", None, "Other", "Asthama", None, "Heart disease"] * 2,
        "Visitors with Patient": [2, 3, 4, 1, 5, 2, 3, 2, 4, 1, 6, 2],
        "Insurance": ["Yes", "No"] * 6,
        "Admission_Deposit": [4000.0, 4500.0, 5000.0, 3500.0, 6000.0, 4200.0, 4800.0, 5100.0, 3900.0, 4400.0, 5300.0, 4700.0],
        "Stay (in days)": [8, 9, 35, 30, 28, 25, 20, 16, 22, 10, 40, 31],
    })
    out = pd.concat([base] * n_rep, ignore_index=True)
    out["patientid"] = range(1, len(out) + 1)  # evita duplicados exactos
    return out


@pytest.fixture(scope="module")
def synthetic():
    raw = raw_like()
    processed, tlog = tr.apply_all(raw)
    df, elog = en.apply_enrichment(processed)
    return raw, df, tlog, elog, tb.build_ea4_tables(df, elog)


# ======================================================================== métricas sobre registros
def test_patient_days_share_hand_calculated():
    t = bm.patient_days_share(small()).set_index("department")
    # a: 2+4+6 = 12 ; b: 10+20 = 30 ; c: estancia nula -> 0 días-paciente (el nulo no es cero en la media)
    assert t.loc["a", "dias_paciente"] == 12 and t.loc["b", "dias_paciente"] == 30 and t.loc["c", "dias_paciente"] == 0
    assert t.loc["a", "pct_dias_paciente"] == pytest.approx(12 / 42 * 100)
    assert t.loc["a", "pct_admisiones"] == pytest.approx(50.0)
    assert t.loc["c", "estancia_validos"] == 0 and math.isnan(t.loc["c", "estancia_media"])
    assert t["pct_dias_paciente"].sum() == pytest.approx(100.0)


def test_expected_stay_by_composition_formula():
    t = bm.expected_stay_by_composition(small()).set_index("severity_of_illness")
    # medias de departamento (válidas): a = 4, b = 15. Minor válidos: a(2), a(6), b(10) -> esperada (4 + 4 + 15) / 3
    assert t.loc["Minor", "estancia_esperada_composicion"] == pytest.approx(23 / 3)
    assert t.loc["Minor", "estancia_observada_media"] == pytest.approx(6.0)
    assert t.loc["Extreme", "estancia_esperada_composicion"] == pytest.approx((4 + 15) / 2)
    assert t.loc["Minor", "registros"] == 4 and t.loc["Minor", "estancia_validos"] == 3  # la fila nula cuenta como registro
    assert t["pct_registros"].sum() == pytest.approx(100.0)
    assert list(t.index) == ["Minor", "Extreme"]  # orden documentado de severidad


def test_eta_squared_definition_and_r2_equivalence():
    df = pd.DataFrame({"g": ["x", "x", "y", "y"], "stay_in_days": [1.0, 3.0, 5.0, 7.0]})
    r = bm.eta_squared(df, "g", "stay_in_days")
    # media 4; SS_total = 9+1+1+9 = 20; SS_entre = 2(2-4)² + 2(6-4)² = 16 -> 0,8
    assert r["eta2"] == pytest.approx(0.8) and r["n_validos"] == 4 and r["n_grupos"] == 2
    rng = np.random.default_rng(0)
    g = rng.choice(list("abc"), 300)
    y = rng.normal(size=300) + (g == "b") * 1.5
    X = pd.get_dummies(pd.Series(g), dtype=float).to_numpy()
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert bm.eta_squared(pd.DataFrame({"g": g, "v": y}), "g", "v")["eta2"] == pytest.approx(r2)


def test_eta_squared_ignores_invalid_target_and_keeps_null_groups():
    df = pd.DataFrame({"g": ["x", None, "y", "y"], "stay_in_days": [1.0, 3.0, np.nan, 7.0]})
    r = bm.eta_squared(df, "g", "stay_in_days")
    assert r["n_validos"] == 3 and r["n_grupos"] == 3  # grupo «(sin dato)» incluido


def test_spearman_uses_valid_pairs_and_average_ranks():
    df = pd.DataFrame({"x": [1, 2, 2, 4, np.nan, 6], "y": [10, 20, 30, 30, 50, np.nan]})
    r = bm.spearman(df, "x", "y")
    # pares válidos: x = 1, 2, 2, 4 -> rangos 1, 2.5, 2.5, 4 ; y = 10, 20, 30, 30 -> rangos 1, 2, 3.5, 3.5
    expected = np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3.5, 3.5])[0, 1]
    assert r["n_pares_validos"] == 4
    assert r["rho_spearman"] == pytest.approx(expected)


def test_distribution_summary_tukey_outliers():
    s = pd.DataFrame({"v": [1, 2, 3, 4, 100]})
    t = bm.distribution_summary(s, ["v"]).iloc[0]
    assert t["q1"] == 2 and t["q3"] == 4 and t["limite_superior_tukey"] == 7 and t["atipicos"] == 1
    assert t["atipicos_pct"] == pytest.approx(20.0)
    centred = bm.distribution_summary(pd.DataFrame({"d": [-1.0, 0.0, 1.0]}), ["d"]).iloc[0]
    assert math.isnan(centred["cv_pct"])  # el CV no se calcula para diferencias centradas en cero


def test_histogram_counts_all_valid_values():
    h = bm.histogram_table(pd.Series([3, 3, 5, np.nan]), "stay", integer=True)
    assert h["registros"].tolist() == [2, 0, 1] and h["registros"].sum() == 3
    h2 = bm.histogram_table(pd.Series(np.linspace(0, 1, 101)), "x", bins=10)
    assert h2["registros"].sum() == 101 and len(h2) == 10


# ======================================================================== cubo de filtros
def test_weighted_quantile_matches_pandas():
    values = pd.Series([3, 3, 4, 9, 9, 9, 12, 15, 15, 40])
    vc = values.value_counts()
    for q in (0.25, 0.5, 0.75):
        assert bm.weighted_quantile(vc.index.to_numpy(), vc.to_numpy(), q) == pytest.approx(values.quantile(q))


def test_cube_reproduces_record_level_metrics(synthetic):
    _, df, _, _, T = synthetic
    cube = T["filter_cube"]
    assert cube["registros"].sum() == len(df)
    a = bm.cube_group_metrics(cube, "department").set_index("department")
    for dept, g in df.groupby("department", observed=True):
        assert a.loc[dept, "registros"] == len(g)
        assert a.loc[dept, "estancia_dias_media"] == pytest.approx(g["stay_in_days"].mean())
        assert a.loc[dept, "estancia_dias_mediana"] == pytest.approx(g["stay_in_days"].median())
        assert a.loc[dept, "deposito_admision_media"] == pytest.approx(g["admission_deposit"].mean())
        if len(g) > 1:
            assert a.loc[dept, "estancia_dias_desv_est"] == pytest.approx(g["stay_in_days"].std())
            assert a.loc[dept, "deposito_admision_desv_est"] == pytest.approx(g["admission_deposit"].std())
    pdays = T["patient_days"].set_index("department")
    assert np.allclose(a["pct_dias_paciente"], pdays.loc[a.index, "pct_dias_paciente"])
    comp = bm.cube_expected_stay(cube).set_index("severity_of_illness")["estancia_esperada_composicion"]
    ref = T["severity_composition"].set_index("severity_of_illness")["estancia_esperada_composicion"]
    assert np.allclose(comp, ref.loc[comp.index])


def test_cube_filters_match_filtered_records(synthetic):
    _, df, _, _, T = synthetic
    cube = T["filter_cube"]
    sub = df[(df["severity_of_illness"].astype(str) == "Minor") & (df["stay_range"].astype(str) == "Prolongada (>7 días)")]
    k = bm.cube_kpis(bm.filter_cube(cube, {"severity_of_illness": ["Minor"]}, ["Prolongada (>7 días)"]))
    assert k["registros"] == len(sub)
    assert k["estancia_dias_mediana"] == pytest.approx(sub["stay_in_days"].median())
    sr = bm.cube_stay_range_table(cube)
    pd.testing.assert_frame_equal(sr, T["stay_range"], check_dtype=False)


def test_cross_tab_consistency(synthetic):
    _, _, _, _, T = synthetic
    cube = T["filter_cube"]
    cross = bm.cube_group_metrics(cube, ["department", "severity_of_illness"])
    by_dept = bm.cube_group_metrics(cube, "department").set_index("department")["registros"]
    assert (cross.groupby("department")["registros"].sum() == by_dept.loc[cross["department"].unique()]).all()
    assert cross["registros"].sum() == cube["registros"].sum()


def test_severity_counts_and_percentages(synthetic):
    _, df, _, _, T = synthetic
    sev = bm.cube_group_metrics(T["filter_cube"], "severity_of_illness")
    assert sev["severity_of_illness"].tolist() == ["Minor", "Moderate", "Extreme"]
    assert sev["pct_registros"].sum() == pytest.approx(100.0)
    assert sev.set_index("severity_of_illness").loc["Minor", "registros"] == (df["severity_of_illness"] == "Minor").sum()


def test_null_stays_are_excluded_not_zero():
    cube = bm.build_cube(small().assign(type_of_admission="Trauma"))
    k = bm.cube_kpis(cube)
    assert k["registros"] == 6 and k["estancia_dias_validos"] == 5
    assert k["estancia_dias_media"] == pytest.approx(42 / 5)  # el nulo no entra como 0


# ======================================================================== tablas y exportación
def test_build_tables_reuse_ea3_and_cover_questions(synthetic):
    _, df, _, _, T = synthetic
    assert set(T) == set(ea4.TABLE_FILES)
    assert T["department"]["registros"].sum() == len(df)
    assert set(T["correlations"]["n_pares_validos"]) == {len(df)}
    assert T["eta_squared"]["eta2"].between(0, 1).all()
    assert set(T["pc2_groups"]["variable"]) == set(tb.PC2_FACTORS)


def test_missing_columns_are_reported(synthetic):
    _, df, _, _, _ = synthetic
    with pytest.raises(KeyError, match="stay_range"):
        tb.build_ea4_tables(df.drop(columns=["stay_range"]))


def test_export_tables_and_csv_roundtrip(synthetic, tmp_path):
    _, _, _, _, T = synthetic
    inv = tb.export_tables(T, tmp_path / "nueva" / "tablas")
    assert len(inv) == len(ea4.TABLE_FILES) and (inv["filas"] > 0).all()
    for key, table in T.items():
        text = (tmp_path / "nueva" / "tablas" / ea4.TABLE_FILES[key]).read_text(encoding="utf-8").replace("\r\n", "\n")
        assert text == table.to_csv(index=False).replace("\r\n", "\n")
    loaded = dl.load_table("department", tmp_path / "nueva" / "tablas")
    assert loaded.attrs["fuente"] == "EA4" and loaded["registros"].sum() == T["department"]["registros"].sum()


def test_sample_is_systematic_and_capped(tmp_path):
    df = pd.DataFrame({"x": range(250_000)})
    info = tb.export_sample(df, tmp_path / "s" / "muestra.csv")
    saved = pd.read_csv(tmp_path / "s" / "muestra.csv")
    assert len(saved) == info["filas_exportadas"] <= ea4.SAMPLE_MAX_ROWS
    assert info["paso"] == 3 and saved["x"].head(3).tolist() == [0, 3, 6]  # determinista, sin azar
    assert info["es_parcial"] and info["filas_dataset_completo"] == 250_000
    small_info = tb.export_sample(df.head(10), tmp_path / "s" / "pequena.csv")
    assert small_info["filas_exportadas"] == 10 and not small_info["es_parcial"]


# ======================================================================== carga y recuperación
def test_missing_file_error_explains_how_to_rebuild(tmp_path):
    with pytest.raises(dl.DataUnavailableError) as err:
        dl.load_table("patient_days", tmp_path)
    assert "nbconvert" in str(err.value) and "04-data-streamlit.ipynb" in str(err.value)


def test_wrong_columns_are_detected(tmp_path):
    pd.DataFrame({"x": [1]}).to_csv(tmp_path / ea4.TABLE_FILES["patient_days"], index=False)
    with pytest.raises(dl.DataUnavailableError, match="Faltan las columnas"):
        dl.load_table("patient_days", tmp_path)


@pytest.mark.skipif(not (dl.EA3_RESULTS_DIR / ea4.TABLE_FILES["department"]).exists(), reason="EA3 no generado")
def test_fallback_to_current_ea3_results(tmp_path):
    path, source = dl.resolve_table_path("department", tmp_path)
    assert source == "EA3" and path.parent == dl.EA3_RESULTS_DIR
    assert dl.resolve_table_path("patient_days", tmp_path)[1] == "faltante"  # tabla exclusiva de EA4


def test_missing_data_panel_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    def app():
        from pathlib import Path

        from EA4.src.data.data_loader import DataUnavailableError
        from EA4.src.visualization.components import missing_data_panel
        missing_data_panel(DataUnavailableError(Path("EA4/results/tables/x.csv"), "No existe el archivo de resultados"))

    at = AppTest.from_function(app).run()
    assert not at.exception and len(at.error) == 1 and len(at.code) == 1


# ======================================================================== dashboard (sin archivos)
def test_dashboard_views_and_insights_with_synthetic_tables(synthetic):
    _, df, _, _, T = synthetic
    dash = HospitalDashboard(T, {}, None)
    assert dash.total_records == len(df)
    f = Filters({"department": ["surgery"]}, [])
    assert f.active and "surgery" in f.describe() and not Filters().active
    cube = dash.filtered_cube(f)
    assert cube["registros"].sum() == (df["department"] == "surgery").sum()
    assert dash.dept_view(cube, filtered=True)["deposito_admision_mediana"].isna().all()  # no recuperable desde el cubo
    texts = dash.official_insights()
    assert set(texts) == set(ins.QUESTIONS)
    for item in texts.values():
        assert {"metrica", "patron", "respuesta", "decision", "limitacion"} <= set(item)
    assert "menos de dos categorías" in ins.p1_departments(dash.dept_view(cube, True))["respuesta"]


def test_charts_build_with_titles_and_units(synthetic):
    _, _, _, _, T = synthetic
    cube = T["filter_cube"]
    figs = [
        ch.dept_bars(bm.cube_group_metrics(cube, "department"), "estancia_dias_media", "T", "Estancia media (días)", "s"),
        ch.heatmap(bm.cube_group_metrics(cube, ["department", "severity_of_illness"]), "estancia_dias_media", "T", "días", "s"),
        ch.eta_bars(T["eta_squared"], "stay_in_days", "s"),
        ch.quality_nulls(T["data_quality"], "s"),
        ch.histogram(T["histograms"], "admission_deposit", "Depósito", "T", "s"),
    ]
    for fig in figs:
        assert fig.layout.title.text and fig.layout.separators == ",."


# ======================================================================== integración con los archivos reales
@needs_files
def test_real_files_are_consistent_between_notebook_and_app():
    dash = HospitalDashboard.from_files()
    meta = dl.load_metadata()
    assert dash.total_records == meta["filas_analizadas"]
    a = bm.cube_group_metrics(dash.cube, "department").set_index("department")
    b = dash.t["department"].set_index("department").loc[a.index]
    assert (a["estancia_dias_media"].round(2) - b["estancia_dias_media"]).abs().max() < 0.006
    assert (a["registros"] == b["registros"]).all()
    status = dl.file_status()
    assert status["existe"].all()


@needs_files
def test_real_sample_respects_row_and_size_limits():
    assert dl.load_sample().shape[0] <= ea4.SAMPLE_MAX_ROWS
    assert ea4.SAMPLE_CSV.stat().st_size / 1e6 < ea4.MAX_FILE_MB
    for f in ea4.TABLES_DIR.glob("*.csv"):
        assert f.stat().st_size / 1e6 < ea4.MAX_FILE_MB


@needs_files
def test_app_runs_every_page_and_filters():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ea4.EA4_DIR / "app.py"), default_timeout=240).run()
    assert not at.exception, at.exception
    for page in ["inicio", "pc1", "p1", "p2", "p3", "p4", "p5", "pc2", "pc3", "fin"]:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)
    at.sidebar.radio[0].set_value("p1").run()
    at.sidebar.multiselect[1].set_value(["Extreme"]).run()
    assert not at.exception and len(at.get("plotly_chart")) > 0


def test_app_modules_do_not_require_datasets():
    """La app debe poder desplegarse con EA4/requirements.txt (sin datasets ni la descarga de Hugging Face)."""
    code = ("import sys; sys.path.insert(0, r'%s'); import EA4.src.visualization.dashboard_streamlit, "
            "EA4.src.data.data_loader; print('datasets' in sys.modules, 'src.data.huggingface_loader' in sys.modules)") % ROOT
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


# ======================================================================== seguridad
def test_no_hugging_face_tokens_in_tracked_files():
    pattern = re.compile(r"hf_[A-Za-z0-9]{30,}")
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        paths = [ROOT / f for f in files] + list((ROOT / "EA4").rglob("*.py"))
    except (OSError, subprocess.CalledProcessError):
        paths = [p for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts
                 and p.name != ".env"]
    offenders = []
    for p in set(paths):
        if p.suffix in {".csv", ".png", ".docx"} or not p.exists() or p.name == ".env":
            continue
        if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"Posibles tokens en: {offenders}"
