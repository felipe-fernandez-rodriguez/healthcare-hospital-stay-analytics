"""Pruebas del cargador, las transformaciones, el resumen y la exportación limitada.

Usan DataFrames sintéticos mínimos solo como entrada de prueba (no se analizan);
los datos reales se descargan en los notebooks.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.data import transformations as tr
from src.data.dataset_io import export_csv_head
from src.data.huggingface_loader import HFDatasetLoader
from src.utils.reporter import DatasetSummarizer, ExploratoryReport


def sample_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Available Extra Rooms in Hospital": [4, 2, 2, 3, 1, 4][:n],
            "Department": [" gynecology", "surgery", "surgery", "anesthesia", "radiotherapy", "gynecology"][:n],
            "Ward_Facility_Code": list("DBBAEF")[:n],
            "doctor_name": ["Dr Sophia"] * n,
            "staff_available": [0, 2, 8, 5, 10, 3][:n],
            "patientid": [1, 2, 2, -3, 5, 6][:n],
            "Age": ["41-50", "31-40", "21-30", "0-10", "51-60", "91-100"][:n],
            "gender": ["Female", "Male", "Other", "Female", "Male", "Female"][:n],
            "Type of Admission": ["Trauma"] * n,
            "Severity of Illness": ["Extreme", "Minor", "Moderate", "Minor", "Extreme", "Moderate"][:n],
            "health_conditions": ["Diabetes", None, "Other", "Asthama", None, "Diabetes"][:n],
            "Visitors with Patient": [4, 2, 2, 1, 3, 0][:n],
            "Insurance": ["Yes", "No", "Yes", "No", "Yes", "No"][:n],
            "Admission_Deposit": [2966.4, 3554.8, 5624.7, 4000.0, 4100.5, 3900.1][:n],
            "Stay (in days)": [8, 9, 7, 30, 20, 9][:n],
        }
    )


# --- transformaciones -------------------------------------------------------
def test_apply_all_keeps_rows_and_adds_group_column():
    df = sample_df()
    out, log = tr.apply_all(df)
    assert len(out) == len(df)
    assert out.shape[1] == df.shape[1] + 1
    assert "stay_in_days" in out.columns and "staff_availability_group" in out.columns
    assert [r["paso"] for r in log][0] == "strip_text_values"
    assert df["Department"].iloc[0] == " gynecology"  # el original no se modifica


def test_strip_and_nulls_preserved():
    out, rec = tr.strip_text_values(sample_df())
    assert out["Department"].iloc[0] == "gynecology"
    assert rec["valores_modificados"] == 1
    assert out["health_conditions"].isna().sum() == 2  # nulos no imputados


def test_duplicates_removed_only_if_exact():
    df = pd.concat([sample_df(), sample_df().iloc[[0]]], ignore_index=True)
    out, rec = tr.drop_exact_duplicates(df)
    assert rec["duplicados_eliminados"] == 1 and len(out) == len(df) - 1
    out2, rec2 = tr.drop_exact_duplicates(sample_df())  # patientid repetido no es duplicado exacto
    assert rec2["duplicados_eliminados"] == 0


def test_numeric_conversion_reports_invalid_and_negatives():
    df = sample_df().astype({"patientid": object})
    df.loc[0, "patientid"] = "abc"
    out, rec = tr.convert_numeric_types(df)
    assert rec["valores_no_convertibles"] == 1 and len(out) == len(df)
    assert rec["valores_negativos_detectados"] == {"patientid": 1}


def test_snake_case_and_ordered_categories():
    assert tr.to_snake_case("Stay (in days)") == "stay_in_days"
    out, _ = tr.apply_all(sample_df())
    assert out["severity_of_illness"].cat.ordered
    assert list(out["severity_of_illness"].cat.categories) == tr.SEVERITY_ORDER
    assert list(out["age"].cat.categories)[0] == "0-10"


def test_staff_groups():
    out, rec = tr.add_staff_availability_group(pd.DataFrame({"staff_available": [0, 3, 4, 6, 7, 10]}))
    assert list(out["staff_availability_group"].astype(str)) == ["Bajo (0-3)"] * 2 + ["Medio (4-6)"] * 2 + ["Alto (7+)"] * 2


# --- cargador ---------------------------------------------------------------
def test_loader_rejects_invalid_id():
    with pytest.raises(ValueError):
        HFDatasetLoader("sin-barra")


def test_loader_validates_split_and_config():
    loader = HFDatasetLoader("a/b", config="default", split="test", token="x")
    with mock.patch.object(HFDatasetLoader, "available_splits", return_value={"default": ["train"]}):
        with pytest.raises(ValueError, match="Split"):
            loader.load()
    loader = HFDatasetLoader("a/b", config="otra", split="train", token="x")
    with mock.patch.object(HFDatasetLoader, "available_splits", return_value={"default": ["train"]}):
        with pytest.raises(ValueError, match="Configuración"):
            loader.load()


def test_loader_returns_all_rows_as_dataframe():
    fake = mock.Mock()
    fake.__len__ = lambda self: 6
    fake.to_pandas.return_value = sample_df()
    with mock.patch.object(HFDatasetLoader, "available_splits", return_value={"default": ["train"]}), \
         mock.patch("src.data.huggingface_loader.load_dataset", return_value=fake) as ld:
        loader = HFDatasetLoader("a/b", split="train", token="x")
        df = loader.load()
    assert len(df) == 6 and loader.info()["muestreo"].startswith("dataset completo")
    assert ld.call_args.args[1] == "default" and ld.call_args.kwargs["split"] == "train"


def test_loader_info_requires_load():
    with pytest.raises(RuntimeError):
        HFDatasetLoader("a/b", token="x").info()


# --- exportación limitada ---------------------------------------------------
def test_export_csv_head_is_deterministic_and_capped(tmp_path):
    df = pd.DataFrame({"x": range(10)})
    info = export_csv_head(df, tmp_path / "out" / "a.csv", max_rows=4)
    saved = pd.read_csv(tmp_path / "out" / "a.csv")
    assert list(saved["x"]) == [0, 1, 2, 3]
    assert info["filas_exportadas"] == 4 and info["filas_dataset_completo"] == 10 and info["es_parcial"]
    assert len(df) == 10
    assert not export_csv_head(df, tmp_path / "b.csv", max_rows=50)["es_parcial"]


def test_csv_limit_constant():
    assert config.CSV_MAX_ROWS == 100_000


# --- reportes ---------------------------------------------------------------
def test_summarizer_content_and_utf8(tmp_path):
    df, log = tr.apply_all(sample_df())
    target = DatasetSummarizer().save(
        df,
        tmp_path / "nueva" / "resumen.txt",
        dataset_info={"dataset_id": "a/b", "config": "default", "split": "train", "muestreo": "dataset completo"},
        transformations=tr.describe_log(log),
    )
    text = target.read_text(encoding="utf-8")
    for expected in ["a/b", "Split: train", "Filas: 6", "Columnas: 16", "Filas duplicadas: 0", "health_conditions: 2 (33.33 %)", "Transformaciones aplicadas"]:
        assert expected in text


def test_summarizer_errors():
    with pytest.raises(TypeError):
        DatasetSummarizer().generate([1, 2])
    with pytest.raises(ValueError):
        DatasetSummarizer().generate(pd.DataFrame())


def test_exploratory_report_saves_sections(tmp_path):
    rep = ExploratoryReport()
    rep.add_section("TABLA", pd.DataFrame({"a": [1]}))
    rep.add_section("LISTA", ["uno", "dos"])
    text = rep.save(tmp_path / "r" / "e.txt").read_text(encoding="utf-8")
    assert "1. TABLA" in text and "2. LISTA" in text and "dos" in text
