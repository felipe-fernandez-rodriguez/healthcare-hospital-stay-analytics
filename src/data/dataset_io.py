"""Flujo de datos compartido: descarga completa, transformación y exportación limitada."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src import config
from src.data.huggingface_loader import HFDatasetLoader
from src.data.transformations import apply_all


def make_loader() -> HFDatasetLoader:
    """Loader del dataset del proyecto (siempre completo, sin muestreo)."""
    return HFDatasetLoader(config.DATASET_ID, config=config.DATASET_CONFIG, split=config.DATASET_SPLIT)


def download_and_transform() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Descarga el dataset completo y lo transforma, sin escribir archivos.

    Returns:
        ``(original, transformado, registro_de_transformaciones, info_de_carga)``.
    """
    loader = make_loader()
    raw = loader.load()
    processed, log = apply_all(raw)
    return raw, processed, log, loader.info()


def export_csv_head(df: pd.DataFrame, path: str | Path, max_rows: int = config.CSV_MAX_ROWS) -> dict[str, Any]:
    """Exporta como máximo las primeras ``max_rows`` filas de ``df`` a CSV (determinista).

    El DataFrame original no se modifica. El resultado indica si el CSV es parcial.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = df.head(max_rows)
    out.to_csv(target, index=False, encoding="utf-8")
    root = config.PROJECT_ROOT
    return {
        "archivo": target.resolve().relative_to(root).as_posix() if target.resolve().is_relative_to(root) else str(target),
        "filas_dataset_completo": len(df),
        "filas_exportadas": len(out),
        "es_parcial": len(out) < len(df),
        "tamano_mb": round(target.stat().st_size / 1e6, 2),
    }
