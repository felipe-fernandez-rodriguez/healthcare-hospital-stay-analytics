"""Generación de reportes de texto (UTF-8) a partir de DataFrames."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

SEP = "=" * 70


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S UTC%z")


def _write(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


class DatasetSummarizer:
    """Resume un DataFrame en un archivo ``.txt``. No se conecta a Hugging Face."""

    def __init__(self, title: str = "RESUMEN DEL DATAFRAME") -> None:
        self.title = title

    def generate(
        self,
        df: pd.DataFrame,
        dataset_info: dict[str, Any] | None = None,
        transformations: list[str] | None = None,
        exports: list[dict[str, Any]] | None = None,
    ) -> str:
        """Construye el texto del resumen (todo se calcula desde ``df``)."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame.")
        if df.empty:
            raise ValueError("El DataFrame está vacío.")
        info = dataset_info or {}
        nulls = df.isna().sum()
        lines = [self.title, SEP, f"Fecha de generación: {_now()}", ""]
        if info:
            lines += [
                f"Dataset: {info.get('dataset_id', 'no indicado')}",
                f"Configuración: {info.get('config', 'no indicada')}",
                f"Split: {info.get('split', 'no indicado')}",
                f"Carga: {info.get('muestreo', 'no indicada')}",
                "",
            ]
        lines += [
            f"Filas: {len(df)}",
            f"Columnas: {len(df.columns)}",
            f"Filas duplicadas: {int(df.duplicated().sum())}",
            "",
            "Nombres de columnas:",
            ", ".join(df.columns),
            "",
            "Tipos de datos:",
            *[f"- {c}: {t}" for c, t in df.dtypes.items()],
            "",
            "Valores nulos por columna (cantidad y porcentaje):",
            *[f"- {c}: {int(n)} ({n / len(df) * 100:.2f} %)" for c, n in nulls.items()],
        ]
        if transformations:
            lines += ["", "Transformaciones aplicadas:", *[f"- {t}" for t in transformations]]
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            lines += ["", "Estadísticas descriptivas (columnas numéricas):", numeric.describe().T.round(3).to_string()]
        categorical = df.select_dtypes(include=["category", "object", "string"])
        if not categorical.empty:
            lines += ["", "Variables categóricas (categorías únicas y más frecuente):"]
            for c in categorical.columns:
                vc = categorical[c].value_counts()
                top = f"{vc.index[0]} ({int(vc.iloc[0])})" if len(vc) else "sin valores"
                lines.append(f"- {c}: {categorical[c].nunique()} categorías; más frecuente: {top}")
        if exports:
            lines += [
                "",
                "Exportaciones CSV (el resumen corresponde al DataFrame completo, NO a los CSV):",
                *[
                    f"- {e['archivo']}: {e['filas_exportadas']} de {e['filas_dataset_completo']} filas "
                    f"({'parcial: primeras filas' if e['es_parcial'] else 'completo'}), {e['tamano_mb']} MB"
                    for e in exports
                ],
            ]
        return "\n".join(lines) + "\n"

    def save(self, df: pd.DataFrame, path: str | Path = config.DATA_SUMMARY_TXT, **kwargs: Any) -> Path:
        """Escribe el resumen en ``path`` (crea la carpeta, sobrescribe) y devuelve la ruta."""
        return _write(path, self.generate(df, **kwargs))


class ExploratoryReport:
    """Compone ``reports/exploratory-analysis.txt`` a partir de secciones calculadas en Python."""

    def __init__(self, title: str = "REPORTE DE ANÁLISIS EXPLORATORIO") -> None:
        self.title = title
        self.sections: list[tuple[str, str]] = []

    def add_section(self, heading: str, body: str | pd.DataFrame | list[str]) -> None:
        """Agrega una sección; acepta texto, lista de líneas o DataFrame."""
        if isinstance(body, pd.DataFrame):
            body = body.to_string()
        elif isinstance(body, list):
            body = "\n".join(body)
        self.sections.append((heading, body))

    def generate(self) -> str:
        lines = [self.title, SEP, f"Fecha de generación: {_now()}"]
        for i, (heading, body) in enumerate(self.sections, 1):
            lines += ["", f"{i}. {heading}", "-" * 70, body]
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path = config.EXPLORATORY_TXT) -> Path:
        return _write(path, self.generate())
