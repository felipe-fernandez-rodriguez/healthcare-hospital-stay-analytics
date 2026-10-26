"""Pipeline EA1 por script: Hugging Face -> transformaciones -> CSV limitados -> resumen TXT.

Uso (desde la raíz del proyecto)::

    python -m src.pipeline
"""
from __future__ import annotations

from src import config
from src.data.dataset_io import download_and_transform, export_csv_head
from src.data.transformations import describe_log
from src.utils.reporter import DatasetSummarizer


def main() -> None:
    raw, processed, log, info = download_and_transform()
    exports = [export_csv_head(raw, config.RAW_CSV), export_csv_head(processed, config.PROCESSED_CSV)]
    DatasetSummarizer().save(
        processed,
        config.DATA_SUMMARY_TXT,
        dataset_info=info,
        transformations=describe_log(log),
        exports=exports,
    )
    print(f"Filas completas: {len(raw)} -> transformadas: {len(processed)}")
    for e in exports:
        print(e)
    print(f"Resumen -> {config.DATA_SUMMARY_TXT}")


if __name__ == "__main__":
    main()
