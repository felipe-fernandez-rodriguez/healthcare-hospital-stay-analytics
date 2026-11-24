# EA4 — Métricas y visualización de datos con Streamlit

**App publicada:** <https://healthcare-hospital-stay-analytics.streamlit.app/>

Entregables de la actividad EA4 del proyecto *Análisis descriptivo de la estancia hospitalaria y factores operativos en admisiones médicas*. La documentación completa (preguntas, fórmulas, fuentes de datos, pruebas, workflow y despliegue) está en la sección **EA4** del [README general](../README.md#ea4--métricas-y-visualización-de-datos-con-streamlit).

```text
EA4/
├── app.py                                  # streamlit run EA4/app.py
├── requirements.txt                        # dependencias livianas para Streamlit Community Cloud
├── notebooks/04-data-streamlit.ipynb       # carga, enriquecimiento (EA3), métricas, validación y prueba de la app
├── src/
│   ├── config.py                           # rutas y constantes de EA4
│   ├── formatting.py                       # formato numérico en español
│   ├── data/data_loader.py                 # lectura y validación de tablas, muestra y metadatos
│   ├── metrics/business_metrics.py         # métricas sobre registros y sobre el cubo de filtros
│   ├── metrics/insights.py                 # interpretaciones derivadas de las cifras
│   ├── metrics/table_builder.py            # construcción y exportación de tablas (usa src/ de EA1-EA3)
│   └── visualization/                      # charts.py (Plotly) · components.py · dashboard_streamlit.py
├── data/processed/healthcare_enriched_sample.csv   # muestra sistemática (1 de cada 5 filas, ≤ 100.000)
├── results/tables/                         # 17 tablas del dataset completo + run_metadata.json
├── reports/ea4-summary.txt                 # reporte de la ejecución
└── tests/test_dashboard_metrics.py         # pruebas de métricas, carga y app
```

Uso rápido, desde la raíz del repositorio:

```bash
jupyter nbconvert --to notebook --execute --inplace EA4/notebooks/04-data-streamlit.ipynb
streamlit run EA4/app.py
python -m pytest -q EA4/tests
```

Todas las métricas oficiales se calculan con el dataset completo (500.000 registros); la muestra solo alimenta vistas de registros individuales, siempre rotuladas. El análisis es descriptivo y el escenario institucional es hipotético.
