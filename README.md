# Análisis descriptivo de la estancia hospitalaria y factores operativos en admisiones médicas

Proyecto académico (EA1 – carga y creación de dataset; EA2 – ideación, diseño y documentación de la necesidad; EA3 – transformación y enriquecimiento; EA4 – métricas y visualización con Streamlit). Carga el dataset [`Nicolybgs/healthcare_data`](https://huggingface.co/datasets/Nicolybgs/healthcare_data) desde Hugging Face, lo transforma, genera reportes `.txt` y realiza una exploración descriptiva.

**Autor:** Felipe Fernández Rodriguez · Institución Universitaria Digital de Antioquia · Analítica de Datos.

## Necesidad analítica y pregunta (escenario hipotético)

Se propone —en un escenario **hipotético**— que la Dirección de Gestión Operativa y Control de Camas de un centro hospitalario necesita caracterizar cómo se distribuyen la duración de la estancia (`Stay (in days)`) y el depósito de admisión (`Admission_Deposit`) según variables clínicas y operativas. No se trata de un problema observado y comprobado en una institución real.

> **Pregunta:** ¿Cómo varían la duración de la estancia hospitalaria y el valor del depósito de admisión según el departamento médico, el nivel de severidad de la enfermedad y los recursos de personal disponibles en los registros del dataset seleccionado?

El enfoque es descriptivo: las diferencias observadas **no** implican causalidad.

## Dataset

- **Identificador:** `Nicolybgs/healthcare_data` · **Enlace:** <https://huggingface.co/datasets/Nicolybgs/healthcare_data>
- **Configuración / split:** `default` / `train` (únicos disponibles, verificados por código). **500.000 filas × 15 columnas.**
- **Ficha (consultada el 25 de octubre de 2026):** el dataset **no tiene dataset card**. No documenta licencia, procedencia, si los datos son reales o sintéticos, periodo cubierto ni el significado/unidades de las columnas; todo ello está **pendiente de verificar** y este proyecto no lo afirma. Antes de redistribuir datos, confirme las condiciones de uso con el autor.
- **Acceso:** público; **no requiere token**. Opcionalmente puede definir `HF_TOKEN` (ver más abajo).

## Estructura del repositorio

```text
healthcare-hospital-stay-analytics/
├── .github/workflows/ml-pipeline.yml   # GitHub Actions: pruebas + ejecución de notebooks
├── data/
│   ├── raw/                            # CSV original (primeras 100.000 filas) – lo genera Actions
│   └── processed/                      # CSV transformado (EA1, ~10 MB, ≤ 100.000 filas) – lo genera Actions
├── .streamlit/config.toml              # tema de la app de EA4
├── EA3/                                # entregables de EA3 (ver EA3/README.md)
│   ├── notebooks/03-data-transformation-enrichment.ipynb
│   ├── data/healthcare_enriched_sample.csv   # muestra ≤ 100.000 filas
│   ├── results/                        # 8 tablas agregadas (dataset completo)
│   └── reports/data-enrichment-summary.txt
├── EA4/                                # métricas y dashboard de Streamlit (ver sección EA4)
│   ├── app.py                          # punto de entrada: streamlit run EA4/app.py
│   ├── requirements.txt                # dependencias livianas para desplegar la app
│   ├── notebooks/04-data-streamlit.ipynb
│   ├── src/                            # data/data_loader.py · metrics/ · visualization/
│   ├── data/processed/healthcare_enriched_sample.csv   # muestra sistemática ≤ 100.000 filas
│   ├── results/tables/                 # 17 tablas + cubo de filtros + run_metadata.json
│   ├── reports/ea4-summary.txt
│   └── tests/test_dashboard_metrics.py
├── docs/
│   └── proyecto_analisis_estancia_hospitalaria.pdf        # documento del proyecto (PDF)
├── notebooks/
│   ├── 01-data-ingestion.ipynb         # EA1: carga, transformaciones, CSV y resumen
│   └── 02-data-description.ipynb       # EA2: exploración descriptiva + reporte
├── reports/
│   ├── data-summary.txt               # resumen del DataFrame transformado COMPLETO (EA1)
│   └── exploratory-analysis.txt       # reporte del análisis exploratorio (EA2)
├── src/
│   ├── config.py                       # rutas relativas y constantes
│   ├── pipeline.py                     # EA1 por script: python -m src.pipeline
│   ├── data/
│   │   ├── huggingface_loader.py       # HFDatasetLoader (solo carga)
│   │   ├── transformations.py          # transformaciones reutilizables (EA1/EA2)
│   │   ├── enrichment.py               # variables derivadas (EA3)
│   │   └── dataset_io.py               # descarga + transformación + exportación limitada
│   ├── analysis/
│   │   ├── descriptive.py              # tablas descriptivas (notebooks 02 y 03)
│   │   ├── quality.py                  # comparación antes/después y calidad (EA3)
│   │   └── results.py                  # tablas de resultados, exportación y hallazgos (EA3)
│   └── utils/reporter.py               # DatasetSummarizer y ExploratoryReport
├── tests/                              # pruebas con pytest (test_pipeline.py: EA1/EA2; test_enrichment.py: EA3; EA4 en EA4/tests)
├── .env.example · .gitignore · requirements.txt
```

## Flujo

1. **Notebook 01** – `HFDatasetLoader` descarga el dataset **completo** → DataFrame original → 6 transformaciones (`src/data/transformations.py`) → DataFrame transformado → `DatasetSummarizer` escribe `reports/data-summary.txt` y se exportan los CSV.
2. **Notebook 02** – **vuelve a descargar** el dataset completo, lo transforma en memoria con las mismas funciones y explora estancia/depósito por departamento, severidad y personal; genera `reports/exploratory-analysis.txt`. **No lee los CSV de `data/`.**

### Transformaciones (registran filas antes/después)

| Transformación | Resultado en este dataset |
|---|---|
| Quitar espacios en valores de texto | Sin casos que corregir |
| Conversión validada a tipos numéricos | Sin valores no convertibles; se reportan negativos en `patientid` |
| Detección de duplicados exactos | 0 duplicados (`patientid` se repite, pero no es duplicado exacto) |
| Nombres de columna a `snake_case` | 15 columnas renombradas (con correspondencia original) |
| Categorías estandarizadas (`category`; severidad y edad ordenadas) | Los nulos de `health_conditions` se conservan, no se imputan |
| Enriquecimiento: `staff_availability_group` (Bajo 0-3, Medio 4-6, Alto 7+) | Umbrales definidos por el proyecto, no por la ficha |

### Límite de 100.000 filas

El límite se aplica **solo a los CSV exportados** (primeras 100.000 filas, sin aleatoriedad) para no superar el peso permitido por GitHub (aviso a 50 MB, bloqueo a 100 MB). Los DataFrames, los resúmenes y la exploración usan las 500.000 filas. Los CSV **no representan el dataset completo**.

- `data/raw/healthcare_raw.csv` (~9 MB) y `data/processed/healthcare_processed.csv` (~10 MB): se versionan; los genera el workflow (o el notebook 01 en local).
- Los archivos de EA3 (`EA3/data/`, `EA3/results/`, `EA3/reports/`) también se versionan (todos < 50 MB).
- `reports/*.txt` y los notebooks ejecutados también se versionan como evidencia.

## Instalación y ejecución

Requisitos: Python 3.12 o superior (probado con 3.14) y conexión a Internet.

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ejecución en orden (desde la raíz; los notebooks localizan la raíz solos):

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01-data-ingestion.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02-data-description.ipynb
jupyter nbconvert --to notebook --execute --inplace EA3/notebooks/03-data-transformation-enrichment.ipynb   # EA3
jupyter nbconvert --to notebook --execute --inplace EA4/notebooks/04-data-streamlit.ipynb                  # EA4
streamlit run EA4/app.py                                                                                  # dashboard
```

o ábralos con `jupyter lab`. Alternativa por script para EA1: `python -m src.pipeline`.

**Token (opcional):** el dataset es público y el proyecto funciona sin él. Para evitar límites de uso anónimo copie `.env.example` a `.env` y complete `HF_TOKEN`, o defina la variable de entorno. `.env` está en `.gitignore`; nunca se imprime ni se sube.

**Documento del proyecto (PDF):** `docs/proyecto_analisis_estancia_hospitalaria.pdf`.

## EA3 — Transformación y enriquecimiento de datos

Los entregables de EA3 están organizados en la carpeta **[`EA3/`](EA3/)** (documentación detallada en [`EA3/README.md`](EA3/README.md)):

| Entregable | Ruta |
|---|---|
| Notebook | `EA3/notebooks/03-data-transformation-enrichment.ipynb` |
| Muestra del dataset enriquecido (≤ 100.000 filas) | `EA3/data/healthcare_enriched_sample.csv` |
| Tablas de resultados (8 CSV) | `EA3/results/` |
| Reporte TXT del DataFrame enriquecido completo | `EA3/reports/data-enrichment-summary.txt` |

Reutiliza `HFDatasetLoader` y las transformaciones de EA1/EA2 y agrega `src/data/enrichment.py`. Los cálculos y las tablas corresponden al **dataset completo** (500.000 filas); solo el CSV de muestra se limita a 100.000 filas. Los archivos de EA1 y EA2 no se modifican. Ejecución:

```bash
jupyter nbconvert --to notebook --execute --inplace EA3/notebooks/03-data-transformation-enrichment.ipynb
```

## EA4 — Métricas y Visualización de Datos con Streamlit

### Objetivo y relación con las preguntas de negocio

Presentar, en una aplicación interactiva de Streamlit, las respuestas a la pregunta principal y a las preguntas P1-P5 (EA3) y PC1-PC3 (EA2) mediante indicadores, tablas y gráficos de Plotly calculados con el **dataset completo**. Cada sección muestra la métrica, su fórmula, el universo de datos y una interpretación generada con las cifras cargadas (sin números escritos a mano).

| ID | Pregunta | Tabla principal (`EA4/results/tables/`) | Visual |
|---|---|---|---|
| P1 | Estancia y depósito por departamento | `results_by_department.csv`, `results_patient_days_by_department.csv` | Barras, % admisiones vs. % días-paciente, cajas |
| P2 | Diferencias por severidad | `results_by_severity.csv`, `results_severity_expected_stay.csv` | Observada vs. esperada por composición, puntos, cajas |
| P3 | Disponibilidad de personal | `results_by_staff_availability.csv`, `results_by_staff_level.csv`, `results_spearman_correlations.csv` | Puntos, barras por nivel, KPI de ρ |
| P4 | Departamento × severidad | `results_by_department_severity.csv`, `results_filter_cube.csv` | Mapa de calor (métrica seleccionable) |
| P5 | Indicadores derivados de EA3 | `results_by_stay_range.csv`, `results_by_complexity_score.csv`, `results_by_department_indicators.csv`, `results_histograms.csv` | Barras, histogramas, caja con la muestra (rotulada) |
| PC1 | Distribución de estancia y depósito | `results_distribution_summary.csv`, `results_histograms.csv` | KPI e histogramas |
| PC2 | Otras variables asociadas | `results_pc2_group_metrics.csv`, `results_eta_squared.csv`, `results_spearman_correlations.csv` | Barras de η² y ρ, barras por categoría |
| PC3 | Calidad de los datos | `results_data_quality.csv`, `run_metadata.json` | % de nulos y tabla de advertencias |

**Indicadores nuevos** (fórmulas del Anexo A del informe): días-paciente = suma de estancias válidas del grupo y su participación (días-paciente del grupo / total × 100); estancia esperada por composición = Σ_d (registros del departamento d en la categoría × estancia media de d) / registros de la categoría; razón de varianza η² = Σ_g n_g (media_g − media)² / Σ_i (y_i − media)²; ρ de Spearman con pares válidos; atípicos por la regla de Tukey (se conservan).

### Estructura y función de cada componente

| Archivo | Función |
|---|---|
| `EA4/notebooks/04-data-streamlit.ipynb` | Descarga el dataset completo con `HFDatasetLoader`, aplica `apply_all` y `apply_enrichment` de EA3 (sin reimplementarlos), calcula las tablas, las valida (34 comprobaciones, coherencia con `EA3/results/`), exporta la muestra, el reporte y los metadatos, y prueba la app con `AppTest`. |
| `EA4/app.py` | Punto de entrada de Streamlit: configura la página, carga los archivos con caché (se invalida al regenerarlos) y muestra instrucciones si falta alguno. |
| `EA4/src/visualization/dashboard_streamlit.py` | Clase `HospitalDashboard`: navegación por secciones, filtros, páginas P1-P5 / PC1-PC3 / Conclusiones (patrón `DashboardStreamlit` del código de referencia del curso). |
| `EA4/src/visualization/charts.py` · `components.py` | Figuras de Plotly (títulos, unidades, formato en español, universo de datos) y componentes de Streamlit (KPI, avisos, interpretaciones, tablas). |
| `EA4/src/metrics/business_metrics.py` | Métricas sobre registros y sobre el cubo agregado (solo pandas/numpy). |
| `EA4/src/metrics/insights.py` | Interpretaciones (métrica, patrón, respuesta, revisión operativa y limitación) derivadas de las cifras. |
| `EA4/src/metrics/table_builder.py` | Construye y exporta las tablas reutilizando `src/analysis` de EA3 (solo lo usa el notebook). |
| `EA4/src/data/data_loader.py` | Lee y valida las tablas, la muestra y los metadatos; error claro con el comando para regenerar. |

**Salidas:** tablas agregadas en `EA4/results/tables/` (8 de EA3 recalculadas con el mismo código + 9 de EA4, incluido el cubo de filtros), muestra en `EA4/data/processed/healthcare_enriched_sample.csv` y reporte en `EA4/reports/ea4-summary.txt`.

### Cómo se generan y usan los datos

* **Métricas oficiales:** se calculan en el notebook con los 500.000 registros y la app las lee de las tablas; no descarga Hugging Face al abrirse ni al filtrar.
* **Filtros** (departamento, severidad, grupo de personal, tipo de admisión y rango de estancia): se aplican a un **cubo agregado** del dataset completo (3.285 filas) que permite recalcular conteos, medias, medianas, cuartiles, desviaciones y días-paciente **exactos**. Las tablas que no tienen esas dimensiones (η², correlaciones, PC2, calidad) no se filtran y lo indican.
* **Muestra:** una de cada 5 filas en el orden original (sin azar), ≤ 100.000 filas (~22 MB). Se eligió frente a «primeras filas» porque conserva mejor la composición por severidad (desviación máxima 0,25 frente a 1,47 puntos porcentuales; comparación en el notebook). Solo alimenta vistas de registros individuales, rotuladas como muestra. La mediana del depósito no se recalcula con filtros porque requiere registros individuales.
* **Prioridad de fuentes de la app:** tablas de EA4 → tablas vigentes de EA3 (las ocho comunes) → muestra solo para vistas de registros.

### Ejecución

```bash
pip install -r requirements.txt                                                   # dependencias
jupyter nbconvert --to notebook --execute --inplace EA4/notebooks/04-data-streamlit.ipynb   # genera los datos
streamlit run EA4/app.py                                                          # inicia la app (http://localhost:8501)
python -m pytest -q                                                               # pruebas (EA1-EA4)
```

Si la terminal está en la carpeta `EA4/`, use `streamlit run app.py`. La app funciona con los archivos ya versionados; solo hay que ejecutar el notebook para regenerarlos. Probado localmente con Python 3.14 (Streamlit 1.67, Plotly 7.1); el workflow usa Python 3.12.

### Pruebas

`EA4/tests/test_dashboard_metrics.py` (28 pruebas) verifica con datos pequeños calculables a mano: días-paciente y su participación, estancia esperada por composición, η² (y su equivalencia con el R² de una regresión con indicadoras), Spearman con pares válidos y empates, cuantiles exactos del cubo, filtros, cruces, nulos, exportación y CSV, límite de 100.000 filas, archivos o columnas faltantes, respaldo en EA3, que la app no requiera `datasets` y la ausencia de tokens en el repositorio. Con los archivos reales generados, ejecuta además la app completa con `AppTest` (diez secciones y filtros) y comprueba la coherencia notebook-app. Es una comprobación técnica, no una revisión visual.

### GitHub Actions

El workflow ejecuta también el notebook 04 después del 03, verifica que existan las 17 tablas, los metadatos, la muestra (≤ 100.000 filas) y el reporte, que ningún archivo supere 50 MB, ejecuta `EA4/tests` con los archivos recién generados (incluida la app con `AppTest`), publica los resultados como artefacto y, solo en `main`, confirma los archivos de `EA4/` junto con los de EA1-EA3.

### Despliegue en Streamlit Community Cloud

La app no está publicada todavía. Para desplegarla: en <https://share.streamlit.io> elija **Create app → Deploy a public app from GitHub**, seleccione este repositorio, la rama `main` y **Main file path = `EA4/app.py`**. Streamlit Cloud usa `EA4/requirements.txt` (dependencias livianas: la app solo lee CSV) y el tema de `.streamlit/config.toml`. No se requieren secretos: la app no descarga datos.

### Limitaciones y reglas de uso

* Análisis descriptivo: ninguna diferencia, correlación o η² demuestra causalidad, eficiencia clínica ni efectos operativos.
* `admission_deposit` es el depósito registrado (unidad no documentada): no es ingreso, beneficio ni costo total.
* `staff_available` no está documentado y no es una medida validada de capacidad asistencial.
* `clinical_complexity_score` es exploratorio (Emergency sin peso: 72,84 % de cobertura); los umbrales de personal y de estancia son reglas del proyecto.
* No hay fechas ni camas por servicio: no se calculan ocupación ni giro de cama. Los resultados no se generalizan a otros hospitales.
* La ficha del dataset no documenta licencia ni condiciones de redistribución; confirme el uso permitido antes de publicar copias de registros (la muestra) o la app.

## Pruebas y GitHub Actions

```bash
python -m pytest -q      # tests/ (EA1-EA3) y EA4/tests
```

El workflow `.github/workflows/ml-pipeline.yml` se ejecuta en `push` a `main`, en `pull_request` y manualmente:

1. **tests:** instala dependencias y ejecuta `pytest` en `tests/` y `EA4/tests` (cargador, transformaciones, enriquecimiento, métricas, cubo, carga de resultados, límite de exportación y app). No descarga el dataset.
2. **notebooks:** ejecuta los cuatro notebooks (cada uno descarga el dataset una vez), verifica que existan los reportes, las tablas de `EA3/results/` y `EA4/results/tables/` y los CSV, comprueba la app de EA4 con sus pruebas, que cada CSV tenga ≤ 100.000 filas y que ningún archivo supere 50 MB; publica reportes y CSV como artefacto y, **solo en `main`**, hace commit de los CSV, `EA3/`, `EA4/`, los reportes y los notebooks ejecutados con `[skip ci]` (no se re-dispara). El secreto `HF_TOKEN` es opcional.

## Limitaciones

- Datos sin documentación de procedencia, licencia, periodo ni definiciones; no se generaliza a otros hospitales.
- Análisis descriptivo, sin causalidad. `Admission_Deposit` no es necesariamente el costo total.
- `patientid` no es único y tiene valores negativos; `health_conditions` tiene ~30 % de nulos sin significado documentado.

## Fuentes

- Hugging Face. *Nicolybgs/healthcare_data* [Conjunto de datos]. <https://huggingface.co/datasets/Nicolybgs/healthcare_data>
- Lhoest, Q. et al. (2021). *Datasets: A community library for natural language processing.* <https://doi.org/10.18653/v1/2021.emnlp-demo.21>
- McKinney, W. (2010). *Data structures for statistical computing in Python.* <https://doi.org/10.25080/Majora-92bf1922-00a>
- Baek, H. et al. (2018). *Analysis of length of hospital stay using electronic health records.* <https://doi.org/10.1371/journal.pone.0195901>
- Lepenioti, K. et al. (2020). *Prescriptive analytics: Literature review and research challenges.* <https://doi.org/10.1016/j.ijinfomgt.2019.04.003>
