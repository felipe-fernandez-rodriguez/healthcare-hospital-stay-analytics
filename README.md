# Análisis descriptivo de la estancia hospitalaria y factores operativos en admisiones médicas

Proyecto académico (EA1 – carga y creación de dataset; EA2 – ideación, diseño y documentación de la necesidad). Carga el dataset [`Nicolybgs/healthcare_data`](https://huggingface.co/datasets/Nicolybgs/healthcare_data) desde Hugging Face, lo transforma, genera reportes `.txt` y realiza una exploración descriptiva.

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
├── EA3/                                # entregables de EA3 (ver EA3/README.md)
│   ├── notebooks/03-data-transformation-enrichment.ipynb
│   ├── data/healthcare_enriched_sample.csv   # muestra ≤ 100.000 filas
│   ├── results/                        # 8 tablas agregadas (dataset completo)
│   └── reports/data-enrichment-summary.txt
├── docs/
│   └── proyecto_analisis_estancia_hospitalaria.docx       # documento del proyecto
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
├── tests/                              # pruebas con pytest (test_pipeline.py: EA1/EA2; test_enrichment.py: EA3)
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
```

o ábralos con `jupyter lab`. Alternativa por script para EA1: `python -m src.pipeline`.

**Token (opcional):** el dataset es público y el proyecto funciona sin él. Para evitar límites de uso anónimo copie `.env.example` a `.env` y complete `HF_TOKEN`, o defina la variable de entorno. `.env` está en `.gitignore`; nunca se imprime ni se sube.

**Documento Word:** `docs/proyecto_analisis_estancia_hospitalaria.docx`.

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

## Pruebas y GitHub Actions

```bash
python -m pytest -q tests
```

El workflow `.github/workflows/ml-pipeline.yml` se ejecuta en `push` a `main`, en `pull_request` y manualmente:

1. **tests:** instala dependencias y ejecuta `pytest` (cargador, transformaciones, resumen, límite de exportación). No descarga el dataset.
2. **notebooks:** ejecuta los tres notebooks (cada uno descarga el dataset una vez), verifica que existan los reportes, las tablas de `EA3/results/` y los CSV, que cada CSV tenga ≤ 100.000 filas y que ningún archivo supere 50 MB; publica reportes y CSV como artefacto y, **solo en `main`**, hace commit de los CSV, `EA3/`, los reportes y los notebooks ejecutados con `[skip ci]` (no se re-dispara). El secreto `HF_TOKEN` es opcional.

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
