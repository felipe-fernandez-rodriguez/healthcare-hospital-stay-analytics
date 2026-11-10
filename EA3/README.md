# EA3 — Transformación y enriquecimiento de datos

Entregables de la actividad EA3 del proyecto *Análisis descriptivo de la estancia hospitalaria y factores operativos en admisiones médicas*. Todo lo que genera esta actividad está en esta carpeta; el código reutilizable vive en `src/` (compartido con EA1 y EA2) y el README general está en [`../README.md`](../README.md).

**Objetivo.** Transformar y enriquecer el dataset completo de forma reproducible, responder las preguntas de negocio de EA2 con tablas calculadas sobre **todos** los registros y generar las evidencias (CSV de muestra, tablas y reporte TXT). Conserva todo lo de EA1 y EA2.

**Notebook:** [`notebooks/03-data-transformation-enrichment.ipynb`](notebooks/03-data-transformation-enrichment.ipynb) (usa `HFDatasetLoader`, `transformations.apply_all` de EA1/EA2 y las funciones nuevas de `src/data/enrichment.py`). 

> **Importante:** los cálculos y las tablas corresponden al **dataset completo** (500.000 filas × 15 columnas originales). Solo el CSV `EA3/data/healthcare_enriched_sample.csv` se limita a **100.000 filas** (las primeras, en su orden original, sin azar). El resumen TXT y las tablas **no** se calculan con ese CSV.

## Estructura de la carpeta

```text
EA3/
├── README.md                                   # este archivo
├── notebooks/
│   └── 03-data-transformation-enrichment.ipynb # flujo completo, ejecutado
├── data/
│   └── healthcare_enriched_sample.csv          # muestra del DataFrame enriquecido (≤ 100.000 filas)
├── results/                                    # tablas agregadas (DataFrame completo)
│   ├── results_by_department.csv
│   ├── results_by_severity.csv
│   ├── results_by_staff_availability.csv
│   ├── results_by_department_severity.csv
│   ├── results_by_stay_range.csv
│   ├── results_by_complexity_score.csv
│   ├── results_by_department_indicators.csv
│   └── results_data_quality.csv
└── reports/
    └── data-enrichment-summary.txt             # resumen del DataFrame enriquecido COMPLETO
```

Código reutilizado desde `src/`: `data/huggingface_loader.py`, `data/transformations.py`, `data/enrichment.py` (nuevo), `data/dataset_io.py`, `analysis/descriptive.py`, `analysis/quality.py` (nuevo), `analysis/results.py` (nuevo) y `utils/reporter.py`. Pruebas: `tests/test_enrichment.py`. Las rutas de salida se definen en `src/config.py` (`EA3_DIR`).

## Preguntas y tablas

| Pregunta (EA2) | Tabla (`EA3/results/`) |
|---|---|
| 1. Estancia y depósito por departamento | `results_by_department.csv` |
| 2. Diferencias por severidad | `results_by_severity.csv` |
| 3. Distribución según disponibilidad de personal (umbrales de EA2) | `results_by_staff_availability.csv` |
| 4. Cruce departamento × severidad | `results_by_department_severity.csv` |
| 5. Utilidad de los indicadores derivados | `results_by_stay_range.csv`, `results_by_complexity_score.csv`, `results_by_department_indicators.csv` |
| Calidad de los datos | `results_data_quality.csv` |

Cada tabla incluye registros, valores válidos (tamaño de muestra de cada estadístico), media, mediana, desviación, mínimo y máximo de estancia (días) y depósito (unidad monetaria no documentada).

## Transformaciones y variables derivadas

Se reutilizan las 6 transformaciones de EA1/EA2 (espacios, tipos numéricos, duplicados exactos, `snake_case`, categorías ordenadas, `staff_availability_group` Bajo 0-3 / Medio 4-6 / Alto 7+). Los nulos no se imputan y `patientid` no se usa como variable cuantitativa. Variables nuevas (`src/data/enrichment.py`); si un indicador no se puede calcular queda **nulo**, nunca 0:

| Variable | Fórmula / regla |
|---|---|
| `deposit_per_stay_day` | `admission_deposit / stay_in_days` (cociente descriptivo; no es costo diario) |
| `visitors_per_day` | `visitors_with_patient / stay_in_days` (no mide carga asistencial) |
| `stay_range` | Corta (1-3 días), Moderada (4-7), Prolongada (>7); umbrales del proyecto |
| `clinical_complexity_score` | `severity_weight × admission_type_weight` (ver abajo) |
| `dept_avg_stay`, `stay_vs_dept_avg` | media de estancia por departamento; diferencia respecto a ella |
| `dept_severity_avg_deposit` | depósito medio por departamento × severidad |

**Puntaje exploratorio de complejidad** (regla del proyecto, **no** una escala clínica validada): severidad Minor 1, Moderate 2, Extreme 3; tipo de admisión Elective 1,0, Urgent 1,25, Trauma 1,5. Las categorías reales de `type_of_admission` son `Trauma`, `Emergency` y `Urgent`: `Elective` no existe en los datos y `Emergency` no tiene peso definido, por lo que esos registros quedan **sin puntaje** (no se inventan equivalencias). El notebook y el reporte indican cuántos registros recibieron puntaje.

## Archivos generados

- `EA3/data/healthcare_enriched_sample.csv` (≤ 100.000 filas, ~22 MB).
- `EA3/results/*.csv` (8 tablas).
- `EA3/reports/data-enrichment-summary.txt`: reporte generado con `DatasetSummarizer` desde el DataFrame final completo.
- `docs/proyecto_analisis_estancia_hospitalaria.docx`: documento actualizado con la etapa de transformación y enriquecimiento.

## Ejecución y validaciones

Desde la raíz del repositorio, con el entorno instalado (`pip install -r requirements.txt`):

```bash
jupyter nbconvert --to notebook --execute --inplace EA3/notebooks/03-data-transformation-enrichment.ipynb
python -m pytest -q tests          # 34 pruebas (sin Internet)
```

El notebook comprueba el número de filas, que el original no se altere, la coherencia de conteos y denominadores y que los CSV coincidan con las tablas.

## Limitaciones

Análisis descriptivo, sin causalidad ni juicios de eficiencia; indicadores exploratorios; umbrales y pesos definidos por el proyecto; ficha del dataset sin licencia, procedencia ni definiciones. La redistribución de copias derivadas del dataset no está verificada como autorizada (la ficha no documenta licencia): confirme las condiciones de uso antes de publicar los CSV.
