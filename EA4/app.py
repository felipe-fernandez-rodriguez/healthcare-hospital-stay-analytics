"""App de Streamlit de EA4: métricas de estancia hospitalaria y depósito de admisión.

Ejecutar desde la raíz del repositorio:
    streamlit run EA4/app.py

Desde la carpeta EA4:
    streamlit run app.py

En Streamlit Community Cloud: repositorio del proyecto, rama main, «Main file path» = EA4/app.py.

La app solo lee los archivos que genera el notebook EA4/notebooks/04-data-streamlit.ipynb
(tablas del dataset completo, cubo para filtros y muestra); no descarga Hugging Face.
"""
import sys
from pathlib import Path

import streamlit as st

EA4_DIR = Path(__file__).resolve().parent
ROOT = EA4_DIR.parent
# La raíz va primero en sys.path: así «EA4» se importa como paquete y no se confunde
# el subpaquete EA4/src con el paquete src de la raíz (EA1-EA3).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from EA4.src.data import data_loader as dl  # noqa: E402
from EA4.src.visualization.components import missing_data_panel  # noqa: E402
from EA4.src.visualization.dashboard_streamlit import HospitalDashboard  # noqa: E402

st.set_page_config(page_title="Estancia hospitalaria · EA4", page_icon="🏥", layout="wide")


@st.cache_resource(show_spinner="Cargando resultados del dataset completo…")
def cargar(firma: tuple) -> HospitalDashboard:
    """La firma (fechas de modificación de los archivos) invalida la caché al regenerarlos."""
    return HospitalDashboard.from_files()


try:
    dashboard = cargar(dl.files_signature())
except dl.DataUnavailableError as exc:
    missing_data_panel(exc, dl.file_status())
    st.stop()

dashboard.render()
