"""Formato numérico en español (punto de miles, coma decimal), compartido por métricas y gráficos."""
from __future__ import annotations

import math


def num(x: float | int | None, decimals: int = 0) -> str:
    """12.345,6 · devuelve «n/d» si el valor no está disponible."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/d"
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x: float | None, decimals: int = 1) -> str:
    """Recibe un porcentaje ya multiplicado por 100 (p. ej. 47.9 → «47,9 %»)."""
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{num(x, decimals)} %"


def days(x: float | None, decimals: int = 1) -> str:
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{num(x, decimals)} días"
