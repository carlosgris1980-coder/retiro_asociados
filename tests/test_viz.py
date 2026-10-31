"""Pruebas de las ayudas de análisis de src/proyecto/viz.py."""
import numpy as np
import pandas as pd

from proyecto.viz import fmt_num, fmt_pct, rate_by_quantile


def test_ties_stay_in_the_same_band_even_if_rows_are_sorted_by_the_target():
    # 600 ceros (300 radican, ordenados primero) y 400 valores distintos: los ceros deben formar UN tramo con tasa 0,5
    x = np.r_[np.zeros(600), np.arange(1, 401)]
    y = np.r_[np.ones(300), np.zeros(300), np.zeros(400)]
    out = rate_by_quantile(pd.Series(x), pd.Series(y), q=10)
    zero_band = out[out["x_median"] == 0]
    assert len(zero_band) == 1
    assert zero_band["n"].iloc[0] == 600 and zero_band["rate"].iloc[0] == 0.5
    assert out["n"].sum() == len(x)


def test_spanish_number_format():
    assert fmt_num(194073) == "194.073"
    assert fmt_num(1.18, 2) == "1,18"
    assert fmt_pct(0.2093) == "20,9 %"
