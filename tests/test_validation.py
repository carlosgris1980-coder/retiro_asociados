"""Pruebas de src/proyecto/models/validation.py."""
import numpy as np
import pandas as pd
import pytest

from proyecto.models.validation import bootstrap_metric, reliability_table, segment_table, shap_importance


def test_bootstrap_metric_contains_the_full_value_and_is_narrow_for_a_constant():
    proba = np.full(200, 0.3)
    value, low, high = bootstrap_metric(lambda rows, p: p[rows].mean(), proba, np.arange(200), n_boot=50)
    assert value == pytest.approx(0.3) and low == pytest.approx(0.3) and high == pytest.approx(0.3)


def test_bootstrap_metric_resamples_whole_entities():
    # dos entidades con 3 filas cada una: cualquier remuestra tiene media 0, 0,5 o 1 (nunca 1/3 o 2/3)
    values = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    rng_values = [bootstrap_metric(lambda rows, p: p[rows].mean(), values, groups, n_boot=30, seed=s)[1] for s in range(5)]
    assert all(v in (0.0, 0.5, 1.0) for v in rng_values)


def test_reliability_table_of_a_perfectly_calibrated_model():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    table = reliability_table(y, p, n_bins=5)
    assert np.allclose(table["probabilidad media predicha"], table["tasa real"], atol=0.03)


def test_segment_table_hand_computed():
    frame = pd.DataFrame({"zona": ["a", "a", "a", "b"], "real": [1, 1, 0, 0], "contactar": [1, 0, 1, 0],
                          "proba": [0.8, 0.4, 0.6, 0.1]})
    table = segment_table(frame, "zona")
    a = table.loc[("zona", "a")]
    assert a["asociados"] == 3 and a["tasa real"] == pytest.approx(2 / 3)
    assert a["recall"] == pytest.approx(0.5) and a["precisión"] == pytest.approx(0.5)
    assert np.isnan(table.loc[("zona", "b"), "recall"])      # sin quienes radican, el recall no existe


def test_shap_importance_sorts_and_reads_direction():
    data = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
    values = np.array([[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0]])
    table = shap_importance(values, data, ["sube", "constante"])
    assert table.index[0] == "sube" and table.loc["sube", "dirección"] == pytest.approx(1.0)
    assert table.loc["constante", "dirección"] == 0.0
