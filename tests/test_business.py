"""Pruebas de las métricas de negocio con cuentas que se pueden hacer a mano."""
import numpy as np
import pytest

from proyecto.models.business import (
    baseline_strategies,
    breakeven_probability,
    campaign_value,
    contact_by_expected_value,
    cost_matrix,
)

# 4 asociados: dos radican (saldos 2 y 1), dos no (saldos 3 y 0)
Y = np.array([1, 1, 0, 0])
VALUE = np.array([2.0, 1.0, 3.0, 0.0])
COST, RATE = 0.1, 0.5


def test_campaign_value_hand_computed():
    out = campaign_value(Y, contact=[1, 0, 1, 0], value=VALUE, contact_cost=COST, retention_rate=RATE)
    assert (out["VP"], out["FP"], out["FN"], out["contactados"]) == (1, 1, 1, 2)
    assert out["valor_conservado"] == pytest.approx(0.5 * 2.0)
    assert out["costo_campaña"] == pytest.approx(0.2)
    assert out["beneficio_neto"] == pytest.approx(0.8)
    assert out["saldo_en_riesgo"] == pytest.approx(3.0)
    assert out["saldo_cubierto"] == pytest.approx(2.0 / 3.0)


def test_contact_rule_uses_expected_value():
    # 0.5 × 0.5 × 2 = 0.5 > 0.1 sí; 0.1 × 0.5 × 1 = 0.05 < 0.1 no; saldo 0 nunca
    contact = contact_by_expected_value([0.5, 0.1, 0.9, 0.9], VALUE, COST, RATE)
    assert contact.tolist() == [1, 0, 1, 0]


def test_breakeven_probability():
    p = breakeven_probability(VALUE, COST, RATE)
    assert p[0] == pytest.approx(0.1)
    assert np.isinf(p[3])


def test_cost_matrix_is_zero_on_the_diagonal():
    m = cost_matrix(mean_value=2.0, contact_cost=COST, retention_rate=RATE)
    assert m.iloc[0, 0] == 0 and m.iloc[1, 1] == 0
    assert m.iloc[0, 1] == pytest.approx(COST)
    assert m.iloc[1, 0] == pytest.approx(RATE * 2.0 - COST)


def test_baselines_no_campaign_is_zero_and_perfect_is_best():
    table = baseline_strategies(Y, VALUE, COST, RATE, share=0.5)
    assert table.loc["Sin campaña (protocolo reactivo actual)", "beneficio_neto"] == 0
    assert table["beneficio_neto"].idxmax() == "Perfecta (techo teórico)"
