"""Pruebas de las variables nuevas y del preprocesamiento (src/proyecto/features/engineering.py)."""
import numpy as np
import pandas as pd
import pytest

from proyecto.config import FEATURES
from proyecto.features.engineering import (
    ENGINEERED_FEATURES,
    SUSPECT_COLUMNS,
    SUSPECT_ENGINEERED,
    add_features,
    build_feature_pipeline,
    build_preprocessor,
    feature_columns,
    signed_log1p,
)


def _frame() -> pd.DataFrame:
    """Cuatro asociados hechos a mano con todas las columnas del modelo."""
    base = {c: [0.0, 0.0, 0.0, 0.0] for c in FEATURES}
    base.update({
        "antig_meses": [6, 7, 24, 25],
        "saldo_aportes": [2.0, 0.0, 1.0, 4.0],
        "prom_trx": [3.0, 5.0, 0.0, 1.0],
        "ingresos": [2.0, 1.0, 0.0, np.nan],
        "saldo_cta": [1.0, 0.5, 0.0, 2.0],
        "prom_saldo_cta": [2.0, 0.0, 0.0, 1.0],
        "saldo_ahorro_prog": [1.0, 0.0, 3.0, 0.0],
        "prom_saldo_ahorro_prog": [4.0, 0.0, 3.0, 0.0],
        "prom_saldo_cartera": [5.0, 0.0, 1.0, 0.0],
        "edad": [30.0, np.nan, 50.0, 40.0],
    })
    return pd.DataFrame(base)


def test_add_features_creates_every_engineered_column():
    out = add_features(_frame())
    assert set(ENGINEERED_FEATURES) <= set(out.columns)


def test_add_features_does_not_modify_the_input():
    df = _frame()
    before = df.copy()
    add_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_risk_window_boundaries_are_inclusive():
    out = add_features(_frame())
    assert out["in_risk_window"].tolist() == [0, 1, 1, 0]   # 6 no, 7 sí, 24 sí, 25 no


def test_dormant_means_no_contributions_or_no_transactions():
    out = add_features(_frame())
    assert out["is_dormant"].tolist() == [0, 1, 1, 0]


def test_n_products_counts_loan_savings_and_account():
    out = add_features(_frame())
    assert out["n_products"].tolist() == [3, 0, 2, 1]


def test_ratios_handle_division_by_zero_and_missing_values():
    out = add_features(_frame())
    # (ahorro + cuenta) / aportes: 2/2 = 1; aportes 0 -> vacío
    assert out["savings_to_contributions"].iloc[0] == pytest.approx(1.0)
    assert np.isnan(out["savings_to_contributions"].iloc[1])
    # aportes / ingresos: ingreso 0 o vacío -> vacío (nunca infinito)
    assert out["contributions_to_income"].iloc[0] == pytest.approx(1.0)
    assert out["contributions_to_income"].iloc[2:].isna().all()
    assert not np.isinf(out[ENGINEERED_FEATURES].to_numpy(dtype=float)).any()
    # tendencia de la cuenta: saldo actual / promedio
    assert out["account_balance_trend"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(out["account_balance_trend"].iloc[1])


def test_signed_log_keeps_the_sign_and_zero():
    result = signed_log1p([-np.e + 1, 0.0, np.e - 1])
    assert result.tolist() == pytest.approx([-1.0, 0.0, 1.0])


def test_feature_columns_can_drop_suspects_and_engineered():
    assert feature_columns(use_engineered=False) == list(FEATURES)
    without = feature_columns(drop_suspects=True)
    assert not (set(SUSPECT_COLUMNS) | set(SUSPECT_ENGINEERED)) & set(without)
    assert "in_risk_window" in without
    assert "edad" not in feature_columns(drop=["edad"])


def test_preprocessor_fills_missing_and_returns_finite_numbers():
    df = add_features(_frame())
    columns = feature_columns()
    result = build_preprocessor(columns).fit_transform(df)
    assert result.shape[0] == len(df)
    assert np.isfinite(result).all()


def test_feature_pipeline_runs_end_to_end_on_raw_columns():
    pipeline = build_feature_pipeline(feature_columns(), log_transform=False, scale=False)
    result = pipeline.fit_transform(_frame())
    assert result.shape[0] == 4
    assert np.isfinite(result).all()
