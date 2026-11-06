"""Pruebas de la inferencia adaptada al caso (src/proyecto/inference/predict.py) y de la app con el modelo real.

Las que usan `models/model.joblib` y `models/input_schema.json` se saltan si todavía no se ha ejecutado el notebook 09.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proyecto.inference.predict import (
    InputValidationError,
    apply_expected_value_rule,
    load_model,
    predict_with_scores,
    prepare_inputs,
)
from proyecto.inference.schema import load_schema

ROOT = Path(__file__).resolve().parents[1]
MODEL, SCHEMA = ROOT / "models" / "model.joblib", ROOT / "models" / "input_schema.json"
needs_model = pytest.mark.skipif(not (MODEL.exists() and SCHEMA.exists()), reason="falta ejecutar el notebook 09")


def test_prepare_inputs_derives_product_flags_and_keeps_unknowns_empty():
    df = pd.DataFrame({"prom_saldo_cartera": [3.0, 0.0, np.nan], "prom_saldo_ahorro_prog": [0.0, 1.0, 2.0],
                       "prom_saldo_cta": [0.5, 0.0, 0.0]})
    out = prepare_inputs(df)
    assert out["has_loan"].iloc[:2].tolist() == [1.0, 0.0] and np.isnan(out["has_loan"].iloc[2])
    assert out["has_prog_savings"].tolist() == [0.0, 1.0, 1.0]
    assert "has_loan" not in df.columns                      # no modifica la entrada


def test_prepare_inputs_does_not_overwrite_existing_flags():
    df = pd.DataFrame({"prom_saldo_cartera": [3.0], "has_loan": [0]})
    assert prepare_inputs(df)["has_loan"].iloc[0] == 0


def test_expected_value_rule_hand_computed():
    rule = {"contact_cost": 0.03, "retention_rate": 0.2}
    out = apply_expected_value_rule(np.array([0.5, 0.1, 0.9]), [1.0, 1.0, np.nan], rule)
    # 0,5 × 0,2 × 1 = 0,10 > 0,03 → contactar; 0,1 × 0,2 × 1 = 0,02 → no; sin saldo → revisar
    assert out["decision"].tolist() == ["contactar", "no contactar", "revisar: falta el saldo"]
    assert out["expected_value"].iloc[0] == pytest.approx(0.10)


@pytest.fixture(scope="module")
def real():
    return load_model(MODEL), load_schema(SCHEMA)


def _example(schema, name=None) -> pd.DataFrame:
    examples = schema["examples"]
    return pd.DataFrame([examples[name or list(examples)[-1]]])


@needs_model
def test_real_model_answers_a_normal_row_with_the_business_rule(real):
    model, schema = real
    result = predict_with_scores(_example(schema), model, schema)
    assert result["decision"].iloc[0] in {"contactar", "no contactar"}
    assert np.isclose(result[["proba_no radica", "proba_radica"]].sum(axis=1).iloc[0], 1.0)
    p, value = result["proba_radica"].iloc[0], _example(schema)["saldo_aportes"].iloc[0]
    rule = schema["decision_rule"]
    assert result["expected_value"].iloc[0] == pytest.approx(p * rule["retention_rate"] * value)


@needs_model
def test_real_model_accepts_unknown_values(real):
    model, schema = real
    row = _example(schema).assign(ingresos=np.nan, prom_saldo_cartera=np.nan)
    assert len(predict_with_scores(row, model, schema)) == 1


@needs_model
@pytest.mark.parametrize("column, value", [("edad", 200), ("porc_meses_publiturno", 1.5), ("saldo_aportes", -1)])
def test_real_model_refuses_impossible_values(real, column, value):
    model, schema = real
    with pytest.raises(InputValidationError, match=column):
        predict_with_scores(_example(schema).assign(**{column: value}), model, schema)


@needs_model
def test_app_runs_end_to_end_with_the_real_model():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not app.exception
    profile = list(load_schema(SCHEMA)["examples"])[-1]
    app.selectbox[0].set_value(profile).run()
    app.button[0].click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Decisión sugerida" in labels
