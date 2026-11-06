"""Pruebas de src/proyecto/models/tuning.py y de la brecha de sobreajuste (sin MLflow: log=False)."""
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from proyecto.config import FEATURES, TARGET
from proyecto.data.split import make_folds
from proyecto.models.train import build_pipeline, evaluate_folds
from proyecto.models.tuning import (
    GAP,
    _mlflow_name,
    bootstrap_difference,
    make_rf,
    rf_space,
    select_best,
    trials_table,
    tune,
)


def _data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) for c in FEATURES})
    df["antig_meses"] = rng.integers(0, 100, n)
    df["saldo_aportes"] = rng.random(n) * 3
    df[TARGET] = (df["porc_meses_publiturno"] + 0.5 * rng.random(n) > 0.9).astype(int)
    df["associate_id"] = np.arange(n)
    df["stratum"] = np.where(df[TARGET] == 1, "radica", "activo")
    return df


def test_space_stays_inside_its_ranges_and_builds_a_forest_without_class_weight():
    params = rf_space(optuna.trial.FixedTrial({"max_depth": 10, "min_samples_leaf": 5, "max_features": 0.3,
                                               "max_samples": 0.8}))
    model = make_rf(params)
    assert model.n_estimators == 200 and model.max_depth == 10 and model.class_weight is None


def test_train_metrics_detect_an_overfitting_forest():
    df = _data()
    folds = make_folds(df, n_splits=3, seed=1)
    table = evaluate_folds(build_pipeline(RandomForestClassifier(random_state=0), FEATURES), df, folds,
                           train_metrics=True)
    assert (table["PR-AUC (ajuste)"] > 0.99).all()       # sin límites, el bosque memoriza el entrenamiento
    assert (table[GAP] > 0).all()


def test_tune_respects_the_gap_constraint_when_selecting():
    df = _data()
    folds = make_folds(df, n_splits=3, seed=1)
    study = tune("Random Forest", df, FEATURES, folds, n_trials=3, log=False, max_gap=1.0)
    table = trials_table(study, max_gap=1.0)
    assert len(table) == 3 and table["admisible"].all()
    best = select_best(study, max_gap=1.0)
    assert best.value == pytest.approx(table["PR-AUC"].max())
    with pytest.raises(ValueError):
        select_best(study, max_gap=-1.0)                  # ningún ensayo puede tener brecha negativa menor que -1


def test_bootstrap_difference_is_zero_for_identical_models_and_positive_for_a_better_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good, noise = y * 0.8 + rng.random(400) * 0.2, rng.random(400)
    from sklearn.metrics import average_precision_score

    def ap(rows, p):
        return average_precision_score(y[rows], p[rows])

    diff, low, high = bootstrap_difference(ap, good, good, np.arange(400), n_boot=50)
    assert diff == 0 and low == 0 and high == 0
    diff, low, high = bootstrap_difference(ap, good, noise, np.arange(400), n_boot=50)
    assert diff > 0 and low > 0


def test_mlflow_names_have_no_forbidden_characters():
    assert _mlflow_name("beneficio neto (por 1.000)") == "beneficio_neto_por_1000"
    assert _mlflow_name("brecha PR-AUC ±") == "brecha_PR-AUC_std"
