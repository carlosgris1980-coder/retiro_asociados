"""Pruebas del módulo de modelos (src/proyecto/models/train.py) y de los pliegues de validación."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from proyecto.config import FEATURES, TARGET
from proyecto.data.split import make_folds
from proyecto.models.train import (
    BUSINESS_METRICS,
    TECHNICAL_METRICS,
    build_pipeline,
    candidates,
    evaluate_folds,
    fold_metrics,
    improvement_over_noise,
    paired_comparison,
    run_tournament,
    summarize,
)


def _data(n: int = 400) -> pd.DataFrame:
    """Datos sintéticos con todas las columnas del modelo; el objetivo depende de una variable."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.random(n) for c in FEATURES})
    df["antig_meses"] = rng.integers(0, 100, n)
    df["saldo_aportes"] = rng.random(n) * 3
    df[TARGET] = (df["porc_meses_publiturno"] + 0.3 * rng.random(n) > 0.8).astype(int)
    df["associate_id"] = np.arange(n)
    df["stratum"] = np.where(df[TARGET] == 1, "radica", "activo")
    return df


def test_candidates_are_the_six_models_of_the_readme():
    names = list(candidates())
    assert len(names) == 6 and names[0].startswith("Baseline")
    for model in candidates().values():
        assert getattr(model, "class_weight", None) is None   # ningún modelo trata el desbalance


def test_perfect_probabilities_give_auc_and_balanced_accuracy_of_one():
    y = np.array([0, 0, 1, 1])
    metrics = fold_metrics(y, proba=np.array([0.1, 0.2, 0.9, 0.8]), value=np.ones(4))
    assert metrics["AUC"] == pytest.approx(1.0) and metrics["PR-AUC"] == pytest.approx(1.0)
    assert metrics["exactitud balanceada"] == pytest.approx(1.0)
    assert metrics["recall"] == 1.0 and metrics["recall clase 0"] == 1.0 and metrics["F1 macro"] == pytest.approx(1.0)


def test_metrics_include_business_and_technical_families():
    metrics = fold_metrics(np.array([0, 1, 0, 1]), proba=np.array([0.3, 0.6, 0.2, 0.4]), value=np.ones(4))
    assert set(BUSINESS_METRICS) <= set(metrics) and set(TECHNICAL_METRICS) <= set(metrics)
    assert "exactitud" not in metrics          # la exactitud simple se excluye a propósito


def test_expected_value_rule_and_threshold_rule_can_differ():
    # saldo alto y probabilidad 0,3: la regla de valor esperado contacta (0,3 × 0,2 × 10 > 0,03); el umbral 0,5 no
    metrics = fold_metrics(np.array([1, 0]), proba=np.array([0.3, 0.1]), value=np.array([10.0, 0.0]))
    assert metrics["beneficio neto (por 1.000)"] > metrics["beneficio neto, umbral 0,5 (por 1.000)"]


def test_constant_model_detects_nobody():
    y = np.array([0, 0, 1, 1])
    metrics = fold_metrics(y, proba=np.zeros(4), value=np.ones(4))
    assert metrics["AUC"] == pytest.approx(0.5)
    assert metrics["recall"] == 0 and metrics["beneficio neto (por 1.000)"] == 0


def test_folds_are_reused_and_never_share_an_entity():
    df = _data()
    folds = make_folds(df, n_splits=5, seed=1)
    assert len(folds) == 5
    for train_idx, valid_idx in folds:
        assert set(df.iloc[train_idx]["associate_id"]).isdisjoint(df.iloc[valid_idx]["associate_id"])
    assert sorted(np.concatenate([v for _, v in folds]).tolist()) == list(range(len(df)))


def test_evaluate_folds_returns_one_row_per_fold_and_out_of_fold_probabilities():
    df = _data()
    folds = make_folds(df, n_splits=4, seed=1)
    pipeline = build_pipeline(LogisticRegression(max_iter=500), FEATURES)
    table, oof = evaluate_folds(pipeline, df, folds, return_oof=True)
    assert len(table) == 4 and not np.isnan(oof).any()
    assert table["AUC"].mean() > 0.8          # la variable que define el objetivo se aprende


def test_train_mask_trains_on_fewer_rows_but_evaluates_on_the_whole_fold():
    df = _data()
    folds = make_folds(df, n_splits=4, seed=1)
    pipeline = build_pipeline(DummyClassifier(strategy="prior"), FEATURES)
    mask = df["associate_id"] % 2 == 0
    _, oof = evaluate_folds(pipeline, df, folds, train_mask=mask, return_oof=True)
    assert not np.isnan(oof).any()            # se predijo para todas las filas, también las excluidas del ajuste


def test_tournament_uses_same_folds_and_summary_has_mean_and_noise():
    df = _data()
    folds = make_folds(df, n_splits=4, seed=1)
    models = {"base": DummyClassifier(strategy="most_frequent"), "logística": LogisticRegression(max_iter=500)}
    by_fold, oof = run_tournament(models, FEATURES, df, folds, verbose=False)
    assert set(by_fold) == set(models) and all(len(t) == 4 for t in by_fold.values())
    table = summarize(by_fold, ["AUC", "PR-AUC"])
    assert list(table.columns) == ["AUC", "± AUC", "PR-AUC", "± PR-AUC"]
    assert table.loc["base", "AUC"] == pytest.approx(0.5)
    assert table.loc["logística", "AUC"] > table.loc["base", "AUC"]


def test_improvement_and_paired_comparison_signs():
    ref, better = [0.70, 0.71, 0.69], [0.75, 0.76, 0.74]
    assert improvement_over_noise(ref, better)["mejora"] == pytest.approx(0.05)
    assert improvement_over_noise(ref, better, higher_is_better=False)["mejora"] == pytest.approx(-0.05)
    result = paired_comparison(better, ref)
    assert result["pliegues que gana"] == "3 de 3" and result["diferencia media"] == pytest.approx(0.05)
