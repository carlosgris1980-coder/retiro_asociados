"""Torneo de modelos: candidatos, métricas y validación cruzada (notebooks 04 y 05).

Módulo construido sobre el `train.py` del ejemplo del curso (CreditSight), con las tres piezas que pide la clase 3:

1. **Candidatos** (`candidates`): los modelos del README del equipo, con parámetros por defecto.
2. **Métricas** (`fold_metrics`): la de **negocio** (beneficio neto con la regla de valor esperado, y la misma con el
   umbral 0,5 para ver cuánto aporta la regla) y las **técnicas** (AUC, PR-AUC, exactitud balanceada, precisión y recall
   por clase, F1 de la clase 1 y macro, Brier, log loss).
3. **Validación cruzada** (`evaluate_folds`): los mismos pliegues para todos, una fila por pliegue.

Y las herramientas para leer el resultado contra el ruido: `summarize`, `improvement_over_noise`, `paired_comparison`.

Reglas que implementa:
  - Todos los modelos reciben **exactamente el mismo trato**: las mismas variables, el mismo preprocesamiento
    (`build_feature_pipeline`) y los mismos pliegues.
  - **Ningún modelo recibe tratamiento del desbalance** (ni `class_weight` ni remuestreo): ese ajuste no mejora la
    capacidad de ordenar, solo mueve el punto de operación. El umbral se decide después, para el modelo elegido.
  - Las métricas de clasificación se reportan en el **umbral por defecto (0,5)**, que no es el punto de operación final;
    las de ordenamiento (AUC, PR-AUC) no dependen del umbral.
  - **No se usa la exactitud simple**: con 79 % de la clase mayoritaria premia al modelo que no detecta a nadie. La
    exactitud **balanceada** (promedio del recall de las dos clases) sí se reporta.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from proyecto.config import CONTACT_COST, RANDOM_STATE, RETENTION_RATE, TARGET
from proyecto.features.engineering import build_feature_pipeline
from proyecto.models.business import campaign_value, contact_by_expected_value

VALUE_COL = "saldo_aportes"   # lo que se pierde si el asociado radica (notebook 01)
DEFAULT_THRESHOLD = 0.5       # umbral "de fábrica"; el punto de operación final se elige después

BUSINESS_METRICS = ["beneficio neto (por 1.000)", "beneficio neto, umbral 0,5 (por 1.000)",
                    "contactos (por 1.000)", "saldo cubierto"]
TECHNICAL_METRICS = ["AUC", "PR-AUC", "exactitud balanceada", "recall", "precisión", "F1", "recall clase 0",
                     "precisión clase 0", "F1 macro", "Brier", "log loss"]


# ---------------------------------------------------------------------------------------------
# 1. Candidatos
# ---------------------------------------------------------------------------------------------
def candidates(random_state: int = RANDOM_STATE) -> dict:
    """Modelos del torneo (los del README del equipo), con los parámetros por defecto de scikit-learn.

    No se optimiza nada aquí: los hiperparámetros se ajustan después, solo para los finalistas.
    """
    return {
        "Baseline (clase mayoritaria)": DummyClassifier(strategy="most_frequent"),
        "Regresión Logística": LogisticRegression(max_iter=2000, random_state=random_state),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "K-Vecinos": KNeighborsClassifier(n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def build_pipeline(model, columns: list[str], log_transform: bool = True, scale: bool = True) -> Pipeline:
    """Variables nuevas + preprocesamiento + modelo, en un solo Pipeline (se ajusta solo con entrenamiento)."""
    return Pipeline([
        ("prep", build_feature_pipeline(columns, log_transform=log_transform, scale=scale)),
        ("model", model),
    ])


# ---------------------------------------------------------------------------------------------
# 2. Métricas
# ---------------------------------------------------------------------------------------------
def fold_metrics(y_true, proba, value, contact_cost: float = CONTACT_COST,
                 retention_rate: float = RETENTION_RATE, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Métricas de un pliegue a partir de la probabilidad de radicar (`proba`) y del saldo en riesgo (`value`).

    - Negocio: beneficio neto por cada 1.000 asociados con la **regla de valor esperado** (la del notebook 01) y con
      la **clase más probable** (umbral 0,5), para medir cuánto aporta la regla de decisión.
    - Técnicas de ordenamiento (no dependen del umbral): AUC, PR-AUC.
    - Técnicas de clasificación (umbral 0,5): exactitud balanceada, recall, precisión y F1 de cada clase, F1 macro.
    - Calidad de las probabilidades: Brier y log loss (la regla de valor esperado usa la probabilidad como número).
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 0.0, 1.0)
    predicted = (proba >= threshold).astype(int)
    n = len(y_true)

    def money(contact):
        return campaign_value(y_true, contact, value, contact_cost, retention_rate)

    by_rule = money(contact_by_expected_value(proba, value, contact_cost, retention_rate))
    by_threshold = money(predicted)
    return {
        # negocio
        "beneficio neto (por 1.000)": by_rule["beneficio_neto"] / n * 1000,
        "beneficio neto, umbral 0,5 (por 1.000)": by_threshold["beneficio_neto"] / n * 1000,
        "contactos (por 1.000)": by_rule["contactados"] / n * 1000,
        "saldo cubierto": by_rule["saldo_cubierto"],
        # técnicas de ordenamiento
        "AUC": roc_auc_score(y_true, proba),
        "PR-AUC": average_precision_score(y_true, proba),
        # técnicas de clasificación, en el umbral por defecto
        "exactitud balanceada": balanced_accuracy_score(y_true, predicted),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "precisión": precision_score(y_true, predicted, zero_division=0),
        "F1": f1_score(y_true, predicted, zero_division=0),
        "recall clase 0": recall_score(y_true, predicted, pos_label=0, zero_division=0),
        "precisión clase 0": precision_score(y_true, predicted, pos_label=0, zero_division=0),
        "F1 macro": f1_score(y_true, predicted, average="macro", zero_division=0),
        # calidad de las probabilidades
        "Brier": brier_score_loss(y_true, proba),
        "log loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }


# ---------------------------------------------------------------------------------------------
# 3. Validación cruzada
# ---------------------------------------------------------------------------------------------
def evaluate_folds(pipeline, data: pd.DataFrame, folds: list, train_mask=None, target: str = TARGET,
                   value_col: str = VALUE_COL, return_oof: bool = False):
    """Validación cruzada con pliegues dados. Una fila por pliegue con todas las métricas y los tiempos.

    - `folds`: lista de (índices de ajuste, índices de validación), los MISMOS para todo lo que se compara.
    - `train_mask`: opcional; booleano por fila. Si se da, el modelo se entrena solo con esas filas del pliegue,
      pero se **evalúa siempre en todo el pliegue de validación** (para comparar poblaciones de entrenamiento).
    - `return_oof`: si es True, devuelve además la probabilidad fuera de pliegue de cada fila.
    """
    y = data[target].to_numpy()
    value = data[value_col].to_numpy(dtype=float)
    data = data.drop(columns=[target])   # el objetivo nunca llega al Pipeline
    mask = np.ones(len(data), dtype=bool) if train_mask is None else np.asarray(train_mask, dtype=bool)
    oof = np.full(len(data), np.nan)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(folds):
        fit_idx = train_idx[mask[train_idx]]
        model = clone(pipeline)
        start = time.perf_counter()
        model.fit(data.iloc[fit_idx], y[fit_idx])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        proba = model.predict_proba(data.iloc[valid_idx])[:, 1]
        predict_seconds = time.perf_counter() - start
        oof[valid_idx] = proba
        rows.append({"pliegue": fold, **fold_metrics(y[valid_idx], proba, value[valid_idx]),
                     "segundos de ajuste": fit_seconds, "segundos de predicción": predict_seconds})
    table = pd.DataFrame(rows).set_index("pliegue")
    return (table, oof) if return_oof else table


def run_tournament(models: dict, columns: list[str], data: pd.DataFrame, folds: list, verbose: bool = True):
    """Corre todos los candidatos con el mismo pipeline y los mismos pliegues.

    Devuelve (resultados por pliegue de cada modelo, probabilidades fuera de pliegue de cada modelo).
    """
    by_fold, oof = {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        by_fold[name], oof[name] = evaluate_folds(build_pipeline(model, columns), data, folds, return_oof=True)
        if verbose:
            print(f"{name:30s} listo en {time.perf_counter() - start:5.0f} s")
    return by_fold, oof


# ---------------------------------------------------------------------------------------------
# Lectura contra el ruido
# ---------------------------------------------------------------------------------------------
def summarize(by_fold: dict, metrics: list[str]) -> pd.DataFrame:
    """Tabla con una fila por modelo: media y desviación entre pliegues (`± métrica`) de cada métrica pedida."""
    rows = {}
    for name, table in by_fold.items():
        row = {}
        for metric in metrics:
            row[metric] = table[metric].mean()
            row[f"± {metric}"] = table[metric].std()
        rows[name] = row
    return pd.DataFrame(rows).T


def improvement_over_noise(reference, variant, higher_is_better: bool = True) -> dict:
    """Mejora de una variante frente a la referencia, dividida por el ruido entre pliegues (guía del curso, 1.3).

    Ruido = desviación entre pliegues de la variante. Mejora / ruido menor que 1 está dentro del ruido.
    """
    reference, variant = np.asarray(reference, dtype=float), np.asarray(variant, dtype=float)
    sign = 1 if higher_is_better else -1
    improvement = sign * (variant.mean() - reference.mean())
    noise = variant.std(ddof=1)
    return {"mejora": improvement, "ruido": noise, "mejora / ruido": improvement / noise if noise > 0 else np.inf}


def paired_comparison(best, other, higher_is_better: bool = True) -> dict:
    """Comparación pliegue a pliegue (guía del curso, 2.4). Positivo: `best` gana.

    Ruido = desviación entre pliegues de la diferencia pareada.
    """
    best, other = np.asarray(best, dtype=float), np.asarray(other, dtype=float)
    diff = (best - other) if higher_is_better else (other - best)
    noise = diff.std(ddof=1)
    return {
        "diferencia media": diff.mean(),
        "pliegues que gana": f"{int((diff > 0).sum())} de {len(diff)}",
        "diferencia / ruido": diff.mean() / noise if noise > 0 else np.inf,
    }
