"""Optimización de hiperparámetros con Optuna, registrando cada ensayo en MLflow (notebook 06).

Construido sobre el `tuning.py` del ejemplo del curso (CreditSight), con dos ajustes al caso de Confiar:

1. **Se optimiza la identificación de la clase minoritaria** (PR-AUC por defecto; ver `OBJECTIVES`) con la misma
   validación por asociado del torneo.
2. **Se controla el sobreajuste**: cada ensayo mide la brecha entre el PR-AUC en la parte de ajuste y en la de
   validación. Un ensayo con brecha mayor que `max_gap` es **no admisible**: el muestreador aprende a evitarlo
   (restricciones de Optuna) y nunca se elige como mejor.

Cada ensayo queda como ejecución anidada en MLflow con **todas** las métricas (negocio, técnicas y brecha), aunque
solo una se optimice. Con `storage` el estudio se guarda en disco y **se reanuda** sin repetir ensayos (ni duplicar
el tracking), la lección que deja el ejemplo del curso.
"""
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from proyecto.config import RANDOM_STATE
from proyecto.models.train import build_pipeline, evaluate_folds
from proyecto.tracking.mlflow_utils import log_run

N_TREES = 200   # el número de árboles se fija (más árboles no sobreajustan, solo cuestan); se afina su FORMA


def rf_space(trial: optuna.Trial) -> dict:
    """Espacio de búsqueda del bosque aleatorio: los parámetros que controlan qué tanto memoriza cada árbol.

    - `max_depth`: profundidad máxima. Por defecto no tiene límite (el árbol memoriza).
    - `min_samples_leaf`: asociados mínimos por hoja (escala logarítmica). Hojas grandes = reglas más generales.
    - `max_features`: fracción de variables candidatas en cada corte. Menos = árboles más distintos entre sí.
    - `max_samples`: fracción de asociados con que se entrena cada árbol. Menos = más diversidad, menos memoria.
    """
    return {
        "max_depth": trial.suggest_int("max_depth", 6, 24),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100, log=True),
        "max_features": trial.suggest_float("max_features", 0.15, 0.8),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
    }


def make_rf(params: dict, random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """Bosque con `N_TREES` árboles y los parámetros del ensayo. Sin `class_weight` (ver notebook 06)."""
    return RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=random_state, **params)


MODELS = {"Random Forest": (rf_space, make_rf)}

# Métricas que se pueden optimizar (nombres de `fold_metrics`) y en qué dirección.
OBJECTIVES = {
    "PR-AUC": "maximize",                       # identificación de la clase minoritaria (no depende del umbral)
    "AUC": "maximize",                          # capacidad de ordenar en general
    "beneficio neto (por 1.000)": "maximize",   # negocio: regla de valor esperado (depende de la calibración)
    "Brier": "minimize",                        # calidad de las probabilidades
    "log loss": "minimize",
    "recall": "maximize",                       # cuidado: solo, empuja a marcar a todos; vigílalo con la precisión
}
GAP = "brecha PR-AUC"


def tune(name: str, data: pd.DataFrame, columns: list[str], folds: list, n_trials: int, metric: str = "PR-AUC",
         max_gap: float | None = 0.10, seed: int = RANDOM_STATE, log: bool = True, storage: str | None = None,
         study_name: str | None = None) -> optuna.Study:
    """Estudio de Optuna que optimiza `metric` con la restricción `brecha PR-AUC <= max_gap`.

    Cada ensayo: validación cruzada con `folds` (los mismos para todos), métricas de validación y de ajuste.
    Devuelve el estudio; cada ensayo guarda sus métricas medias en `trial.user_attrs`.
    """
    if metric not in OBJECTIVES:
        raise ValueError(f"Métrica no soportada: {metric}. Opciones: {sorted(OBJECTIVES)}")
    space_fn, make_fn = MODELS[name]

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        table = evaluate_folds(build_pipeline(make_fn(params), columns), data, folds, train_metrics=True)
        board = {col: float(table[col].mean()) for col in table.columns}
        board.update({f"{col} ±": float(table[col].std()) for col in table.columns})
        for key, value in board.items():
            trial.set_user_attr(key, value)
        if max_gap is not None:
            trial.set_constraint("brecha", board[GAP] - max_gap)   # <= 0: admisible; el muestreador lo aprende
        if log:
            log_run(f"{name.lower().replace(' ', '-')}-ensayo-{trial.number:02d}", params,
                    {_mlflow_name(k): v for k, v in board.items()},
                    tags={"etapa": "06_optimizacion", "modelo": name, "metrica_optimizada": metric,
                          "admisible": str(max_gap is None or board[GAP] <= max_gap)},
                    nested=True)
        return board[metric]

    sampler = optuna.samplers.TPESampler(seed=seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # sin mensajes informativos de Optuna
    study = optuna.create_study(
        direction=OBJECTIVES[metric], sampler=sampler, load_if_exists=True,
        study_name=study_name or f"{name}-{metric}",
        storage=(f"sqlite:///{Path(storage).as_posix()}" if storage else None),
    )
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(n_trials - done, 0)
    if done:
        print(f"  {name}: se reanuda el estudio ({done} ensayos ya hechos, faltan {remaining})")
    study.set_user_attr("metric", metric)
    study.set_user_attr("max_gap", max_gap)
    if log and remaining:
        with mlflow.start_run(run_name=f"optuna-{name.lower().replace(' ', '-')}") as parent:
            study.optimize(objective, n_trials=remaining)
            best = select_best(study, max_gap)
            mlflow.log_metric(f"mejor_{_mlflow_name(metric)}", best.value)
            mlflow.log_params({f"mejor_{k}": v for k, v in best.params.items()})
            mlflow.set_tags({"etapa": "06_optimizacion", "modelo": name, "metrica_optimizada": metric})
            study.set_user_attr("mlflow_parent_run_id", parent.info.run_id)
    elif remaining:
        study.optimize(objective, n_trials=remaining)
    return study


def select_best(study: optuna.Study, max_gap: float | None = None) -> optuna.trial.FrozenTrial:
    """El mejor ensayo **admisible** (brecha <= max_gap). Si ninguno lo es, falla con un mensaje claro."""
    finished = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    admissible = [t for t in finished if max_gap is None or t.user_attrs.get(GAP, np.inf) <= max_gap]
    if not admissible:
        raise ValueError(f"Ningún ensayo cumple la brecha <= {max_gap}: amplía el presupuesto o revisa el espacio.")
    maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE
    return max(admissible, key=lambda t: t.value) if maximize else min(admissible, key=lambda t: t.value)


def trials_table(study: optuna.Study, max_gap: float | None = None) -> pd.DataFrame:
    """Una fila por ensayo: parámetros, métricas medias de validación y de ajuste, y si es admisible."""
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"ensayo": t.number, **t.params, **{k: v for k, v in t.user_attrs.items() if not k.endswith(" ±")}}
        row["admisible"] = max_gap is None or t.user_attrs.get(GAP, np.inf) <= max_gap
        rows.append(row)
    return pd.DataFrame(rows).set_index("ensayo")


def bootstrap_difference(metric_fn, proba_a, proba_b, groups, n_boot: int = 500, seed: int = RANDOM_STATE):
    """Diferencia de una métrica (A menos B) con intervalo del 95 %, remuestreando ENTIDADES completas.

    `metric_fn(rows, proba)` devuelve la métrica calculada sobre las filas `rows` (índices) con las probabilidades
    `proba`; así puede usar cualquier dato adicional (objetivo, saldo). En este dataset cada asociado es una fila,
    pero se remuestrea por entidad para que la función siga siendo correcta si el dataset se vuelve un panel.
    Devuelve (diferencia con todos los datos, límite inferior, límite superior).
    """
    groups = np.asarray(groups)
    rows_by_entity = pd.Series(np.arange(len(groups))).groupby(groups).apply(lambda s: s.to_numpy()).to_numpy()
    all_rows = np.arange(len(groups))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(rows_by_entity), len(rows_by_entity))
        rows = np.concatenate([rows_by_entity[i] for i in picked])
        diffs.append(metric_fn(rows, proba_a) - metric_fn(rows, proba_b))
    full = metric_fn(all_rows, proba_a) - metric_fn(all_rows, proba_b)
    return float(full), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _mlflow_name(name: str) -> str:
    """MLflow solo acepta letras, números y algunos signos en los nombres de métricas."""
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n", "±": "std", "(": "", ")": "",
                    ",": "", ".": "", " ": "_"}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.strip("_")
