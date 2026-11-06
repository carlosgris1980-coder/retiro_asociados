"""Validación final e interpretación (notebook 07): intervalos, calibración, errores por segmento y SHAP.

Reglas que apoya este módulo (guía `guia-validacion-registro-y-app.md`):
  - Los intervalos remuestrean **entidades** (asociados), no filas.
  - La calibración se decide con asociados **reservados** del entrenamiento, nunca con la prueba.
  - Un buen promedio puede esconder un segmento al que el modelo le va mal: se mide por segmento.
  - SHAP explica **al modelo**, no al mundo.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from proyecto.config import RANDOM_STATE


def _rows_by_entity(groups) -> np.ndarray:
    groups = np.asarray(groups)
    return pd.Series(np.arange(len(groups))).groupby(groups).apply(lambda s: s.to_numpy()).to_numpy()


def bootstrap_metric(metric_fn, proba, groups, n_boot: int = 500, seed: int = RANDOM_STATE):
    """Una métrica con su intervalo del 95 %, remuestreando ENTIDADES completas.

    `metric_fn(rows, proba)` calcula la métrica sobre las filas `rows`. Devuelve (valor, límite inferior, superior).
    """
    rows_by_entity = _rows_by_entity(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(rows_by_entity), len(rows_by_entity))
        values.append(metric_fn(np.concatenate([rows_by_entity[i] for i in picked]), proba))
    full = metric_fn(np.arange(len(np.asarray(groups))), proba)
    return float(full), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def reliability_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Curva de calibración por cuantiles: probabilidad media predicha frente a la tasa real en cada tramo."""
    observed, predicted = calibration_curve(np.asarray(y_true), np.asarray(proba), n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"probabilidad media predicha": predicted, "tasa real": observed})


def segment_table(frame: pd.DataFrame, segment: str, target: str = "real", contact: str = "contactar",
                  proba: str = "proba") -> pd.DataFrame:
    """Por cada valor de `segment`: asociados, tasa real de radicación, % contactado, recall y precisión.

    La señal de alarma de la guía: un segmento donde el % contactado y la tasa real se separan mucho, o donde el recall
    cae frente al resto.
    """
    rows = []
    for value, part in frame.groupby(segment, observed=True, dropna=False):
        filers = part[target] == 1
        contacted = part[contact] == 1
        rows.append({
            "segmento": segment, "valor": "sin dato" if pd.isna(value) else str(value), "asociados": len(part),
            "tasa real": filers.mean(), "% contactado": contacted.mean(),
            "probabilidad media": part[proba].mean(),
            "recall": (contacted & filers).sum() / filers.sum() if filers.sum() else np.nan,
            "precisión": (contacted & filers).sum() / contacted.sum() if contacted.sum() else np.nan,
        })
    return pd.DataFrame(rows).set_index(["segmento", "valor"])


def shap_importance(values: np.ndarray, data: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Importancia SHAP por variable: influencia media (|SHAP|) y dirección.

    La dirección es la correlación entre el valor de la variable y su aporte: positiva significa que valores altos
    empujan la probabilidad de radicar hacia arriba.
    """
    values, data = np.asarray(values, dtype=float), np.asarray(data, dtype=float)
    direction = []
    for j in range(values.shape[1]):
        x, s = data[:, j], values[:, j]
        ok = np.std(x) > 0 and np.std(s) > 0
        direction.append(float(np.corrcoef(x, s)[0, 1]) if ok else 0.0)
    out = pd.DataFrame({"influencia media |SHAP|": np.abs(values).mean(axis=0), "dirección": direction}, index=names)
    return out.sort_values("influencia media |SHAP|", ascending=False)
