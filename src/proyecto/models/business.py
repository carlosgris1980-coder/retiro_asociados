"""Métricas de negocio: traducen las decisiones del modelo a dinero (notebook 01, `reports/00_caso.md`).

La decisión que apoya el modelo: a qué asociados incluir en la campaña proactiva de fidelización.
  - Contactar a quien iba a radicar (VP): con probabilidad `retention_rate` desiste y la cooperativa
    conserva sus aportes; el contacto cuesta `contact_cost`.
  - Contactar a quien no iba a radicar (FP): se paga el contacto y no se recupera nada.
  - No contactar a quien iba a radicar (FN): se pierde la oportunidad de conservar sus aportes.
  - No contactar a quien no iba a radicar (VN): no cuesta nada.

Valores en la unidad del dataset (millones de COP). `contact_cost` y `retention_rate` son supuestos.
"""
import numpy as np
import pandas as pd


def campaign_value(y_true, contact, value, contact_cost: float, retention_rate: float) -> dict:
    """Resultado económico de contactar a quienes tienen `contact == 1`.

    Devuelve contactados, VP, FP, FN, valor conservado, costo, beneficio neto, el saldo en riesgo
    y la parte de ese saldo que la campaña alcanzó a cubrir (VP / todos los que radican, ponderado por saldo).
    """
    y_true = np.asarray(y_true).astype(int)
    contact = np.asarray(contact).astype(int)
    value = np.asarray(value, dtype=float)
    tp = (contact == 1) & (y_true == 1)
    fp = (contact == 1) & (y_true == 0)
    fn = (contact == 0) & (y_true == 1)
    at_risk = value[y_true == 1].sum()
    saved = retention_rate * value[tp].sum()
    cost = contact_cost * contact.sum()
    return {
        "contactados": int(contact.sum()),
        "VP": int(tp.sum()),
        "FP": int(fp.sum()),
        "FN": int(fn.sum()),
        "valor_conservado": float(saved),
        "costo_campaña": float(cost),
        "beneficio_neto": float(saved - cost),
        "saldo_en_riesgo": float(at_risk),
        "saldo_cubierto": float(value[tp].sum() / at_risk) if at_risk > 0 else 0.0,
    }


def contact_by_expected_value(proba, value, contact_cost: float, retention_rate: float) -> np.ndarray:
    """Regla de decisión: contactar si p × efectividad × saldo > costo del contacto (valor esperado positivo)."""
    proba = np.asarray(proba, dtype=float)
    value = np.asarray(value, dtype=float)
    return (proba * retention_rate * value > contact_cost).astype(int)


def breakeven_probability(value, contact_cost: float, retention_rate: float) -> np.ndarray:
    """Probabilidad mínima de radicar a partir de la cual contactar a un asociado deja ganancia.

    p* = costo / (efectividad × saldo). Con saldo 0 nunca conviene (devuelve infinito).
    """
    value = np.asarray(value, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(value > 0, contact_cost / (retention_rate * value), np.inf)


def cost_matrix(mean_value: float, contact_cost: float, retention_rate: float) -> pd.DataFrame:
    """Matriz de costos binaria (filas = real, columnas = decisión) frente a una decisión perfecta.

    - FN (iba a radicar, no se contactó): se deja de ganar efectividad × saldo − costo.
    - FP (no iba a radicar, se contactó): se pierde el costo del contacto.
    """
    fn_cost = retention_rate * mean_value - contact_cost
    return pd.DataFrame(
        [[0.0, contact_cost], [fn_cost, 0.0]],
        index=["real: no radica", "real: radica"],
        columns=["no contactar", "contactar"],
    )


def baseline_strategies(y_true, value, contact_cost: float, retention_rate: float, share: float = 0.2,
                        seed: int = 42) -> pd.DataFrame:
    """Estrategias sin modelo (y la perfecta como techo), con su resultado económico.

    - Sin campaña (hoy: protocolo reactivo).
    - Contactar a todos.
    - Contactar al azar a `share` de los asociados.
    - Perfecta: contactar solo a quienes van a radicar y tienen saldo (techo teórico, no alcanzable).
    """
    y_true = np.asarray(y_true).astype(int)
    value = np.asarray(value, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    random_contact = np.zeros(n, dtype=int)
    random_contact[rng.choice(n, size=round(share * n), replace=False)] = 1
    strategies = {
        "Sin campaña (protocolo reactivo actual)": np.zeros(n, dtype=int),
        "Contactar a todos": np.ones(n, dtype=int),
        f"Contactar al azar al {round(share * 100)} %": random_contact,
        "Perfecta (techo teórico)": ((y_true == 1) & (retention_rate * value > contact_cost)).astype(int),
    }
    rows = {name: campaign_value(y_true, c, value, contact_cost, retention_rate) for name, c in strategies.items()}
    return pd.DataFrame(rows).T
