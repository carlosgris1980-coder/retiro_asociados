"""Variables nuevas y preprocesamiento (etapa 4).

Cada variable nueva es una **hipótesis** que sale del EDA (notebook 02) y que se mide en el notebook 04 con
y sin ella, con los mismos pliegues y la métrica de negocio. Todas se calculan **fila por fila** con columnas
que se conocen al momento de decidir a quién contactar: no aprenden nada de los datos, así que pueden ir
dentro del Pipeline sin riesgo de fuga entre particiones.

| Variable | Cómo se calcula | Por qué podría ayudar (EDA) | ¿Se conoce al decidir? |
|---|---|---|---|
| `in_risk_window` | 1 si la antigüedad está entre 7 y 24 meses | La radicación supera el 30 % en esa ventana (sección 5) | Sí: antigüedad de hoy |
| `is_dormant` | 1 si no tiene aportes o no transacciona | El asociado "dormido" casi no radica (sección 5) | Sí |
| `n_products` | crédito + ahorro programado + cuenta (0 a 3) | Profundidad del vínculo: "el crédito ata" | Sí |
| `savings_to_contributions` | (ahorro programado + cuenta) / aportes | Cuánto de su dinero está fuera de los aportes | Sí |
| `contributions_to_income` | aportes / ingresos | Peso de los aportes frente al ingreso; al ser un cociente, atenúa el efecto de los pesos nominales | Sí (el ingreso es sospechoso; ver abajo) |
| `account_balance_trend` | saldo actual de la cuenta / saldo promedio | ¿Está vaciando la cuenta? Primera señal de tendencia en un dataset sin dimensión temporal | Sí, pero hereda la sospecha: el saldo actual se mide al radicar |
| `prog_savings_trend` | saldo actual de ahorro programado / promedio | Igual, para ahorro programado y CDT | Igual que la anterior |

Las variables construidas con columnas sospechosas de fuga (`config.LEAKAGE_SUSPECTS`, `ingresos` y los saldos
"al momento de efectuar solicitud") **heredan la sospecha**: `feature_columns(drop_suspects=True)` las quita junto
con sus columnas de origen, para medir el modelo con y sin ellas.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from proyecto.config import FEATURES, LEAKAGE_SUSPECTS

# Ventana de mayor riesgo según el EDA (tasa de radicación de 30,5 % y 30,3 % entre los 7 y 24 meses)
RISK_WINDOW_MONTHS = (7, 24)

ENGINEERED_FEATURES = [
    "in_risk_window",
    "is_dormant",
    "n_products",
    "savings_to_contributions",
    "contributions_to_income",
    "account_balance_trend",
    "prog_savings_trend",
]

# Variables nuevas que dependen de columnas sospechosas y heredan la sospecha
SUSPECT_ENGINEERED = ["contributions_to_income", "account_balance_trend", "prog_savings_trend"]
SUSPECT_COLUMNS = LEAKAGE_SUSPECTS + ["ingresos"]

# Colas largas (EDA, decisión 10): se transforman con logaritmo con signo dentro del Pipeline
HEAVY_TAILED = [
    "saldo_aportes", "prom_saldo_capitalizacion_aportes", "saldo_ahorro_prog", "prom_saldo_ahorro_prog",
    "saldo_cta", "prom_saldo_cta", "prom_saldo_cartera", "prom_asesorias_publiturno", "prom_trx", "ingresos",
    "savings_to_contributions", "contributions_to_income", "account_balance_trend", "prog_savings_trend",
]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """División segura: si el denominador es cero, negativo o vacío, el resultado queda vacío (NaN)."""
    return num / den.where(den > 0)


def signed_log1p(values):
    """Logaritmo que acepta negativos: signo(x) × log(1 + |x|). Acorta colas largas sin perder el signo."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables nuevas de `ENGINEERED_FEATURES`. No modifica el DataFrame que recibe."""
    out = df.copy()
    low, high = RISK_WINDOW_MONTHS
    out["in_risk_window"] = out["antig_meses"].between(low, high).astype(int)
    out["is_dormant"] = ((out["saldo_aportes"] <= 0) | (out["prom_trx"] <= 0)).astype(int)
    out["n_products"] = (
        (out["prom_saldo_cartera"] > 0).astype(int)
        + (out["prom_saldo_ahorro_prog"] > 0).astype(int)
        + (out["prom_saldo_cta"] > 0).astype(int)
    )
    out["savings_to_contributions"] = _ratio(out["saldo_ahorro_prog"] + out["saldo_cta"], out["saldo_aportes"])
    out["contributions_to_income"] = _ratio(out["saldo_aportes"], out["ingresos"])
    out["account_balance_trend"] = _ratio(out["saldo_cta"], out["prom_saldo_cta"])
    out["prog_savings_trend"] = _ratio(out["saldo_ahorro_prog"], out["prom_saldo_ahorro_prog"])
    return out


def feature_columns(use_engineered: bool = True, drop_suspects: bool = False, drop: list[str] | None = None) -> list[str]:
    """Lista de columnas que recibe el modelo.

    - `use_engineered`: agrega las variables nuevas.
    - `drop_suspects`: quita las sospechas de fuga y las variables nuevas que dependen de ellas.
    - `drop`: columnas adicionales a quitar (por ejemplo, una de cada pareja redundante).
    """
    columns = list(FEATURES) + (list(ENGINEERED_FEATURES) if use_engineered else [])
    excluded = set(drop or [])
    if drop_suspects:
        excluded |= set(SUSPECT_COLUMNS) | set(SUSPECT_ENGINEERED)
    return [c for c in columns if c not in excluded]


def build_preprocessor(columns: list[str], log_transform: bool = True, scale: bool = True) -> ColumnTransformer:
    """Imputación con la mediana (+ indicador de vacío), logaritmo en colas largas y escalado.

    Todo se ajusta **solo con el conjunto de entrenamiento** cuando el Pipeline hace `fit`.
    Para modelos de árboles conviene `log_transform=False, scale=False` (no lo necesitan).
    """
    heavy = [c for c in columns if c in HEAVY_TAILED] if log_transform else []
    rest = [c for c in columns if c not in heavy]

    def numeric_steps(with_log: bool) -> Pipeline:
        steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if with_log:
            steps.append(("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    transformers = []
    if heavy:
        transformers.append(("heavy", numeric_steps(with_log=True), heavy))
    if rest:
        transformers.append(("num", numeric_steps(with_log=False), rest))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_feature_pipeline(columns: list[str] | None = None, log_transform: bool = True, scale: bool = True) -> Pipeline:
    """Variables nuevas + preprocesamiento, en un solo paso que se pone antes del modelo.

    Uso: `Pipeline([("prep", build_feature_pipeline(cols)), ("model", LogisticRegression())])`.
    """
    columns = columns or feature_columns()
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("preprocess", build_preprocessor(columns, log_transform=log_transform, scale=scale)),
    ])
