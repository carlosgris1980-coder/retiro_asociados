"""Inferencia: validar entradas y predecir clase y probabilidades.

Contrato (el mismo que el del ejemplo del curso):
  - Entrada: una tabla con las columnas CRUDAS que una persona conoce al decidir (las del esquema).
  - Salida: la clase decidida, la probabilidad de cada clase y una señal de "revisar" cuando el modelo duda.
  - Nunca devuelve una predicción silenciosa con datos imposibles: lanza `InputValidationError`.

Uso:

    from proyecto.inference.predict import load_model, predict_with_scores
    from proyecto.inference.schema import load_schema
    model, schema = load_model("models/model.joblib"), load_schema("models/input_schema.json")
    resultado = predict_with_scores(filas_nuevas, model, schema)

Funciona con cualquier clasificador de scikit-learn (o un Pipeline) que tenga `predict_proba`.

Adaptaciones para el caso de retiro de asociados (Confiar):
  - `prepare_inputs` calcula los indicadores de tenencia de producto (`has_loan`, `has_prog_savings`, `has_account`) a
    partir de los saldos promedio, igual que la limpieza del notebook 03: quien usa la app no tiene que digitarlos.
  - Si el esquema trae `decision_rule` de tipo `valor_esperado`, la decisión es la del negocio (notebook 01):
    **contactar si probabilidad de radicar × efectividad × saldo de aportes > costo del contacto**. Sin saldo no se puede
    calcular el valor esperado: la fila queda marcada para revisión.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from proyecto.config import PRODUCT_FLAGS


class InputValidationError(ValueError):
    """Los datos de entrada no son válidos; el mensaje dice qué corregir."""


def load_model(path: str | Path):
    """Carga el modelo exportado (un Pipeline completo, con el preprocesamiento adentro)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta antes el notebook 09_inferencia para exportarlo.")
    return joblib.load(path)


def validate_inputs(df: pd.DataFrame, schema: dict) -> list[str]:
    """Valida columnas, tipos y rangos. Lanza InputValidationError si hay errores; devuelve advertencias."""
    numeric, categorical = schema["numeric"], schema["categorical"]
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise InputValidationError(f"Faltan columnas: {', '.join(missing)}")

    errors, warnings = [], []
    for col in numeric:
        values = pd.to_numeric(df[col], errors="coerce")
        if (values.isna() & df[col].notna()).any():
            errors.append(f"{col}: hay valores que no son números")
        if col in schema.get("valid_range", {}):
            low, high = schema["valid_range"][col]
            if (values.notna() & ~values.between(low, high)).any():
                errors.append(f"{col}: debe estar entre {low:g} y {high:g}")
    if errors:
        raise InputValidationError("; ".join(errors))

    for col, known in schema.get("categories", {}).items():
        unseen = set(df[col].dropna().astype(str)) - set(known)
        if unseen:
            warnings.append(f"{col}: categoría no vista en el entrenamiento ({', '.join(sorted(unseen))})")
    for col, (low, high) in schema.get("observed_range", {}).items():
        values = pd.to_numeric(df[col], errors="coerce")
        if ((values < low) | (values > high)).any():
            warnings.append(f"{col}: fuera del rango visto en el entrenamiento ({low:g} a {high:g})")
    return warnings


def prepare_inputs(df: pd.DataFrame, flags: dict = PRODUCT_FLAGS) -> pd.DataFrame:
    """Agrega las columnas derivadas que el modelo espera y que una persona no tiene por qué digitar.

    Los indicadores de tenencia (1 si el saldo promedio es mayor que cero) se calculan solo si falta la columna y existe
    su saldo de origen. Si el saldo es desconocido (vacío), el indicador queda vacío y el pipeline lo imputa.
    """
    out = df.copy()
    for name, source in flags.items():
        if name not in out.columns and source in out.columns:
            values = pd.to_numeric(out[source], errors="coerce")
            out[name] = np.where(values.isna(), np.nan, (values > 0).astype(float))
    return out


def apply_expected_value_rule(probability: np.ndarray, value, rule: dict) -> pd.DataFrame:
    """Regla de valor esperado: contactar si probabilidad × efectividad × saldo > costo del contacto."""
    value = pd.to_numeric(pd.Series(value), errors="coerce").to_numpy(dtype=float)
    expected = probability * rule["retention_rate"] * value
    decision = np.where(np.isnan(expected), rule.get("label_unknown", "revisar: falta el saldo"),
                        np.where(expected > rule["contact_cost"], rule.get("label_yes", "contactar"),
                                 rule.get("label_no", "no contactar")))
    return pd.DataFrame({"decision": decision, "expected_value": expected})


def decide_min_cost(proba: np.ndarray, cost_matrix) -> np.ndarray:
    """Elige la clase de menor costo esperado: argmin_j sum_i p_i * costo(i, j)."""
    return np.argmin(proba @ np.asarray(cost_matrix, dtype=float), axis=1)


def _ordered_proba(model, X: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    """Probabilidades en el orden de class_names (reordena si el modelo guarda sus clases con otro orden)."""
    proba = model.predict_proba(X)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):  # Pipeline
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is not None:
        as_text = [str(c) for c in classes]
        if set(as_text) == set(class_names) and as_text != class_names:
            proba = proba[:, [as_text.index(name) for name in class_names]]
    return proba


def predict_with_scores(df: pd.DataFrame, model, schema: dict) -> pd.DataFrame:
    """Predice. Una fila de salida por fila de entrada.

    Columnas: decision, most_likely, proba_<clase>, needs_review y (si el esquema trae matriz de costos)
    expected_cost. Las advertencias de validación quedan en `resultado.attrs["warnings"]`.
    """
    warnings = validate_inputs(df, schema)
    class_names = schema["class_names"]
    proba = _ordered_proba(model, prepare_inputs(df), class_names)
    most_likely = proba.argmax(axis=1)
    out = pd.DataFrame(proba, columns=[f"proba_{c}" for c in class_names], index=df.index)

    rule = schema.get("decision_rule")
    if rule and rule.get("type") == "valor_esperado":
        # Decisión del negocio: depende de la probabilidad de la clase positiva y del saldo de cada asociado
        decided = apply_expected_value_rule(proba[:, 1], df[rule["value_column"]].to_numpy(), rule)
        out.insert(0, "decision", decided["decision"].to_numpy())
        out["most_likely"] = [class_names[i] for i in most_likely]
        out["expected_value"] = decided["expected_value"].to_numpy()
        out["contact_cost"] = rule["contact_cost"]
        out["needs_review"] = (proba.max(axis=1) < schema.get("review_threshold", 0.55)) | np.isnan(out["expected_value"])
        out.attrs["warnings"] = warnings
        return out

    cost = schema.get("cost_matrix")
    decision = decide_min_cost(proba, cost) if cost is not None else most_likely
    out.insert(0, "decision", [class_names[i] for i in decision])
    out["most_likely"] = [class_names[i] for i in most_likely]
    if cost is not None:
        out["expected_cost"] = (proba @ np.asarray(cost, dtype=float))[np.arange(len(df)), decision]
    out["needs_review"] = (proba.max(axis=1) < schema.get("review_threshold", 0.55)) | (decision != most_likely)
    out.attrs["warnings"] = warnings
    return out
