"""Limpieza del dataset de asociados de Confiar (`data/raw/bd_confiar_2.xlsx`).

Cada función implementa una decisión del EDA (notebook 02, `reports/01_eda.md`):
  1. `load_raw`: elimina el Nit (dato personal) al cargar y lo reemplaza por un identificador sustituto.
  2. `add_time_context` + `drop_time_columns`: guarda el año de proceso y la marca "salió sin radicar"
     como columnas auxiliares y elimina las dos fechas, que juntas delatan la respuesta.
  3. `mask_implausible`: la edad fuera de 14–100 años pasa a vacío (no se borra la fila).
  4. `flag_min_wage_income`: marca los ingresos que son exactamente un salario mínimo de 2017–2025.
  5. `add_product_flags`: convierte los ceros de saldo en indicadores de "no tiene el producto".

Lo que NO se hace aquí (va dentro del Pipeline de la etapa 4, ajustado solo con entrenamiento):
imputar la edad vacía, transformar colas largas o escalar.
"""
import numpy as np
import pandas as pd

from proyecto.config import (
    ID_COL,
    MIN_WAGE_BY_YEAR,
    PII_COLUMNS,
    PLAUSIBLE_RANGE,
    PROCESS_COL,
    PRODUCT_FLAGS,
    RAW_FILE,
    TARGET,
    TIME_COLUMNS,
)


def load_raw(path=RAW_FILE) -> pd.DataFrame:
    """Lee el Excel crudo, elimina el Nit y agrega `associate_id` (0..n-1). No modifica el archivo."""
    df = pd.read_excel(path)
    return drop_personal_data(df)


def drop_personal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza el documento del asociado por un identificador sustituto sin significado."""
    out = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
    out.insert(0, ID_COL, np.arange(len(out)))
    return out


def add_time_context(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas auxiliares de tiempo (no son variables del modelo).

    - `process_year`: año del registro (mes de radicación o último mes observado).
    - `exit_without_request`: 1 si no radicó y su registro termina antes del último mes del panel,
      es decir, dejó de estar en la base sin radicar (sale de la fecha y del objetivo: solo diagnóstico).
    - `stratum`: grupo para partir de forma estratificada: radica / activo / salió sin radicar.
    """
    out = df.copy()
    period = pd.PeriodIndex(out[PROCESS_COL].astype(str), freq="M")
    last_period = period.max()
    out["process_year"] = period.year.astype(int)
    out["exit_without_request"] = ((out[TARGET] == 0) & (period < last_period)).astype(int)
    out["stratum"] = np.select(
        [out[TARGET] == 1, out["exit_without_request"] == 1], ["radica", "salio_sin_radicar"], "activo"
    )
    return out


def drop_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina `anio_mes_proceso` y `cohorte`: juntas reconstruyen el mes de proceso, que delata la respuesta."""
    return df.drop(columns=[c for c in TIME_COLUMNS if c in df.columns])


def mask_implausible(df: pd.DataFrame, ranges: dict = PLAUSIBLE_RANGE) -> pd.DataFrame:
    """Marca como vacío (NaN) todo valor fuera de su rango posible. No borra filas."""
    out = df.copy()
    for col, (low, high) in ranges.items():
        out[col] = out[col].astype(float).where(out[col].between(low, high))
    return out


def flag_min_wage_income(df: pd.DataFrame, wages: dict = MIN_WAGE_BY_YEAR) -> pd.DataFrame:
    """Agrega `income_is_min_wage` = 1 si el ingreso es exactamente el salario mínimo de algún año 2017–2025."""
    out = df.copy()
    values = np.array(list(wages.values()))
    income = out["ingresos"].to_numpy(dtype=float)[:, None]
    out["income_is_min_wage"] = np.isclose(income, values[None, :], rtol=0, atol=1e-6).any(axis=1).astype(int)
    return out


def add_product_flags(df: pd.DataFrame, flags: dict = PRODUCT_FLAGS) -> pd.DataFrame:
    """Agrega indicadores 0/1 de tenencia de producto: un promedio de saldo en cero = no tiene el producto."""
    out = df.copy()
    for name, source in flags.items():
        out[name] = (out[source] > 0).astype(int)
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica en orden las decisiones del EDA. Devuelve el mismo número de filas que recibe."""
    n_rows = len(df)
    out = add_time_context(df)
    out = drop_time_columns(out)
    out = mask_implausible(out)
    out = flag_min_wage_income(out)
    out = add_product_flags(out)
    assert len(out) == n_rows, "la limpieza no debe perder filas"
    return out
