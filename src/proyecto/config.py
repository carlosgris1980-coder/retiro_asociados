"""Constantes del proyecto: rutas, columnas, reglas de limpieza y supuestos del caso de negocio.

Todo lo que un notebook o una función necesita "saber" del proyecto vive aquí, en un solo lugar.
Cada valor tiene su origen escrito al lado (EDA, negocio o supuesto por validar).
"""
from pathlib import Path

# src/proyecto/config.py -> la raíz del repo está dos niveles arriba de src/
ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = ROOT / "data" / "raw" / "bd_confiar_2.xlsx"   # nunca se modifica
PROCESSED_DIR = ROOT / "data" / "processed"              # se regenera con el notebook 03; no se versiona
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# --- Columnas ---------------------------------------------------------------------------------
TARGET = "sol_retiro"          # 1 = radicó solicitud de retiro total; 0 = no radicó
ID_COL = "associate_id"        # identificador sustituto (0..n-1) que reemplaza al Nit al cargar
PII_COLUMNS = ["Nit"]          # documento del asociado: se elimina al cargar, antes de cualquier análisis

# Fechas del registro. Juntas reconstruyen el mes de proceso, que delata la respuesta (EDA, decisión 2):
# quien no está en el último mes del panel radicó o salió. Nunca entran al modelo.
PROCESS_COL = "anio_mes_proceso"
COHORT_COL = "cohorte"
TIME_COLUMNS = [PROCESS_COL, COHORT_COL]

# Columnas auxiliares que crea la limpieza: sirven para partir, diagnosticar y medir en la etapa 4.
# NO son variables del modelo (las dos primeras se derivan de la fecha de proceso o del objetivo).
AUX_COLUMNS = ["process_year", "exit_without_request", "income_is_min_wage", "stratum"]

# Variables candidatas para el modelo (las 21 originales + 3 indicadores de tenencia de producto).
BASE_FEATURES = [
    "saldo_aportes", "prom_saldo_capitalizacion_aportes", "porc_meses_capitaliza_aportes", "antig_meses",
    "saldo_ahorro_prog", "prom_saldo_ahorro_prog", "rel_min-max_ahorro_prog",
    "saldo_cta", "prom_saldo_cta", "rel_min-max_saldo_cta",
    "prom_saldo_cartera", "rel_min-max_saldo_cartera",
    "prom_asesorias_publiturno", "porc_meses_publiturno", "prom_trx",
    "edad", "femenino", "ingresos", "asalariado", "ed_superior", "zona_med",
]
PRODUCT_FLAGS = {  # un cero en el promedio de saldo significa "no tiene el producto", no "dato perdido"
    "has_loan": "prom_saldo_cartera",
    "has_prog_savings": "prom_saldo_ahorro_prog",
    "has_account": "prom_saldo_cta",
}
FEATURES = BASE_FEATURES + list(PRODUCT_FLAGS)

# Sospechosas de fuga (EDA, decisión 8): pueden reflejar la preparación del retiro (saldar el crédito, ir a la sucursal).
# El equipo confirmó que la ventana no incluye el mes de radicación; queda por aclarar por qué los porcentajes dividen
# por antigüedad + 1. No se descartan por intuición: en la etapa 4 el modelo se mide con y sin ellas.
LEAKAGE_SUSPECTS = ["porc_meses_publiturno", "prom_asesorias_publiturno", "rel_min-max_saldo_cartera"]

# --- Reglas de limpieza (salen del EDA) -------------------------------------------------------
# Rango posible de la edad de un asociado. Fuera de él el valor pasa a vacío (no se borra la fila).
# Por debajo de 14 los conteos son de 1 a 5 casos por edad y saltan a 48 en los 15 años; por encima de 100
# hay 15 casos, 10 de ellos con exactamente 125 años (valor centinela).
PLAUSIBLE_RANGE = {"edad": (14, 100)}

# Salario mínimo legal mensual vigente (SMMLV) de Colombia por año, en millones de pesos.
# El 23 % de los ingresos es exactamente uno de estos valores: el dato es el SMMLV del año en que se
# registró el ingreso, lo que lo vuelve una huella del calendario (EDA, decisión 9).
MIN_WAGE_BY_YEAR = {
    2017: 0.737717, 2018: 0.781242, 2019: 0.828116, 2020: 0.877803, 2021: 0.908526,
    2022: 1.000000, 2023: 1.160000, 2024: 1.300000, 2025: 1.423500,
}

# --- Caso de negocio (notebook 01) ------------------------------------------------------------
# Unidad monetaria: millones de pesos colombianos (confirmado por el equipo el 31-oct-2026; coincide con los
# ingresos, cuyos valores más frecuentes son el SMMLV de cada año).
MONEY_UNIT = "millones de COP"

# Supuestos de trabajo. NO salen de los datos: son el escenario de referencia, se reportan con un análisis
# de sensibilidad y deben reemplazarse por cifras reales (costo de campaña y piloto con grupo de control).
CONTACT_COST = 0.03     # costo de un contacto de fidelización (llamada + detalle) = $30.000 por asociado
RETENTION_RATE = 0.20   # de cada 10 asociados que iban a radicar y se contactan, 2 desisten
