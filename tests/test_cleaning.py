"""Una prueba por decisión de limpieza del EDA (reports/01_eda.md), con datos hechos a mano."""
import numpy as np
import pandas as pd

from proyecto.config import FEATURES, ID_COL, TARGET
from proyecto.data.cleaning import (
    add_product_flags,
    add_time_context,
    clean,
    drop_personal_data,
    drop_time_columns,
    flag_min_wage_income,
    mask_implausible,
)


def _frame() -> pd.DataFrame:
    """Cuatro asociados: activo, radica, salió sin radicar y uno con datos sucios."""
    return pd.DataFrame({
        "Nit": ["900", "901", "902", "903"],
        "anio_mes_proceso": ["2025-12", "2023-04", "2021-07", "2025-12"],
        "cohorte": ["2020-01", "2019-02", "2021-01", "2025-06"],
        "saldo_aportes": [1.0, 0.5, 0.0, 0.2],
        "prom_saldo_capitalizacion_aportes": [0.02, 0.01, 0.0, -0.1],
        "porc_meses_capitaliza_aportes": [0.5, 0.3, 0.0, 0.1],
        "antig_meses": [71, 50, 6, 6],
        TARGET: [0, 1, 0, 0],
        "saldo_ahorro_prog": [0.0, 1.0, 0.0, 0.0],
        "prom_saldo_ahorro_prog": [0.0, 0.8, 0.0, 0.0],
        "rel_min-max_ahorro_prog": [0.0, 0.1, 0.0, 0.0],
        "saldo_cta": [0.1, 0.0, 0.0, 0.0],
        "prom_saldo_cta": [0.2, 0.0, 0.0, 0.0],
        "rel_min-max_saldo_cta": [0.1, 0.0, 0.0, 0.0],
        "prom_saldo_cartera": [3.0, 0.0, 0.0, 0.0],
        "rel_min-max_saldo_cartera": [0.2, 0.0, 0.0, 0.0],
        "prom_asesorias_publiturno": [0.3, 0.5, 0.0, 1.0],
        "porc_meses_publiturno": [0.2, 0.4, 0.0, 0.3],
        "prom_trx": [2.0, 1.0, 0.0, 0.5],
        "edad": [45, 30, 3, 125],
        "femenino": [1, 0, 1, 0],
        "ingresos": [2.5, 1.16, 0.738, 1.4235],
        "asalariado": [1, 1, 0, 1],
        "ed_superior": [0, 1, 0, 0],
        "zona_med": [1, 0, 0, 0],
    })


def test_drop_personal_data_removes_nit_and_adds_surrogate_id():
    out = drop_personal_data(_frame())
    assert "Nit" not in out.columns
    assert out[ID_COL].tolist() == [0, 1, 2, 3]


def test_time_context_marks_exit_without_request_and_stratum():
    out = add_time_context(_frame())
    assert out["process_year"].tolist() == [2025, 2023, 2021, 2025]
    # solo el asociado 902 no radicó y su registro termina antes del último mes (2025-12)
    assert out["exit_without_request"].tolist() == [0, 0, 1, 0]
    assert out["stratum"].tolist() == ["activo", "radica", "salio_sin_radicar", "activo"]


def test_drop_time_columns_removes_both_dates():
    out = drop_time_columns(_frame())
    assert "anio_mes_proceso" not in out.columns and "cohorte" not in out.columns


def test_mask_implausible_age_keeps_rows_and_sets_missing():
    out = mask_implausible(_frame())
    assert len(out) == 4
    assert out["edad"].iloc[:2].tolist() == [45.0, 30.0]
    assert out["edad"].iloc[2:].isna().all()          # 3 y 125 años no son posibles


def test_min_wage_income_flag_matches_only_exact_values():
    out = flag_min_wage_income(_frame())
    # 1,16 (2023) y 1,4235 (2025) son salarios mínimos exactos; 0,738 está cerca de 2017 pero no es exacto
    assert out["income_is_min_wage"].tolist() == [0, 1, 0, 1]


def test_product_flags_read_zero_balance_as_no_product():
    out = add_product_flags(_frame())
    assert out["has_loan"].tolist() == [1, 0, 0, 0]
    assert out["has_prog_savings"].tolist() == [0, 1, 0, 0]
    assert out["has_account"].tolist() == [1, 0, 0, 0]


def test_clean_keeps_rows_and_leaves_no_pii_or_dates():
    raw = drop_personal_data(_frame())
    out = clean(raw)
    assert len(out) == len(raw)
    assert not {"Nit", "anio_mes_proceso", "cohorte"} & set(out.columns)
    assert set(FEATURES) <= set(out.columns)
    assert np.isnan(out.loc[3, "edad"])
    # la limpieza no modifica el DataFrame que recibe
    assert raw["edad"].tolist() == [45, 30, 3, 125]
