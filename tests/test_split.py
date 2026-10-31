"""La regla más importante: ninguna entidad aparece en entrenamiento y prueba a la vez.

Los datos de prueba repiten cada entidad en varias filas a propósito: si alguien cambia la partición
por una partición por fila (`train_test_split`), `test_no_entity_in_two_partitions` falla.
"""
import numpy as np
import pandas as pd
import pytest

from proyecto.data.split import assert_disjoint, split_by_group


def _panel(n_entities: int = 300, rows_per_entity: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    strata = rng.choice(["activo", "radica", "salio_sin_radicar"], size=n_entities, p=[0.67, 0.21, 0.12])
    return pd.DataFrame({
        "associate_id": np.repeat(np.arange(n_entities), rows_per_entity),
        "stratum": np.repeat(strata, rows_per_entity),
        "x": rng.random(n_entities * rows_per_entity),
    })


def test_no_entity_in_two_partitions():
    df = _panel()
    train, test = split_by_group(df, test_size=0.2, seed=1)
    assert set(train["associate_id"]).isdisjoint(test["associate_id"])


def test_no_row_is_lost_and_test_share_is_close_to_requested():
    df = _panel()
    train, test = split_by_group(df, test_size=0.2, seed=1)
    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - 0.2) < 0.03


def test_split_keeps_the_mix_of_strata():
    df = _panel(n_entities=2000, rows_per_entity=1)
    train, test = split_by_group(df, test_size=0.2, seed=1)
    share_train = train["stratum"].value_counts(normalize=True)
    share_test = test["stratum"].value_counts(normalize=True)
    assert (share_train - share_test).abs().max() < 0.01


def test_assert_disjoint_raises_on_overlap():
    df = _panel(10)
    with pytest.raises(AssertionError):
        assert_disjoint(df.iloc[:12], df.iloc[9:20])
