"""Partición entrenamiento / prueba por la unidad de análisis.

En este dataset cada asociado aparece una sola vez (194.073 Nit distintos en 194.073 filas), así que
partir por asociado y partir por fila dan lo mismo. Aun así la función parte **por grupo**: si el equipo
reconstruye el dataset como panel (un asociado en varios meses), la partición seguirá siendo honesta
sin cambiar una línea. Además estratifica, para que entrenamiento y prueba tengan la misma mezcla de
asociados que radican, que siguen activos y que salieron sin radicar.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from proyecto.config import ID_COL, RANDOM_STATE, TEST_SIZE


def split_by_group(
    df: pd.DataFrame,
    group_col: str = ID_COL,
    strat_col: str = "stratum",
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento y prueba sin repartir ninguna entidad entre los dos lados, estratificando por `strat_col`.

    Usa un pliegue de `StratifiedGroupKFold` con 1/test_size pliegues (0,20 -> 5 pliegues -> 20 % de prueba).
    """
    n_splits = round(1 / test_size)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df[strat_col], groups=df[group_col]))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)
    assert_disjoint(train, test, group_col=group_col)
    return train, test


def make_folds(
    df: pd.DataFrame,
    group_col: str = ID_COL,
    strat_col: str = "stratum",
    n_splits: int = 5,
    seed: int = RANDOM_STATE,
) -> list[tuple]:
    """Pliegues de validación cruzada por entidad y estratificados. Se crean UNA vez y se reutilizan en todo."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(df, df[strat_col], groups=df[group_col]))
    for train_idx, valid_idx in folds:
        assert_disjoint(df.iloc[train_idx], df.iloc[valid_idx], group_col=group_col)
    return folds


def assert_disjoint(*frames: pd.DataFrame, group_col: str = ID_COL) -> None:
    """Falla si alguna entidad aparece en más de una partición."""
    seen: set = set()
    for i, frame in enumerate(frames):
        entities = set(frame[group_col])
        overlap = seen & entities
        if overlap:
            raise AssertionError(f"{len(overlap)} entidades aparecen en más de una partición (partición {i})")
        seen |= entities
