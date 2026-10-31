"""Estilo único de gráficos del proyecto (ver especificaciones/estilo-graficos.md).

Todos los notebooks llaman a `set_style()` al inicio y usan estos colores y ayudas,
para que cada gráfico se vea igual y su título diga la conclusión, no el contenido.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Paleta. Contraste contra fondo blanco (WCAG): verde 7,4:1 · naranja 4,2:1 · gris oscuro 4,8:1.
STAY = "#2F5D59"        # verde profundo: asociado que permanece (no radica)
CHURN = "#C2611F"       # naranja quemado: asociado que radica solicitud de retiro
NEUTRAL = "#64748B"     # gris oscuro: referencia, contexto, totales
LIGHT = "#CBD5E1"       # gris claro: fondo de barras, elementos que no se leen
ALERT = "#B42318"       # rojo: líneas de referencia (azar, umbral, tope)
TEXT = "#1F2937"        # texto principal
SUBTEXT = "#475569"     # subtítulos y notas

TARGET_COLORS = {0: STAY, 1: CHURN}
TARGET_LABELS = {0: "No radica", 1: "Radica"}


def set_style() -> None:
    """Aplica el estilo del proyecto a matplotlib (fondo blanco, sin bordes sobrantes, rejilla suave)."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": TEXT,
        "axes.labelsize": 9.5,
        "axes.labelcolor": SUBTEXT,
        "axes.edgecolor": LIGHT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.8,
        "xtick.color": SUBTEXT,
        "ytick.color": SUBTEXT,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def titled(ax, conclusion: str, subtitle: str | None = None) -> None:
    """Título = la conclusión del gráfico; subtítulo = qué se está mostrando (más pequeño y gris)."""
    ax.set_title(conclusion, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=SUBTEXT, va="bottom", ha="left")


def pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    """Formatea un eje con proporciones (0–1) como porcentaje con coma decimal."""
    fmt = FuncFormatter(lambda v, _: f"{v * 100:.{decimals}f} %".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def decimal_axis(ax, axis: str = "y", decimals: int = 1) -> None:
    """Formatea un eje con coma decimal (0,2 en lugar de 0.2)."""
    fmt = FuncFormatter(lambda v, _: fmt_num(v, decimals))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def thousands_axis(ax, axis: str = "y") -> None:
    """Formatea un eje de conteos con punto de miles (14.000)."""
    fmt = FuncFormatter(lambda v, _: fmt_num(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fmt_pct(value: float, decimals: int = 1) -> str:
    """0.2093 -> '20,9 %' (formato en español)."""
    return f"{value * 100:.{decimals}f} %".replace(".", ",")


def fmt_num(value: float, decimals: int = 0) -> str:
    """194073 -> '194.073'; 1.18 -> '1,18' (separadores en español)."""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def table(df, decimals: int = 2, pct_cols: list | None = None, col_decimals: dict | None = None):
    """Tabla con formato en español (coma decimal, punto de miles).

    `pct_cols` se muestran como porcentaje; `col_decimals` fija los decimales de columnas puntuales.
    """
    import pandas as pd

    pct_cols = pct_cols or []
    col_decimals = col_decimals or {}
    formats = {}
    for c in df.select_dtypes("number").columns:
        if c in pct_cols:
            formats[c] = lambda v: fmt_pct(v)
        elif c in col_decimals:
            formats[c] = lambda v, d=col_decimals[c]: fmt_num(v, d)
        elif pd.api.types.is_integer_dtype(df[c]):
            formats[c] = lambda v: fmt_num(v, 0)
        else:
            formats[c] = lambda v, d=decimals: fmt_num(v, d)
    return df.style.format(formats)


def rate_by_quantile(x, y, q: int = 10):
    """Tasa de `y` (0/1) por tramos de `x` cortados en sus cuantiles (hasta `q` tramos).

    Los valores iguales quedan **siempre en el mismo tramo**: si una variable tiene muchos ceros, todos los ceros forman
    un solo tramo (más grande) y salen menos de `q` tramos. Repartir los empates por el orden de las filas sería un
    error, porque los datos pueden venir ordenados por la variable respuesta.

    Devuelve un DataFrame con la mediana de `x` en cada tramo, la tasa y el número de filas.
    """
    import pandas as pd

    x = pd.Series(x).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    bins = pd.qcut(x, q, labels=False, duplicates="drop")
    out = pd.DataFrame({"x": x, "y": y, "bin": bins}).groupby("bin").agg(
        x_median=("x", "median"), rate=("y", "mean"), n=("y", "size"))
    return out


def note(fig, text: str) -> None:
    """Nota al pie del gráfico (fuente, supuestos), en gris y pequeña."""
    fig.text(0.01, -0.02, text, fontsize=8, color=SUBTEXT, ha="left", va="top")
