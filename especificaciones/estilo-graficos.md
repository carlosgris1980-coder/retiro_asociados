# Estilo de gráficos y tablas

> Memoria del proyecto: Claude la lee antes de hacer cualquier gráfico. Así todos los notebooks salen con el mismo formato.
> Implementación: `src/proyecto/viz.py` (cada notebook llama a `set_style()` al inicio).

## Reglas

1. **El título dice la conclusión, no el contenido.** «Quien radica casi nunca tiene crédito vigente», no
   «Distribución de saldo de cartera por clase». Lo que se muestra va en el **subtítulo**, en gris.
2. **Un mensaje por gráfico.** Si hay que explicar dos cosas, son dos gráficos.
3. **Siempre contra la variable respuesta.** Del EDA en adelante, cada variable se mira separando a quien radica de
   quien no.
4. **Poco texto dentro del gráfico.** Etiquetas solo donde se leen cifras clave; la explicación va en el markdown
   que sigue al gráfico.
5. **Cifras en español:** coma decimal y punto de miles (`20,9 %`, `194.073`). Usar `fmt_pct` y `fmt_num`.
6. **Ejes que no mienten:** las barras empiezan en cero; si una variable tiene colas largas se dice en el
   subtítulo y se usa escala logarítmica o recorte al p99, nunca en silencio.
7. **Fondo blanco, sin bordes superiores ni derechos, rejilla suave.** Sin 3D, sin tortas.

## Paleta (contraste contra blanco, WCAG)

| Uso | Nombre en código | Color | Contraste |
|---|---|---|---|
| Asociado que **permanece** (`sol_retiro = 0`) | `STAY` | `#2F5D59` verde profundo | 7,4 : 1 |
| Asociado que **radica** (`sol_retiro = 1`) | `CHURN` | `#C2611F` naranja quemado | 4,2 : 1 |
| Referencia, contexto, totales | `NEUTRAL` | `#64748B` gris oscuro | 4,8 : 1 |
| Fondo de barras, elementos secundarios | `LIGHT` | `#CBD5E1` gris claro | (no lleva texto) |
| Líneas de referencia (azar, umbral, tope) | `ALERT` | `#B42318` rojo | 6,6 : 1 |
| Texto / subtítulos | `TEXT` / `SUBTEXT` | `#1F2937` / `#475569` | 14,7 : 1 / 7,6 : 1 |

La pareja verde–naranja se distingue también en escala de grises y para daltonismo rojo-verde (difieren en
luminosidad y en tono azul-amarillo). Parte de la paleta que ya usaba el equipo (`Confiar_prediccion_retiro_v2.ipynb`),
con el naranja y el agua oscurecidos porque estaban por debajo de 3 : 1.

## Tipos de gráfico por pregunta

| Pregunta | Gráfico |
|---|---|
| ¿Cómo está el objetivo? | Barra horizontal apilada al 100 % |
| ¿Qué tan distinta es una variable entre clases? | Tasa de radicación por tramo (deciles o categorías), con la tasa global como línea de referencia |
| ¿Qué variables separan más? | Barras horizontales ordenadas (AUC univariado o diferencia de medianas) |
| ¿Hay colas pesadas o valores imposibles? | Tabla de percentiles (p50 … p99, máximo); histograma en escala log si hace falta |
| ¿Cambia en el tiempo? | Línea por periodo, con la tasa global de referencia |
| ¿Están correlacionadas? | Mapa de calor triangular, divergente centrado en 0 (`vlag`), solo con números de dos cifras |
