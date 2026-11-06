# Informe de la optimización

> Resultado de `notebooks/06_optimizacion.ipynb`. La mejor puntuación de una búsqueda es **optimista**: se comprueba en
> asociados que la búsqueda no vio. Código: `src/proyecto/models/tuning.py` (construido sobre el `tuning.py` del ejemplo
> del curso). Tablas: `06_optimizacion_tabla.csv` (cada ensayo), `06_comprobacion.csv`, `06_intervalos.csv`; decisión en
> `06_best_params.json`.

## Diseño de la búsqueda

- **Finalista optimizado:** Random Forest (primero del torneo, notebook 05).
- **Objetivos del equipo:** evitar el sobreajuste al máximo y optimizar la identificación de la clase minoritaria.
- **Métrica optimizada:** PR-AUC en validación cruzada (identificación de quien radica, sin depender del umbral). Controles
  registrados en cada ensayo: beneficio neto, Brier, AUC, recall.
- **Restricción contra el sobreajuste (fijada antes de buscar):** brecha de PR-AUC (ajuste − validación) de 0,10 o menos,
  un tercio de la del modelo por defecto (0,3043). Solo los ensayos que la cumplen son admisibles y elegibles.
- **Espacio de búsqueda:**

| Parámetro | Rango | Motivo |
|---|---|---|
| `max_depth` | 6 a 24 | Por defecto no tiene límite y el árbol aísla a cada asociado |
| `min_samples_leaf` | 1 a 100 (escala log) | Hojas más grandes = reglas más generales |
| `max_features` | 0,15 a 0,80 | Menos variables por corte = árboles más distintos, menos varianza |
| `max_samples` | 0,50 a 1,00 | Cada árbol ve una parte de los asociados: más diversidad, menos memoria |
| `n_estimators` | fijo en 200 | Más árboles no sobreajustan: se afina la forma, no la cantidad |

- **Presupuesto:** 20 ensayos, fijados antes (la guía recomienda 12–20). Cada ensayo: 3 ajustes de 200 árboles.
- **Entidades reservadas:** 25 % de los asociados del entrenamiento (38.815), por asociado, estratificado, semilla 7. La
  búsqueda usa los otros 116.443, con 3 pliegues por asociado. La prueba sigue cerrada.
- **`class_weight`:** fuera de la búsqueda (mueve el punto de operación y distorsiona las probabilidades que usa la regla
  de valor esperado); se midió de forma explícita en la comprobación.

## Seguimiento (tracking)

- **Dónde quedó registrado:** MLflow **local** (`mlflow_local.db`, sin `.env` de Databricks), experimento
  `retiro-asociados-optimizacion`.
- **¿Está cada ensayo como una ejecución con parámetros, métricas y etiquetas?** ☒ sí: 20 ensayos anidados en una
  ejecución madre, con todas las métricas (y su desviación entre pliegues) y las etiquetas `etapa`, `modelo`,
  `metrica_optimizada` y `admisible`.
- **Mejor ejecución admisible:** `random-forest-ensayo-19`, PR-AUC 0,6966 (± 0,0030), brecha 0,0881.
- **Estudio reanudable:** `data/processed/06_optuna_rf.db` (si se corta, se retoma sin duplicar ensayos).

## Resultados de la búsqueda

- **Punto de partida** (Random Forest por defecto, misma validación): PR-AUC 0,6957 en validación y **1,0000** en lo que vio
  (brecha 0,3043); AUC 0,8928 frente a 1,0000.
- **14 de 20 ensayos admisibles.** Los cuatro de mayor PR-AUC (0,7065 a 0,7075) tienen hojas de 1 a 5 asociados y brechas de
  0,22 a 0,26: los últimos 0,01 de PR-AUC se pagan memorizando.
- **Mejor admisible (ensayo 19):** PR-AUC 0,6966, brecha 0,0881; `max_depth` 14, `min_samples_leaf` 15, `max_features`
  0,269, `max_samples` 0,926.
- **Importancia de los parámetros:** `min_samples_leaf` 0,477; `max_features` 0,311; `max_depth` 0,205; `max_samples` 0,007.
- Con brechas por debajo de 0,05 el PR-AUC cae (0,640 a 0,686): hay un punto de equilibrio.

## Comprobación honesta (asociados reservados)

> Los tres modelos se entrenan con los 116.443 asociados de búsqueda y se evalúan en los 38.815 reservados. Intervalo del
> 95 % con 500 remuestreos de asociados.

| Modelo | PR-AUC reservados | PR-AUC en lo que vio | Brecha | AUC | Brier | Beneficio neto por 1.000 | Contactos para detectar el 75 % |
|---|---|---|---|---|---|---|---|
| Partida (por defecto) | 0,6993 | 1,0000 | 0,3007 | 0,8937 | 0,1010 | 33,44 | 10.567 |
| **Optimizado** | 0,6971 | 0,7707 | **0,0737** | 0,8932 | 0,1022 | 33,44 | 10.639 |
| Optimizado + `class_weight` | 0,6890 | 0,7475 | 0,0585 | 0,8916 | 0,1319 | 32,18 | 10.725 |

| Comparación | Métrica | Diferencia | Intervalo 95 % | Lectura |
|---|---|---|---|---|
| Optimizado − partida | PR-AUC | −0,0022 | [−0,0059 ; +0,0014] | no se distinguen |
| Optimizado − partida | Beneficio neto | −0,0072 | [−0,1361 ; +0,1218] | no se distinguen |
| Optimizado − partida | Brier | +0,0012 | [+0,0007 ; +0,0017] | partida algo mejor calibrada |
| `class_weight` − optimizado | PR-AUC | −0,0081 | [−0,0096 ; −0,0064] | `class_weight` peor |
| `class_weight` − optimizado | Beneficio neto | −1,2562 | [−1,4134 ; −1,0928] | `class_weight` peor |
| `class_weight` − optimizado | Brier | +0,0297 | [+0,0280 ; +0,0313] | `class_weight` peor |

**Regla de decisión escrita antes de mirar:** se adoptan los optimizados si (a) todo el intervalo de la diferencia de
PR-AUC está por encima de cero, o (b) el intervalo cruza el cero, la brecha es menor y el beneficio neto no empeora (su
intervalo no queda entero por debajo de cero).

**Resultado de la regla:** (a) no; (b) **sí**: PR-AUC no se distingue, la brecha baja de 0,301 a 0,074 y el beneficio no
empeora.

## Decisión

- **Parámetros que se usarán:** los **optimizados** (ensayo 19, con 200 árboles). Misma capacidad de identificar a quien
  radica y de generar beneficio, con un cuarto del sobreajuste: un modelo que generaliza y es más estable.
- **`class_weight` se descarta con evidencia:** sube el recall en el umbral 0,5 (0,478 → 0,821) a costa de la precisión
  (0,748 → 0,513), y empeora PR-AUC, Brier y beneficio neto, con intervalos que no tocan el cero.
- **Lo que no pudimos concluir:** que optimizar mejore la detección. No la mejoró: el modelo por defecto ya estaba en el
  techo que se alcanza sin memorizar. La ganancia de esta etapa es de robustez, no de puntaje.

## Para la clase siguiente

- Notebook 07: abrir la prueba **una sola vez** con el modelo optimizado; revisar la calibración (perdió 0,0012 de Brier);
  elegir el punto de operación (regla de valor esperado o umbral), que es donde se gana detección; explicar con SHAP;
  revisar el error por segmento (sexo, edad, zona).
- Llevar el seguimiento a Databricks completando `.env` (el código no cambia).
