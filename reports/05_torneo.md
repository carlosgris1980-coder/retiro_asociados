# Informe del torneo de modelos

> Resultado de `notebooks/05_torneo.ipynb`. El torneo **descarta**: elige 1–2 finalistas con evidencia, no un campeón
> definitivo. Cada pliegue de cada modelo: `reports/05_torneo_pliegues.csv`; tabla resumen: `reports/05_torneo_tabla.csv`.

## Reglas del torneo

- **Mismos datos y mismos pliegues para todos:** 155.258 asociados de entrenamiento, 5 pliegues por asociado
  (`associate_id`), estratificados por grupo, los mismos del notebook 04 (`data/processed/folds.joblib`). La prueba sigue
  cerrada.
- **Mismo trato:** las 27 variables decididas en `reports/04_features.md` y el mismo pipeline (variables nuevas,
  imputación, logaritmo, escalado) para todos; parámetros por defecto de scikit-learn; **ningún tratamiento del
  desbalance** (ni `class_weight` ni remuestreo), porque mueve el punto de operación sin mejorar la capacidad. El umbral se
  elige después, para el modelo ya elegido.
- **Línea base:** `DummyClassifier(strategy="most_frequent")`, siempre "no radica". No compite: mide cuánto aporta modelar.
  AUC 0,5; PR-AUC 0,2092 (la tasa de radicación); exactitud 79,1 %.
- **Métricas:** AUC-ROC y PR-AUC (ordenamiento, no dependen del umbral); recall, precisión y F1 de la clase 1 en el umbral
  por defecto 0,5; puntaje de Brier (calidad de las probabilidades); y la de negocio: **beneficio neto por cada 1.000
  asociados** con la regla de valor esperado del notebook 01 (contacto $30.000, efectividad 20 %). **Sin exactitud.**
- **Regla:** toda diferencia se compara con la variación entre pliegues; menor que esa variación no es un hallazgo.

## Tabla comparativa

> Media entre pliegues (± desviación). Ordenada por AUC.

| Modelo | AUC (±) | PR-AUC (±) | Recall (0,5) | Precisión (0,5) | F1 (0,5) | Brier | Beneficio neto por 1.000 (±) | Segundos de ajuste* |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | **0,8951** (0,0017) | **0,7015** (0,0064) | 0,5056 | 0,7429 | 0,6016 | **0,1005** | **33,63** (0,74) | 14,9 |
| Gradient Boosting | 0,8877 (0,0015) | 0,6827 (0,0069) | 0,4713 | 0,7254 | 0,5714 | 0,1044 | 33,58 (0,69) | 48,8 |
| Regresión Logística | 0,8189 (0,0020) | 0,5145 (0,0047) | 0,2810 | 0,6146 | 0,3856 | 0,1291 | 32,17 (0,64) | 0,9 |
| K-Vecinos | 0,7921 (0,0015) | 0,4843 (0,0018) | 0,4315 | 0,5769 | 0,4937 | 0,1348 | 27,50 (0,67) | 0,6 (16,6 para predecir) |
| Árbol de Decisión | 0,6988 (0,0031) | 0,3713 (0,0031) | 0,5289 | 0,5158 | 0,5222 | 0,2024 | 15,39 (0,44) | 2,9 |
| Baseline (clase mayoritaria) | 0,5000 | 0,2092 | 0 | 0 | 0 | 0,2092 | 0,00 | 0,6 |
| *Contactar a todos (sin modelo)* | — | — | 1 | 0,209 | — | — | 18,75 (0,62) | — |

\* Los tiempos varían un poco entre ejecuciones.

## ¿Coinciden los rankings de negocio y técnico?

| Modelo | AUC | PR-AUC | Beneficio neto | Esfuerzo a igual detección | Brier | Recall (0,5) | F1 (0,5) |
|---|---|---|---|---|---|---|---|
| Random Forest | 1 | 1 | 1 | 1 | 1 | 2 | 1 |
| Gradient Boosting | 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| Regresión Logística | 3 | 3 | 3 | 3 | 3 | 5 | 5 |
| K-Vecinos | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Árbol de Decisión | 5 | 5 | 5 | 5 | 5 | 1 | 3 |

- **Sí coinciden** en todo lo que no depende del umbral de fábrica. Solo recall y F1 en 0,5 premian al árbol, porque sus
  probabilidades son 0 o 1 y "dice sí" con facilidad, no porque detecte mejor.
- **Ningún modelo tiene recall nulo**, pero en el umbral 0,5 todos dejan fuera a muchos de quienes radican (la logística
  detecta solo el 28 %): el punto de operación se elige después.

## Comparación pareada de los mejores

| Mejor | Contra | Métrica | Diferencia media | Pliegues que gana | Diferencia / ruido |
|---|---|---|---|---|---|
| Random Forest | Gradient Boosting | AUC | 0,0075 | 5 de 5 | 12,1 |
| Random Forest | Gradient Boosting | PR-AUC | 0,0188 | 5 de 5 | 4,6 |
| Random Forest | Gradient Boosting | Beneficio neto | 0,045 | 5 de 5 | **0,8 (empate)** |
| Random Forest | Regresión Logística | Beneficio neto | 1,459 | 5 de 5 | 7,2 |
| Gradient Boosting | Regresión Logística | Beneficio neto | 1,414 | 5 de 5 | 9,1 |

Frente al líder, todas las diferencias de AUC superan el ruido (Gradient Boosting a 4,3 veces; la logística a 44 veces).
El líder supera al baseline por 0,3951 de AUC, 228 veces su variación entre pliegues.

## ¿Cuánto vale la diferencia?

- **Frente a contactar a todos:** Random Forest deja **14,87 millones más por cada 1.000 asociados**, con 406 contactos
  por cada 1.000 en lugar de 1.000, y cubre el 94,0 % del saldo de aportes en riesgo. Orden de magnitud: aplicado una vez a
  los 129.794 activos, unos 1.930 millones (supuestos de costo y efectividad; tasa real de los activos por confirmar).
- **Entre los dos mejores:** Random Forest y Gradient Boosting empatan en beneficio (0,045 millones por cada 1.000).
- **El costo de la explicabilidad:** la Regresión Logística deja 1,46 millones menos por cada 1.000 asociados que
  Random Forest y, para detectar al 75 % de quienes radican, necesita unos 15.000 contactos más (57.380 frente a 42.344
  en los 155.258 asociados de entrenamiento; unos 451 millones en contactos con el supuesto de $30.000).

## Peor caso: sin las variables sospechosas de fuga

| Modelo | PR-AUC con todas → sin sospechas | AUC con todas → sin sospechas | Beneficio con todas → sin sospechas |
|---|---|---|---|
| Random Forest | 0,7015 → 0,5946 | 0,8951 → 0,8401 | 33,63 → 31,26 |
| Gradient Boosting | 0,6827 → 0,5743 | 0,8877 → 0,8304 | 33,58 → 30,95 |
| Regresión Logística | 0,5145 → 0,4374 | 0,8189 → 0,7516 | 32,17 → 29,66 |

Aun en el peor caso, el modelo supera con holgura a contactar a todos (18,75) y el orden de los modelos no cambia.

## Finalistas

| Finalista | Por qué (consistencia, métrica, tiempo, interpretabilidad) | Qué podría cambiar con la optimización |
|---|---|---|
| **Random Forest** | Primero en AUC, PR-AUC, Brier, beneficio y esfuerzo; gana a Gradient Boosting en los 5 pliegues en ordenamiento; unos 15 s por ajuste | Limitar profundidad, hojas mínimas y variables por árbol: mejor calibración y un modelo más liviano para la app |
| **Gradient Boosting** | Segundo en todo; empatado en beneficio con Random Forest (0,8 veces el ruido); la familia que más suele ganar al optimizar | Tasa de aprendizaje, número y profundidad de árboles; la versión histograma entrena mucho más rápido |

**Modelos descartados y por qué:**

- **Regresión Logística:** su ventaja es la explicabilidad, y el torneo mide su costo (1,46 millones por cada 1.000
  asociados y unos 15.000 contactos más a igual detección). Es una decisión de negocio: se conserva como referencia
  interpretable; Random Forest con SHAP puede dar la explicación por asociado.
- **K-Vecinos:** peor en todo, lento para predecir y con solo 6 valores de probabilidad: no permite elegir un punto de
  operación fino.
- **Árbol de Decisión:** con parámetros por defecto memoriza; queda por debajo de contactar a todos.

**Por qué el torneo no elige un campeón:** la ventaja de Random Forest sobre Gradient Boosting es real en ordenamiento pero
desaparece en beneficio neto, y la optimización puede cambiar el orden. Además, la prueba sigue cerrada, el umbral y la
calibración no están elegidos y los supuestos de negocio están por confirmar.

## Efectos que la métrica principal no ve

- **La regla de valor esperado contacta a cerca del 40 %** aunque radique el 21 %, porque los asociados con mucho saldo
  valen la llamada aun con probabilidad baja. Hay que confirmar con mercadeo que esa capacidad existe.
- **Probabilidades gruesas** (K-Vecinos: 6 valores; árbol: 2) impiden fijar un punto de operación: a igual meta de 75 % de
  detección, K-Vecinos salta al 86 % contactando a 79.189 asociados y el árbol tiene que contactar a todos.
- **Dependencia de las sospechas de fuga:** sin ellas el PR-AUC de Random Forest cae 0,107. El piloto con grupo de control
  es la prueba definitiva.
