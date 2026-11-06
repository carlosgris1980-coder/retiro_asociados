# Ficha del modelo — Priorización de contacto para la retención de asociados

> Generada por `notebooks/09_inferencia.ipynb` el 2026-11-06 con las cifras de los notebooks 07 y 08.

## Qué es

Estima la probabilidad de que un asociado de Confiar **radique la solicitud de retiro total de aportes**, para que el equipo
de fidelización **priorice a quién contactar antes**. Apoya la decisión; **no decide por sí solo**.

- **Modelo:** Random Forest sin calibrar (200 árboles, profundidad máxima 14, hojas de al menos
  15 asociados, 27 % de las variables por corte).
- **Registro:** `retiro_asociados_rf` versión 1, alias `champion` (MLflow local).
  Exportado a `models/model.joblib`.
- **Regla de decisión:** contactar si *probabilidad × 20 % × saldo de aportes > 0,03
  millones* (valor esperado del contacto mayor que su costo). Supuestos de costo y efectividad **por validar**.

## Datos de entrenamiento

- **Fuente:** datos internos de Confiar anonimizados por el equipo (`bd_confiar_2.xlsx`); asociados vinculados entre 2017
  y 2025, un registro por asociado. Montos en millones de pesos.
- **Tamaño:** 155.258 asociados de entrenamiento; prueba de 38.815 asociados.
- **Partición:** por asociado, estratificada; la prueba se abrió **una sola vez** (decisiones congeladas el
  2026-11-05T20:35:20).
- **Datos personales:** el `Nit` se eliminó al cargar; las fechas del registro se eliminaron por fuga.

## Rendimiento (prueba, asociados nunca vistos)

| Métrica | Valor | Intervalo del 95 % (remuestrea asociados) |
|---|---|---|
| Beneficio neto por cada 1.000 asociados (millones, principal) | 33,60 | 31,81 a 35,40 |
| Beneficio frente a contactar a todos (modelo − contactar a todos) | 14,81 | 14,61 a 15,00 |
| PR-AUC (identificación de quien radica) | 0,700 | 0,690 a 0,711 |
| PR-AUC frente al baseline (modelo − baseline) | 0,491 | 0,482 a 0,500 |
| AUC | 0,896 | 0,893 a 0,899 |
| Brier (calidad de las probabilidades) | 0,101 | 0,100 a 0,103 |

Con la regla de valor esperado contacta al 41 % de los asociados,
encuentra al 75,7 % de quienes radican y cubre el 94,1 % del saldo
de aportes en riesgo.

## Variables que más pesan (SHAP)

- `porc_meses_publiturno`: 15,5 % de la influencia (valores altos suben el riesgo)
- `saldo_aportes`: 12,3 % de la influencia (valores altos suben el riesgo)
- `porc_meses_capitaliza_aportes`: 11,5 % de la influencia (valores altos suben el riesgo)
- `rel_min-max_saldo_cartera`: 11,4 % de la influencia (valores altos bajan el riesgo)
- `ingresos`: 8,6 % de la influencia (valores altos bajan el riesgo)
- `antig_meses`: 7,9 % de la influencia (valores altos bajan el riesgo)

Explican **al modelo, no al mundo**: no son causas del retiro.

## Límites y usos no recomendados

- **Asociados nuevos (0 a 6 meses):** encuentra solo al 49,2 % de quienes radican (poca historia y poco
  saldo). Usar también una estrategia de bienvenida.
- **Sospechas de fuga:** visitas a sucursal, cartera mínima en cero, ingreso y tendencias de saldo suman cerca del 40 % de la
  influencia. El equipo confirmó que no incluyen el mes de radicación; falta aclarar la ventana de cálculo. Sin ellas el modelo
  seguía superando a contactar a todos (notebook 05).
- **Supuestos de negocio:** las cifras en pesos dependen del costo del contacto ($30.000) y de la efectividad de la retención
  (20 %). Validarlos con un piloto con grupo de control.
- **Capacidad:** la regla contacta a cerca del 41 % de los asociados; confirmar que mercadeo puede hacerlo.
- **Atributos sensibles** (edad, sexo, zona, educación, ocupación): pesan poco y el error por segmento es parecido entre
  grupos, pero deben revisarse con el área jurídica antes de producción.
- **Qué no hacer:** negar productos o servicios; leer las variables como causas; decidir sobre grupos sociodemográficos.
- **No se evaluó:** estabilidad en el tiempo (deriva), comportamiento con nuevas cohortes o productos, efecto real de la
  campaña. Recalibrar y revisar cada seis meses.

## Reproducibilidad

Código en `src/proyecto/`, pruebas en `tests/` (`pytest -q`), notebooks 01 a 09 en `notebooks/`. App: `streamlit run
app/streamlit_app.py` (carga `models/model.joblib` y `models/input_schema.json`; no consulta Databricks).
