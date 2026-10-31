# Informe de variables

> Resultado de `notebooks/04_features.ipynb`. Cada idea de variable es una **hipótesis medida** con la misma
> validación; se conserva solo si la mejora supera el ruido entre pliegues. Código: `src/proyecto/features/engineering.py`
> (variables y preprocesamiento) y `src/proyecto/models/train.py` (pipeline y evaluación). Cada pliegue de cada variante:
> `reports/04_features_pliegues.csv`.

## Cómo se midió

- **Modelos (parámetros fijos):** regresión logística (`max_iter=2000`) y boosting de árboles
  (`HistGradientBoostingClassifier` por defecto), los dos con semilla 42. El primero es sensible a cómo se presentan las
  variables; el segundo representa a los modelos de árboles del torneo.
- **Métrica de decisión:** PR-AUC (la clase que interesa es minoritaria, 20,9 %). **Control:** beneficio neto por cada
  1.000 asociados (regla de valor esperado del notebook 01) y AUC.
- **Pliegues:** 5, por asociado (`associate_id`) y estratificados por grupo (radicó / activo / salió sin radicar),
  semilla 42. Guardados en `data/processed/folds.joblib` y **reutilizados en el torneo**. Cada pliegue: 124.206 de
  ajuste y 31.052 de validación, tasa de radicación 0,209 en todos.
- **Preprocesamiento (dentro del Pipeline, ajustado solo con la parte de ajuste):** variables nuevas, imputación con la
  mediana, logaritmo con signo en colas largas y escalado.
- **Regla de decisión escrita antes de medir:** agregar una variable si su mejora / ruido del PR-AUC es **> 1 en al menos
  uno de los dos modelos y > −1 en el otro**, y se conoce al decidir. Quitar variables si la pérdida queda **dentro del
  ruido en los dos** (mejora / ruido > −1 en ambos). Sospechas de fuga y población: el número dice cuánto pesa la
  decisión; la decisión se toma con el conocimiento del negocio.

## Tabla de ideas medidas

> Mejora / ruido = mejora del PR-AUC frente a la referencia dividida por la desviación entre pliegues de la variante.

| # | Idea | ¿Se conoce al decidir? | PR-AUC logística (± ruido) | Mejora / ruido logística · árboles | Decisión |
|---|---|---|---|---|---|
| 0 | **Referencia** (24 variables) | — | 0,4722 (± 0,0042) · árboles 0,7167 (± 0,0064) | 0 · 0 | — |
| 1 | `in_risk_window` (antigüedad entre 7 y 24 meses) | sí | 0,4910 (± 0,0045) | 4,21 · −0,02 | ☒ conservar |
| 2 | `is_dormant` (sin aportes o sin transacciones) | sí | 0,4923 (± 0,0034) | 5,82 · 0,00 | ☒ conservar |
| 3 | `n_products` (crédito + ahorro programado + cuenta) | sí | 0,4722 (± 0,0042) | −0,01 · 0,03 | ☒ descartar |
| 4 | `savings_to_contributions` ((ahorro + cuenta) / aportes) | sí | 0,5024 (± 0,0042) | 7,22 · 0,04 | ☒ conservar |
| 5 | `contributions_to_income` (aportes / ingresos) | sí (ingreso sospechoso) | 0,4722 (± 0,0042) | −0,01 · 0,01 | ☒ descartar |
| 6 | `account_balance_trend` (saldo cuenta actual / promedio) | sí, hereda sospecha | 0,4764 (± 0,0038) | 1,08 · 0,02 | ☒ conservar (vigilada) |
| 7 | `prog_savings_trend` (ahorro programado actual / promedio) | sí, hereda sospecha | 0,4780 (± 0,0051) | 1,13 · −0,00 | ☒ conservar (vigilada) |
| 8 | Sin logaritmo en colas largas | — | 0,4546 (± 0,0041) | −4,30 · −0,01 | ☒ conservar el logaritmo |
| 9 | Sin `prom_asesorias_publiturno` ni `prom_saldo_capitalizacion_aportes` | — | 0,4707 (± 0,0044) | −0,36 · 0,10 | ☒ quitarlas |
| 10 | Sin las 8 variables débiles del EDA | — | 0,4506 (± 0,0037) | −5,91 · −2,04 | ☒ conservarlas |
| 11 | Sin las 4 sospechas de fuga | — | 0,4171 (± 0,0067) · árboles 0,6153 (± 0,0100) | −8,26 · −10,17 | ☒ conservarlas, con condiciones |
| 12 | Entrenar sin los 23.688 retiros unilaterales | — | 0,4620 (± 0,0039) · árboles 0,6914 (± 0,0060) | −2,65 · −4,23 | ☒ conservarlos en el entrenamiento |

**Selección final (27 variables), medida:** logística 0,4722 → **0,5145** (mejora / ruido 8,92, gana en 5 de 5 pliegues;
AUC 0,8189); árboles 0,7167 → **0,7170** (0,05, dentro del ruido; AUC 0,9006). Beneficio neto por 1.000 asociados:
32,17 (logística) y 33,99 (árboles) millones de COP.

## Decisiones y motivos

- **Se conservan 5 variables nuevas** (`in_risk_window`, `is_dormant`, `savings_to_contributions`,
  `account_balance_trend`, `prog_savings_trend`): todas mejoran la logística por encima del ruido y ganan en los 5
  pliegues; ninguna empeora a los árboles. Las dos tendencias pasan la regla por poco (1,08 y 1,13).
- **Se descartan `n_products` y `contributions_to_income`**: dentro del ruido en los dos modelos. `n_products` es la
  suma de tres indicadores que ya están en la referencia.
- **Las variables nuevas ayudan al modelo lineal y no a los árboles**: son "atajos" (cortes y razones) que los árboles
  construyen solos. Se usan para todos los modelos del torneo para que nadie compita en condiciones distintas.
- **Se confirma el logaritmo** en colas largas (decisión 10 del EDA): quitarlo cuesta −4,30 veces el ruido en la logística.
- **Se quitan** `prom_asesorias_publiturno` (ρ = 0,94 con `porc_meses_publiturno`) y
  `prom_saldo_capitalizacion_aportes` (ρ = 0,84 con `saldo_aportes`): perderlas queda dentro del ruido. Parsimonia.
- **Se conservan las 8 variables débiles**: por separado casi no separan, pero juntas sí aportan.
- **Se conservan las sospechas de fuga** que quedan (`porc_meses_publiturno`, `rel_min-max_saldo_cartera`, `ingresos`)
  porque el equipo confirmó que las variables no incluyen el mes de radicación. Pero el modelo **depende mucho** de ellas:
  sin ellas el PR-AUC cae 0,055 (logística) y 0,101 (árboles).
- **Los retiros unilaterales se quedan en el entrenamiento** con `sol_retiro = 0`: sacarlos empeora el PR-AUC en los dos
  modelos, evaluado sobre todos los asociados (como en producción).

## Supuestos y riesgos

- Las sospechas de fuga solo son legítimas si se conocen antes de la radicación. Condiciones: aclarar por qué los
  porcentajes dividen por antigüedad + 1 (notebook 02); el torneo reporta a los mejores **sin** ellas; el piloto con grupo
  de control es la prueba definitiva.
- `account_balance_trend` y `prog_savings_trend` usan el saldo "al momento de efectuar solicitud" (diccionario del README):
  heredan la sospecha y salen en el escenario sin sospechas.
- Todas las variables nuevas se calculan fila por fila con datos del propio asociado: la app las puede calcular con lo
  que ya pide el formulario, sin columnas adicionales.

## Qué se lleva el pipeline final

27 variables (lista exacta en `reports/04_seleccion_variables.json`): las 21 originales menos las 2 redundantes, los 3
indicadores de tenencia de producto y las 5 variables nuevas conservadas. Preprocesamiento: imputación con la mediana
(con indicador de vacío), logaritmo con signo en colas largas y escalado. Código en `src/proyecto/features/engineering.py`
(`add_features`, `build_feature_pipeline`) y `src/proyecto/models/train.py` (`build_pipeline`).
