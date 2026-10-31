# Informe del EDA — Retiro de asociados (Confiar)

> Resultado de `notebooks/02_eda.ipynb`. Cada hallazgo termina en una decisión, y cada decisión de limpieza está
> implementada con pruebas en `src/proyecto/data/cleaning.py` (`tests/test_cleaning.py`).

## Qué son los datos

- **Filas × columnas:** 194.073 × 25 (24 sin el `Nit`).
- **Unidad de análisis:** **el asociado**. Hay 194.073 `Nit` distintos en 194.073 filas: nadie se repite. Partir por
  asociado equivale a partir por fila; se usa `split_by_group` por asociado, estratificado.
- **Cómo se construyó:** quien radica se registra en el mes en que radicó (2017–2025); quien no radica, en el último mes
  observado. Los 129.794 activos están todos en **diciembre de 2025**. `antig_meses` = meses entre `cohorte` y
  `anio_mes_proceso` en el 100 % de las filas.
- **Objetivo y distribución:** `sol_retiro` = 1 en el **20,92 %** (40.591). El 0 mezcla a 129.794 activos (66,9 %) y a
  **23.688 que salieron sin radicar (12,2 %)**: vínculos que la cooperativa terminó de forma unilateral (confirmado por
  el equipo), con un perfil propio: 17 meses de antigüedad mediana y un 32,4 % sin aportes.
- **Línea base:** responder siempre "no radica" acierta el 79,1 % y no encuentra a nadie. En negocio, la línea base es
  "contactar a todos" (notebook 01).

## Calidad de los datos

- **Sin vacíos y todo numérico**: el equipo ya trató nulos y atípicos en el origen.
- **Topes de origen** (filas en el máximo): `saldo_ahorro_prog` = 30 (1.260), `prom_saldo_ahorro_prog` = 27,9 (1.165),
  `prom_saldo_cta` = 10 (831), `ingresos` = 22 (476), `saldo_cta` = 35 (161).
- **Colas largas reales:** `prom_saldo_cartera` p99,9 = 114 y máximo 1.333; `prom_trx` p99,9 = 78 y máximo 531.
- **39 edades imposibles:** 24 entre 3 y 13 años y 15 por encima de 100 (10 con exactamente 125, valor centinela).
  Entre 15 y 17 años hay 263 asociados reales.
- **421 capitalizaciones promedio negativas (0,22 %)**: compatibles con retiros parciales de aportes.
- **306 filas gemelas (163 copias de más)**: todas de asociados distintos, ninguno radica, aportes promedio 0,002.
- **Ceros = "no tiene el producto"**: sin crédito 51,8 %, sin ahorro programado 49,9 %, sin cuenta 42,5 %. Las razones
  mínimo/máximo son ambiguas en cero (el 59,9 % de los `rel_min-max_saldo_cartera = 0` nunca tuvo crédito).
- **El ingreso depende del calendario:** 44.606 ingresos (23,0 %) son exactamente el salario mínimo de un año entre 2017
  y 2025, y el ingreso mediano sube con el año del registro (de 0,85 en 2017 a 1,6 en 2025). Coincide con que los montos
  están en **millones de pesos** (confirmado por el equipo).
- **Redundancias:** visitas y asesorías en sucursal ρ = 0,94; aportes y capitalización ρ = 0,84.

## Señal

- AUC univariado más alto: `porc_meses_publiturno` 0,679, `prom_asesorias_publiturno` 0,658,
  `porc_meses_capitaliza_aportes` 0,654. Ocho variables por debajo de 0,53; las sociodemográficas ≤ 0,521.
- La radicación se concentra entre los **7 y 24 meses** de antigüedad (30,5 % y 30,3 %) y cae a 6,1 % después de 85
  meses. Las cohortes recientes radican menos porque no han tenido tiempo (censura).
- El asociado "dormido" (sin aportes, sin movimiento, sin visitas) casi no radica; el riesgo está en el asociado activo
  y transaccional.

## Sospechas de fuga de información

- **Fechas (confirmada):** registro en dic-2025 → 0,6 % radica; en cualquier otro mes → 62,7 %. `cohorte + antig_meses`
  reconstruye el mes del registro. Demostración en el notebook 03: con el mismo modelo y validación cruzada de 5
  pliegues sobre el entrenamiento, agregar la fecha de vinculación sube el PR-AUC de 0,716 a 0,906 y el AUC de 0,900 a
  0,977. Es una mejora falsa: el modelo lee cuándo se tomó la foto.
- **Ventana de cálculo (por aclarar):** el equipo confirmó que las variables **no incluyen el mes de radicación**, pero
  `porc_meses_publiturno × (antig_meses + 1)` es entero en el 96,9 % de las filas (el conteo cubre antigüedad + 1
  meses). Verificar en la consulta que generó el dataset qué meses cubre la ventana.
- **Cartera:** entre quienes tuvieron crédito, el 97,3 % de quien radica tiene el mínimo de cartera en cero (activos:
  68,9 %). Puede reflejar que saldó el crédito para prepararse a retirarse.
- **Visitas:** solo el 1,8 % de quien radica nunca visitó una sucursal (activos: 8,5 %). Puede ser señal real o visitas
  para preparar el retiro.
- **Ingreso:** el salario mínimo de 2025 solo aparece en registros de 2025; esos asociados radican 5,2 % frente a 21,5 %
  del resto.

## Decisiones

| # | Hallazgo (con cifra) | Decisión | Motivo |
|---|---|---|---|
| 1 | `Nit` = documento; 194.073 distintos en 194.073 filas | Se elimina al cargar y se reemplaza por `associate_id` | Dato personal; un identificador único no aporta señal |
| 2 | Una fila = un asociado; ninguno se repite | Partición por asociado, estratificada por radica / activo / salió, 80/20, semilla 42 | Unidad de análisis; misma mezcla en ambos lados |
| 3 | Fuera de dic-2025 radica el 62,7 %; en dic-2025, el 0,6 %; `cohorte + antig` = mes del registro (100 %) | Se eliminan `anio_mes_proceso` y `cohorte` | Delatan la respuesta |
| 4 | Tasa 20,9 %; "siempre no" acierta 79,1 % | Métrica: beneficio neto + PR-AUC; nunca exactitud | La exactitud premia al modelo que no encuentra a nadie |
| 5 | 23.688 (12,2 %) salieron sin radicar: retiro unilateral de la cooperativa (confirmado) | Se conservan como 0; marca auxiliar `exit_without_request` | Fuera de la pregunta (no es retiro voluntario); medir con y sin ellos en la etapa 4 |
| 6 | 39 edades imposibles (< 14 o > 100; 10 con 125) | Fuera de 14–100 → vacío; imputación en el Pipeline | Error o centinela; no se borran filas |
| 7 | Saldo promedio en cero = sin producto (51,8 % sin crédito) | Indicadores `has_loan`, `has_prog_savings`, `has_account` | Distinguir "no tiene" de "tiene poco" |
| 8 | Con crédito, 97,3 % de quien radica con cartera mínima cero (activos 68,9 %); ventana por aclarar (96,9 % divide por antigüedad + 1) | Sospechas: medir con y sin en la etapa 4 | Pueden reflejar la preparación del retiro |
| 9 | 23,0 % de ingresos = salario mínimo exacto; ingreso sube con el año | Se conserva; marca auxiliar `income_is_min_wage`; sospecha a medir | Pesos nominales: lleva parte de la fecha |
| 10 | Topes de origen en 5 variables; colas largas (cartera 1.333, transacciones 531) | No se recortan; log en el Pipeline para modelos sensibles a escala | Ya tratados; valores reales |
| 11 | 421 capitalizaciones negativas; 306 filas gemelas de asociados distintos | Se conservan | Retiros parciales reales; personas distintas |
| 12 | Visitas–asesorías ρ = 0,94; aportes–capitalización ρ = 0,84 | Probar una de cada pareja en la etapa 4 | Parsimonia y estabilidad de coeficientes y SHAP |

## Qué queda abierto

- **Confirmado por el equipo (31-oct-2026):** los 23.688 que salieron sin radicar son retiros unilaterales de la
  cooperativa; los montos están en millones de pesos; las variables no incluyen el mes de radicación; el permiso de uso
  de los datos y el Excel en el repositorio público quedan a cargo del grupo.
- **Por aclarar:** por qué los porcentajes de meses dividen por antigüedad + 1 (revisar la consulta de origen).
- **Horizonte de predicción:** el dataset dice *si* el asociado radicó, no *cuándo*. Con el panel mensual se puede
  reconstruir como "variables a la fecha t, ¿radica en los próximos k meses?". Es la mejora de mayor valor.
- **Etapa 4:** medir con y sin las sospechas de fuga, con y sin quienes salieron sin radicar, una variable por pareja
  redundante y la transformación logarítmica.
