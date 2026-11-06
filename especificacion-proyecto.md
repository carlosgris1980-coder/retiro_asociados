# Especificación del proyecto

> Claude la lee antes de generar código: lo que está escrito aquí manda sobre cualquier otra suposición.
> Lo marcado como **por confirmar** debe validarlo el grupo con la fuente de los datos.

## 1. Grupo

| Integrante | Rol (datos / modelado / app / negocio) |
|---|---|
| Carlos Galvis | por definir |
| Jaqueline Acevedo | por definir |
| Daniel Villa | por definir |
| | |
| | |

Nombre de la "startup": _por definir con el equipo_   Repositorio (público): https://github.com/carlosgris1980-coder/retiro_asociados

## 2. Caso de negocio

- **Pregunta de negocio** (una frase, que alguien sin formación técnica entienda):
  ¿Qué asociados de Confiar tienen mayor probabilidad de radicar la solicitud de retiro total de aportes, para que el
  equipo de fidelización los contacte primero y conserve la mayor cantidad posible de aportes con el presupuesto de la
  campaña?
- **¿Quién usará el resultado y qué decisión tomará con él?**
  El **área comercial y de mercadeo** (equipo de fidelización). Cada mes decide a qué asociados incluir en una campaña
  proactiva de retención (llamada del asesor, mensaje, beneficio) **antes** de que radiquen. Hoy el protocolo es
  reactivo: se activa cuando el asociado ya está radicando. El modelo ordena la lista; el asesor decide.
- **¿Qué cuesta equivocarse?** (cuál error es peor y por qué):
  - **Falso negativo** (iba a radicar y no se contacta): se pierde la oportunidad de conservar sus aportes. Con los
    supuestos de referencia cuesta **0,203 millones** por asociado (20 % × 1,166 millones de saldo promedio − 0,03 del
    contacto). **Es el error caro.**
  - **Falso positivo** (no iba a radicar y se contacta): se paga el contacto, **0,03 millones**, sin recuperar nada.
  - Proporción: **6,8 a 1**. Fuente: `notebooks/01_caso_negocio.ipynb`, sección 3.

## 3. Datos

- **Fuente:** datos internos de Confiar (cooperativa financiera), anonimizados por dos integrantes del grupo que trabajan
  en la cooperativa. Archivo `bd_confiar_2.xlsx` (copia de trabajo en `data/raw/`, fuera de Git).
- **Filas / columnas:** 194.073 / 25 (21 variables, el objetivo, el `Nit` y dos fechas).
- **Variable objetivo:** `sol_retiro` (1 = radicó solicitud de retiro total; 0 = no radicó). Tasa: 20,9 %.
- **Tipo de problema:** ☒ clasificación  ☐ regresión  ☐ clusterización
- **¿Una misma entidad (cliente, paciente, empresa) aparece en varias filas?** ☒ no  ☐ sí → una fila por asociado
  (194.073 `Nit` distintos). La partición se hace igual **por asociado** (`split_by_group`), estratificada por grupo.
- **Datos personales o confidenciales presentes:** ☐ no  ☒ sí → el `Nit` se elimina al cargar (`load_raw`) y se
  reemplaza por `associate_id`. Edad, sexo, zona, educación y ocupación se conservan como variables (sin identificar a
  nadie) y se revisan por equidad en la etapa 7.
- **Licencia o permiso de uso de los datos:** a cargo del grupo (autorización de la cooperativa para uso académico y
  decisión sobre el `bd_confiar_2.xlsx` que está en la raíz del repositorio público).
- **Unidad monetaria:** millones de pesos (confirmado por el equipo, 31-oct-2026).

## 4. Métrica

- **Métrica principal:** **beneficio neto por cada 1.000 asociados** (millones de COP) con la regla de decisión
  *contactar si probabilidad × efectividad × saldo de aportes > costo del contacto*.
  **Por qué esa y no otra:** traduce los dos errores a dinero y premia encontrar a quien se va **con más aportes**, que es
  lo que la cooperativa pierde.
- **Métricas secundarias:** saldo de aportes en riesgo cubierto (%), contactos por 1.000 asociados; técnicas: **PR-AUC**
  (control principal), ROC-AUC, recall y precisión de "radica" con su umbral, lift en el 10 % y 20 % superior, Brier y
  curva de calibración. **No se usa exactitud.**
- **Línea base:** "contactar a todos" = **18,76 millones** de beneficio neto por cada 1.000 asociados (con los supuestos);
  "sin campaña" = 0; techo perfecto = 42,68. Exactitud de "nadie radica" = 79,1 %. PR-AUC al azar ≈ 0,21.
- **Supuestos (por confirmar):** costo por contacto $30.000 y efectividad de la retención 20 % (`src/proyecto/config.py`).
  Se reemplazan con el costo real de mercadeo y un piloto con grupo de control.

## 5. Riesgos

- **Fuga de información posible:**
  - `anio_mes_proceso` y `cohorte`: el mes del registro delata la respuesta (0,6 % radica si es dic-2025; 62,7 % si no).
    **Eliminadas.**
  - Sospechas a medir con y sin en la etapa 4: `porc_meses_publiturno`, `prom_asesorias_publiturno` y
    `rel_min-max_saldo_cartera` (pueden reflejar la preparación del retiro; el equipo confirmó que no incluyen el mes de
    radicación, pero el cálculo divide por antigüedad + 1 y queda por aclarar); `ingresos` (pesos nominales,
    23 % = salario mínimo exacto del año de registro).
- **Sesgos o equidad:** sexo, edad, zona, educación y ocupación separan poco (AUC univariado ≤ 0,521). Se revisará el
  error por segmento; el modelo no se usa para dejar de atender a ningún grupo.
- **Datos faltantes o de mala calidad:** sin vacíos de origen (ya tratados por el equipo); 39 edades imposibles → vacío;
  topes de origen en 5 variables; 23.688 asociados cuyo vínculo terminó la cooperativa de forma unilateral, mezclados en
  el 0 (no son retiros voluntarios; se medirá el modelo con y sin ellos).
- **Límites de lo que el modelo puede decir:** asociación, no causalidad; el dataset dice *si* radicó, no *cuándo*
  (falta horizonte de predicción); los activos se observan en dic-2025 y quienes radican en su propio mes.

## 6. Entregables y fechas

| Hito | Contenido | Fecha |
|---|---|---|
| H1 | Ficha aprobada, EDA y preprocesamiento reproducible | Cierre de la clase 2 (vie 30 oct) |
| H2 | Torneo, optimización con tracking, modelo registrado, repo modular, app local | Cierre de la clase 4 (jue 5 nov) |
| H3 | App desplegada y presentación (≤ 5 diapositivas) | Clase 5 (vie 6 nov) |
| H4 | Informe técnico final (hasta 8 días tras la última clase) | Domingo 15 nov |

## 7. Decisiones tomadas (agréguenlas a medida que avancen)

| Fecha | Decisión | Motivo |
|---|---|---|
| 2026-10-31 | Métrica principal = beneficio neto por 1.000 asociados con regla de valor esperado | Los dos errores no cuestan lo mismo y los retiros no pesan igual (20 % de quienes radican concentra el 61 % de los aportes) |
| 2026-10-31 | Se eliminan `anio_mes_proceso` y `cohorte` | Delatan la respuesta; demostrado en el notebook 03 |
| 2026-10-31 | Se conservan las 306 filas gemelas (el notebook del equipo las eliminaba) | Son asociados distintos (Nit distintos); no hay fuga |
| 2026-10-31 | Edad fuera de 14–100 → vacío, imputada en el Pipeline | 39 valores imposibles; no se borran filas |
| 2026-10-31 | Partición 80/20 por asociado, estratificada por radica / activo / salió sin radicar, semilla 42 | Unidad de análisis; misma mezcla en ambos lados |
| 2026-10-31 | Detalle completo de decisiones del EDA | `reports/01_eda.md` |
| 2026-10-31 | Confirmado con el equipo: salidas sin radicar = retiro unilateral; montos en millones de pesos; variables sin el mes de radicación | Respuestas de Carlos y Jaqueline; queda por aclarar por qué los porcentajes dividen por antigüedad + 1 |
| 2026-10-31 | 27 variables: se agregan 5 nuevas, se quitan 2 redundantes, se conservan las sospechas (con condiciones) y los retiros unilaterales en el entrenamiento | Mejora / ruido medido en `reports/04_features.md` |
| 2026-10-31 | Finalistas del torneo: Random Forest y Gradient Boosting; la Regresión Logística queda como referencia interpretable | `reports/05_torneo.md`: empatan en beneficio neto; la logística cuesta 1,46 millones por 1.000 asociados |
| 2026-11-05 | Decisiones congeladas antes de abrir la prueba: Random Forest optimizado, sin calibrar (isotónica no llegó al umbral de 0,005 de log loss), regla de valor esperado. Prueba abierta una sola vez: beneficio 33,60 por 1.000 (+14,81 sobre contactar a todos), PR-AUC 0,700 | `reports/07_validacion.md` |
| 2026-11-05 | Se optimiza Random Forest maximizando PR-AUC con brecha de sobreajuste ≤ 0,10; se adoptan los parámetros optimizados (profundidad 14, hojas ≥ 15, 27 % de variables, 93 % de asociados, 200 árboles) | `reports/06_optimizacion.md`: igual PR-AUC y beneficio en asociados reservados, brecha de 0,30 a 0,07; `class_weight` descartado con evidencia |
