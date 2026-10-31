# retiro_asociados (Confiar) — Constitución

> **Borrador sembrado con las reglas del curso.** Su primer trabajo con Spec Kit es **ajustarla a su grupo**:
> ejecuten `/speckit-constitution` y pídanle, por ejemplo: *«Ajusta esta constitución a mi proyecto: nombre
> `<proyecto>`, caso `<una frase>`, métrica `<métrica>`. Conserva los principios I a VII y agrega un
> principio VIII con una regla propia del grupo»*. Todo lo que quede entre `[corchetes]` se llena.
> Claude la lee antes de planear o programar: **lo que diga aquí manda** sobre sus suposiciones.

## Principios

### I. Los datos originales no se tocan
`data/raw/` MUST NOT modificarse. Toda limpieza se hace en código (`src/<paquete>/data/`) y se guarda en
`data/processed/`, que se regenera desde cero con el notebook de preprocesamiento. *Razón:* sin el original no hay
reproducibilidad.

### II. Se parte por la unidad de análisis
La unidad de análisis de este proyecto es **el asociado (una fila por asociado)** y la entidad que se repite es
**ninguna** (`associate_id`, sustituto del `Nit`; 194.073 distintos en 194.073 filas). Aun así la partición se hace por
`associate_id`, para que siga siendo correcta si el dataset se reconstruye como panel mensual.
Si una entidad aparece en varias filas, la partición y la validación cruzada MUST hacerse por grupo
(`GroupShuffleSplit`, `GroupKFold`, `StratifiedGroupKFold`), nunca por fila. Una prueba automática MUST verificar que
ninguna entidad cae en dos particiones. *Razón:* partir por fila filtra información y infla la métrica.

### III. La prueba se abre una sola vez
El conjunto de prueba MUST NOT usarse para elegir variables, modelo, umbral ni hiperparámetros. Se abre una vez, en la
etapa de validación. Todo preprocesamiento que aprenda de los datos (medias, categorías, escalado) MUST ir dentro de un
`Pipeline` ajustado solo con entrenamiento. *Razón:* es la única medida honesta de cómo se comportará en producción.

### IV. Cada decisión de datos se escribe con su motivo
Cada decisión (qué se limpia, qué se descarta, qué se repara) MUST quedar en `reports/` con la cifra que la respalda y
su motivo, y MUST tener una función en `src/` con al menos una prueba en `tests/`. *Razón:* otra persona debe poder
discutirla sin leer el código.

### V. Nada sensible sale del equipo
`.env`, tokens y datos personales MUST NOT subirse a GitHub (el repositorio es público). Las columnas con datos personales
se eliminan al cargar, antes de cualquier análisis que las imprima. *Razón:* privacidad y seguridad de las cuentas.

### VI. El código es nuestro: lo entendemos
Cada integrante MUST poder explicar el código que generó la IA. Antes de aceptarlo se revisa con
`material-del-curso/kit/checklist-revision-ia.md`. La lógica reutilizable vive en `src/`; los notebooks solo orquestan.
Textos y comentarios en español; nombres de código en inglés. *Razón:* la autoría individual es parte de la evaluación.

### VII. Nunca se inventan datos ni resultados
Ni la IA ni el grupo MUST fabricar cifras, columnas ni métricas. Lo que no se pudo medir se dice. *Razón:* un número
inventado en el informe invalida todo lo demás.

### VIII. El modelo prioriza, no excluye (propuesta, pendiente de aprobación del grupo)
El resultado del modelo MUST usarse solo para **ordenar** a quién contactar primero en la campaña de fidelización;
MUST NOT usarse para negar productos, servicios o atención a ningún asociado. Los supuestos económicos (costo del
contacto, efectividad de la retención) MUST mostrarse junto a toda cifra en pesos, y las variables sociodemográficas
(sexo, edad, zona, educación, ocupación) MUST revisarse por error por segmento antes de desplegar. Ninguna variable que
se mida en el mes de la radicación o después MUST entrar al modelo sin haberse medido con y sin ella.
*Razón:* el modelo afecta la relación con los asociados de una cooperativa y describe asociaciones, no causas.

## Restricciones del proyecto

- **Caso:** ¿qué asociados de Confiar tienen mayor probabilidad de radicar la solicitud de retiro total, para que el
  equipo de fidelización los contacte primero y conserve la mayor cantidad de aportes? · **Métrica principal:** beneficio
  neto por cada 1.000 asociados con la regla de valor esperado (traduce los dos errores a dinero; control técnico:
  PR-AUC) · **Tipo de problema:** clasificación binaria.
- **Entorno:** Python 3.12; dependencias fijadas en `requirements.txt`.
- **Seguimiento de experimentos:** MLflow en Databricks (o local si no hay conexión); la app **no** consulta Databricks.
- **App:** Streamlit, cargando `models/model.joblib` desde el repositorio.

## Flujo de trabajo

- Cada etapa sigue **especificación → plan → tareas → implementación** (`/speckit-specify`, `/speckit-plan`,
  `/speckit-tasks`, `/speckit-implement`). Los artefactos viven en `specs/`.
- Las guías de etapa de `especificaciones/` se llenan **antes** de pedir una especificación a Spec Kit: son su entrada.
- Los **subagentes** de `.claude/agents/` se usan para revisar (fuga, pruebas, código de la IA) antes de dar una etapa por
  terminada.
- Cada hito MUST tener criterios de aceptación verificables (`especificaciones/`, sección de cierre).

## Gobierno

Esta constitución prevalece sobre otras prácticas del proyecto. Una enmienda se hace con `/speckit-constitution`, se
justifica y se anota en la sección 7 de `especificacion-proyecto.md`. El instructor puede pedir aclaraciones sobre
cualquier principio en los puntos de control.

**Versión**: 0.2.0 (borrador ajustado al caso) | **Ratificada**: pendiente (el grupo la revisa con `/speckit-constitution`) | **Última enmienda**: 2026-10-31
