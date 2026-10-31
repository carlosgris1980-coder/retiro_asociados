# Especificación · Etapa 2 · EDA

> Se llena **antes** de abrir `notebooks/02_eda.ipynb` (las secciones 1 y 2) y se completa **mientras**
> lo ejecutan (3 y 4). El resultado final, con las decisiones, va a `reports/01_eda.md`.
> Guía: `material-del-curso/kit/guia-eda-y-preprocesamiento.md` · Ejemplo: `material-del-curso/ejemplo-creditsight/reports/01_eda.md`.

**Grupo:** retiro_asociados (Confiar)   **Fecha:** 2026-10-31   **Estado:** ☒ borrador  ☐ cerrada (falta revisión del grupo)

## 1. Qué esperamos encontrar (se escribe ANTES de mirar los datos)

| Pregunta | Lo que creemos |
|---|---|
| ¿Qué representa una fila? | Un asociado (registro de su último mes o del mes en que radicó, según el README) |
| ¿Alguna entidad se repite en varias filas? | No: el equipo construyó un registro por asociado |
| ¿Qué variable **debería** ser estable y no cambiar? | Sexo, educación, zona; el ingreso debería ser comparable entre asociados sin importar el año del registro |
| ¿Qué significaría un vacío en nuestras columnas clave? | El equipo ya imputó los nulos: un vacío no debería existir; un **cero** en un saldo significaría "no tiene el producto" |
| ¿Qué columna solo se conoce **después** de la decisión que queremos apoyar? | El mes del registro (`anio_mes_proceso`) y todo lo que se mida en el mes de la radicación |

## 2. Qué vamos a medir

| Chequeo (guía, sección 3) | ¿Aplica? | Columnas a revisar | Quién lo hace |
|---|---|---|---|
| 3.1 Números guardados como texto | ☐ sí ☒ no: las 21 variables llegan numéricas; solo las fechas son texto (`AAAA-MM`) | — | — |
| 3.2 Valores centinela (`-1`, `9999`, `?`) | ☒ sí ☐ no: | `edad` (125), `ingresos` (valores repetidos) | Claude + grupo |
| 3.3 Valores imposibles (percentiles, negativos) | ☒ sí ☐ no: | todas las continuas; `edad`; `prom_saldo_capitalizacion_aportes` | Claude + grupo |
| 3.4 Constancia por entidad | ☐ sí ☒ no: una fila por asociado. Se traslada al **tiempo**: ¿qué depende del año del registro? | `ingresos` | Claude + grupo |
| 3.5 Vacíos que significan algo | ☒ sí ☐ no: no hay vacíos, pero los **ceros** significan "sin producto" | saldos y razones mín./máx. | Claude + grupo |
| 3.6 Duplicados | ☒ sí ☐ no: | todas sin el `Nit` | Claude + grupo |
| 3.7 Fechas y orden temporal | ☒ sí ☐ no: | `anio_mes_proceso`, `cohorte`, `antig_meses` | Claude + grupo |
| 3.8 Categorías raras / alta cardinalidad | ☐ sí ☒ no: no hay categóricas; las binarias tienen 0/1 | — | — |
| 3.9 Identidad y datos personales | ☒ sí ☐ no: | `Nit` | Claude + grupo |

## 3. Lo que medimos (se llena mientras corre el notebook)

- **Filas × columnas:** 194.073 × 25
- **Unidad de análisis (una frase):** cada fila es un asociado, fotografiado en el mes en que radicó o en el último mes observado.
- **Columna que identifica a la entidad (`ID_COL`):** `Nit` → reemplazado por `associate_id`   **¿Se repite?** ☒ no ☐ sí
- **Objetivo (`TARGET`):** `sol_retiro`   **Distribución:** 1 = 20,92 % (40.591); 0 = 79,08 % (129.794 activos + 23.688 que salieron sin radicar)
- **Línea base** (regla trivial) y su resultado: "siempre no radica" = 79,1 % de exactitud y 0 % de recall; "contactar a todos" = 18,76 millones de beneficio neto por 1.000 asociados (notebook 01)
- **Vacíos en el objetivo:** 0   **Qué haremos con esas filas:** no aplica
- **Datos personales encontrados:** `Nit` (eliminado al cargar)
- **Columnas sospechosas de fuga:** `anio_mes_proceso` y `cohorte` (confirmadas, se eliminan); `porc_meses_publiturno`,
  `prom_asesorias_publiturno`, `rel_min-max_saldo_cartera` e `ingresos` (se miden con y sin ellas en la etapa 4)

## 4. Decisiones (mínimo 5)

| # | Hallazgo (con cifra) | Decisión | Motivo | Función que la implementa |
|---|---|---|---|---|
| 1 | `Nit` único en las 194.073 filas | Eliminar al cargar; `associate_id` sustituto | Dato personal | `load_raw` / `drop_personal_data` |
| 2 | Ninguna entidad se repite | Partición por asociado, estratificada, 80/20 | Unidad de análisis | `split_by_group` |
| 3 | Fuera de dic-2025 radica el 62,7 %; en dic-2025, el 0,6 % | Eliminar `anio_mes_proceso` y `cohorte` | Delatan la respuesta | `drop_time_columns` |
| 4 | 23.688 salieron sin radicar (12,2 %) | Conservar como 0; marca auxiliar | Medir con y sin en la etapa 4 | `add_time_context` |
| 5 | 39 edades fuera de 14–100 | → vacío | Error o centinela | `mask_implausible` |
| 6 | Ceros = sin producto (51,8 % sin crédito) | Indicadores de tenencia | Distinguir "no tiene" de "poco" | `add_product_flags` |
| 7 | 23,0 % de ingresos = salario mínimo exacto | Marca auxiliar; sospecha a medir | Pesos nominales ligados al año | `flag_min_wage_income` |

Tabla completa (12 decisiones) en `reports/01_eda.md`.

## 5. Criterios de cierre

- ☒ `notebooks/02_eda.ipynb` corre de principio a fin sin errores.
- ☒ Unidad de análisis y `ID_COL` declarados por escrito (sección 3).
- ☒ Al menos **5 decisiones** con cifra, acción y motivo.
- ☒ Datos personales identificados y plan para eliminarlos al cargar.
- ☒ Valores imposibles, centinelas y significado de los vacíos revisados.
- ☒ Fuga sospechada: lista escrita.
- ☒ `reports/01_eda.md` completado.
- ☐ Quien **no** escribió una decisión sabe explicar por qué se tomó. *(pendiente: revisión del grupo)*
