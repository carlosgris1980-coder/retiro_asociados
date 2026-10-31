# Especificación · Etapa 3 · Preprocesamiento (limpieza y partición)

> Se llena **antes** de pedirle código a Claude, a partir de la tabla de decisiones de
> `especificaciones/02-eda.md`. Claude la lee y la implementa; ustedes revisan que el código haga
> **exactamente** esto (`material-del-curso/kit/checklist-revision-ia.md`).
> Guía: `material-del-curso/kit/guia-eda-y-preprocesamiento.md`, sección 4 · Ejemplo: `material-del-curso/ejemplo-creditsight/src/creditsight/data/`.

**Grupo:** retiro_asociados (Confiar)   **Fecha:** 2026-10-31   **Estado:** ☒ borrador  ☐ cerrada (falta revisión del grupo)

## 1. Unidad de análisis y partición

- **Una fila es:** un asociado.
- **Entidad que se repite (columna):** ninguna  ☒ no se repite → partir por fila es válido porque: hay 194.073 `Nit`
  distintos en 194.073 filas. Aun así se parte **por grupo** (`associate_id`) para que el código siga siendo correcto si
  el dataset se reconstruye como panel.
- **Se parte por:** ☒ grupo (`StratifiedGroupKFold`, un pliegue de 5) ☐ fila — estratificado por `stratum`
  (radica / activo / salió sin radicar).
- **Proporción de prueba:** 20 %   **Semilla:** 42
- **¿Hay orden temporal que respetar?** ☐ no ☒ sí → las fechas del registro no entran al modelo (delatan la respuesta).
  Una validación fuera de tiempo no es posible con este diseño (todos los activos están en el mismo mes); queda como
  recomendación al reconstruir el dataset con horizonte.
- **Dónde se guarda la prueba:** `data/processed/test.joblib` (cerrada: **no se mira** hasta la etapa 7)

## 2. Datos personales

| Columna | Qué es | Se elimina en | Prueba que lo verifica |
|---|---|---|---|
| `Nit` | documento del asociado | `load_raw()` → `drop_personal_data()` | `test_drop_personal_data_removes_nit_and_adds_surrogate_id`, `test_clean_keeps_rows_and_leaves_no_pii_or_dates` |

## 3. Reglas de limpieza

| # (de la tabla del EDA) | Columna(s) | Problema | Regla exacta (con el umbral o valor) | Función en `src/proyecto/data/cleaning.py` | Prueba en `tests/test_cleaning.py` |
|---|---|---|---|---|---|
| 3 | `anio_mes_proceso`, `cohorte` | Delatan la respuesta | Se eliminan; antes se guardan `process_year` y `stratum` como auxiliares | `add_time_context`, `drop_time_columns` | `test_time_context_marks_exit_without_request_and_stratum`, `test_drop_time_columns_removes_both_dates` |
| 5 | `sol_retiro` + fecha | El 0 mezcla activos y retiros unilaterales de la cooperativa (confirmado) | `exit_without_request` = 1 si `sol_retiro` = 0 y el registro es anterior al último mes del panel | `add_time_context` | `test_time_context_marks_exit_without_request_and_stratum` |
| 6 | `edad` | 39 valores imposibles | Fuera de [14, 100] → NaN; la fila se conserva | `mask_implausible` | `test_mask_implausible_age_keeps_rows_and_sets_missing` |
| 7 | `prom_saldo_cartera`, `prom_saldo_ahorro_prog`, `prom_saldo_cta` | Cero = sin producto | `has_loan`, `has_prog_savings`, `has_account` = 1 si el promedio > 0 | `add_product_flags` | `test_product_flags_read_zero_balance_as_no_product` |
| 9 | `ingresos` | Salario mínimo exacto del año | `income_is_min_wage` = 1 si el ingreso es igual (±1e-6) al SMMLV de algún año 2017–2025 | `flag_min_wage_income` | `test_min_wage_income_flag_matches_only_exact_values` |
| — | todas | La limpieza no pierde filas ni modifica la entrada | `clean()` aplica todo en orden y verifica el número de filas | `clean` | `test_clean_keeps_rows_and_leaves_no_pii_or_dates` |

## 4. Vacíos

| Columna | Qué significa el vacío | Tratamiento hoy | Tratamiento en la etapa 4 (dentro del `Pipeline`) |
|---|---|---|---|
| `edad` (39 filas) | ☐ no aplica ☐ no se anotó ☐ perdido ☒ imposible | ☒ dejar vacío ☐ categoría explícita ☐ reparar con valor de la entidad | Mediana del entrenamiento (`SimpleImputer`) |
| Saldos en cero | No es vacío: "no tiene el producto" | Se conserva el cero + indicador de tenencia | Sin imputación |

> Imputar con medias o medianas **no se hace aquí**: depende del entrenamiento y va dentro del `Pipeline`.

## 5. Entradas y salidas

| | Ruta |
|---|---|
| Entrada | `data/raw/bd_confiar_2.xlsx` (nunca se modifica) |
| Salida: entrenamiento | `data/processed/train.joblib` |
| Salida: prueba (cerrada) | `data/processed/test.joblib` |
| Código de limpieza | `src/proyecto/data/cleaning.py` |
| Código de partición | `src/proyecto/data/split.py` |
| Configuración (reglas, columnas, supuestos) | `src/proyecto/config.py` |
| Notebook que orquesta | `notebooks/03_preprocesamiento.ipynb` (solo llama a las funciones) |

## 6. Pruebas que deben existir

- ☒ **Partición:** ninguna entidad aparece en dos particiones y no se pierde ninguna fila (`tests/test_split.py`).
- ☒ **La prueba de partición falla** si se cambia temporalmente a partición por fila (comprobado con `train_test_split`
  sobre datos con asociados repetidos).
- ☒ **Una prueba por regla de limpieza** de la sección 3, con un caso normal y un caso sucio hechos a mano.
- ☒ **Datos personales:** el resultado de cargar no contiene el `Nit`.
- ☒ `pytest` en verde desde la raíz del repo.

## 7. Criterios de cierre (H1)

- ☒ La limpieza corre **de principio a fin desde cero** (`data/processed/` se regenera con el notebook 03), sin pasos manuales.
- ☒ Unidad de análisis y forma de partir **escritas** (sección 1) y documentadas en `reports/01_eda.md`.
- ☒ `pytest` en verde.
- ☒ Conjunto de prueba guardado y sin mirar.
- ☒ `data/raw/` intacta y `.env` fuera de Git (`git status`).
- ☐ Todo subido a GitHub.
- ☐ Revisaron el código de Claude con `checklist-revision-ia.md` y cada integrante puede explicar una función.
- ☒ **Hallazgo que no esperaban:** el 0 del objetivo mezcla a los activos con 23.688 asociados que salieron sin radicar,
  y las fechas del registro (con la antigüedad) delatan la respuesta.
