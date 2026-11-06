# Registro del modelo

> Resultado de `notebooks/08_registro_modelo.ipynb`. Constancia completa en `reports/08_registro.json`.

## 1. Dónde quedó registrado

| Dato | Valor |
|---|---|
| Plataforma | ☐ Databricks  ☒ MLflow local (respaldo) → motivo: el equipo aún no completa `.env` con el perfil o el token de Databricks. Con `.env` completo, el mismo notebook registra en Unity Catalog (`catálogo.esquema.retiro_asociados_rf`) |
| Experimento | `retiro-asociados-modelo-final` (los ensayos de la búsqueda están aparte, en `retiro-asociados-optimizacion`) |
| Nombre del modelo registrado | `retiro_asociados_rf` |
| Versión | 1 |
| Alias | `champion` → `models:/retiro_asociados_rf@champion` |
| Quién lo registró y cuándo | Grupo retiro_asociados (cuenta de Daniel Villa), 2026-11-05 21:02:15, ejecución `7bfca02731c243728892ec2e6c70223d` |

## 2. Qué se registró junto al modelo

- **Parámetros (11):** modelo Random Forest, sin calibrar; `n_estimators` 200, `max_depth` 14, `min_samples_leaf` 15,
  `max_features` 0,269, `max_samples` 0,926; regla de decisión (contactar si p × 0,2 × saldo de aportes > 0,03 millones);
  supuestos (costo del contacto 0,03 millones; efectividad 20 %); 22 columnas de entrada.
- **Métricas de la prueba (28):** todas las del notebook 07 con prefijo `test_` (PR-AUC 0,7004; AUC 0,8958; beneficio neto
  33,60 por cada 1.000 asociados; Brier 0,1014; contactos 410 por cada 1.000; saldo cubierto 94,1 %…), el beneficio de
  contactar a todos (18,79) y los intervalos del 95 %. Se leyeron de vuelta desde MLflow y coinciden con el 07.
- **Firma y ejemplo de entrada:** 22 columnas crudas, todas numéricas (decimales): `saldo_aportes`,
  `porc_meses_capitaliza_aportes`, `antig_meses`, `saldo_ahorro_prog`, `prom_saldo_ahorro_prog`, `rel_min-max_ahorro_prog`,
  `saldo_cta`, `prom_saldo_cta`, `rel_min-max_saldo_cta`, `prom_saldo_cartera`, `rel_min-max_saldo_cartera`,
  `porc_meses_publiturno`, `prom_trx`, `edad`, `femenino`, `ingresos`, `asalariado`, `ed_superior`, `zona_med`, `has_loan`,
  `has_prog_savings`, `has_account`. Las 5 variables nuevas se calculan dentro del pipeline. Salida: probabilidad de cada
  clase. Ejemplo de entrada: 5 asociados del entrenamiento.
- **Etiquetas:** `etapa=08_registro`, fecha de las decisiones congeladas, marca de la prueba abierta, unidad de partición
  (`associate_id`), `supera_referencias_con_confianza=True`; en la versión, además, `decisiones_congeladas`, `validacion` y
  una descripción.
- **Artefactos de auditoría:** `07_decisiones_congeladas.json` y `07_resultados.json`.
- **Código del proyecto** (`src/proyecto`, con `code_paths`): el pipeline usa funciones propias (variables nuevas); sin ese
  código no se podría cargar en otro equipo.

## 3. Comprobación de ida y vuelta

- ¿El modelo recuperado del registro predice igual que el local? ☒ sí (diferencia máxima 3,3 × 10⁻¹⁶).
- Cómo se comprobó: se cargó con `mlflow.sklearn.load_model("models:/retiro_asociados_rf@champion")` y se comparó
  `predict_proba` con el pipeline evaluado en el 07 sobre 1.000 asociados del entrenamiento (la prueba no se volvió a usar).

## 4. Tracking frente a registro (en nuestras palabras)

- **Tracking:** guarda cada ensayo de la optimización (20 ensayos y la ejecución madre en `retiro-asociados-optimizacion`)
  con sus parámetros, métricas y brecha de sobreajuste. Sirvió para comparar configuraciones, elegir la mejor admisible y
  poder auditar por qué se eligió.
- **Registro:** guarda el modelo que se usa, con versión. El alias `champion` evita que la app o cualquier usuario dependa de
  un número de versión: si una versión 2 (por ejemplo, con calibración isotónica) se valida con datos nuevos, se mueve el
  alias y nadie cambia su código; si falla, se devuelve.

## 5. Reproducibilidad

- ¿Alguien de otro grupo podría recuperar el modelo con lo que dejamos escrito? ☒ sí, en el mismo equipo o con acceso al
  registro: `models:/retiro_asociados_rf@champion` trae el modelo, su código, su firma y sus métricas.
- **Qué falta para que sea compartido:** el registro está en `mlflow_local.db`, que vive en un solo computador y no se sube a
  Git. Para que todo el equipo lo vea hay que completar `.env` y registrar en Databricks (volver a correr este notebook). La app
  no depende de esto: usará el `model.joblib` que el notebook 09 exporta desde `champion`.
