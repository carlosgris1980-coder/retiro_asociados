# Ficha de caso

> Resumen de `notebooks/01_caso_negocio.ipynb`. Cada cifra sale de ese notebook o del 02.

- **case_id:** `retiro-asociados-confiar`

- **business_question:** En Confiar, cooperativa financiera, el **área comercial y de mercadeo** (equipo de
  fidelización) decide cada mes **a qué asociados incluir en una campaña proactiva de retención** antes de que radiquen
  la solicitud de retiro total de aportes. Hoy el protocolo es reactivo: solo se activa cuando el asociado ya está
  radicando. Queremos un modelo que estime, con el perfil de productos, saldos, transacciones, visitas a sucursal y datos
  sociodemográficos, la **probabilidad de que cada asociado radique**, para ordenar la lista de contacto y conservar la
  mayor cantidad de aportes con el presupuesto de la campaña. El asesor conserva la decisión final.

- **dataset_source:** datos internos de Confiar, anonimizados por integrantes del grupo que trabajan en la cooperativa
  (`bd_confiar_2.xlsx`; asociados vinculados entre 2017 y 2025, un registro por asociado). Montos en millones de pesos.
  Permiso de uso académico: a cargo del grupo.

- **dataset_rows:** 194.073

- **dataset_columns:** 25 (21 variables, el objetivo `sol_retiro`, el `Nit` y dos fechas)

- **target:** `sol_retiro` (1 = radicó solicitud de retiro total). 40.591 asociados radicaron (**20,9 %**). El 0 mezcla
  a 129.794 asociados activos en dic-2025 y a 23.688 cuyo vínculo terminó la cooperativa de forma unilateral (no son
  retiros voluntarios; se medirá el modelo con y sin ellos). Unidad de análisis: **el asociado** (una
  fila cada uno); la partición se hace por asociado.

- **problem_type:** clasificación binaria (desbalance moderado), usada como **ranking por probabilidad**.

- **primary_metric:** **beneficio neto por cada 1.000 asociados** (millones de COP) con la regla *contactar si
  probabilidad × efectividad × saldo de aportes > costo del contacto*. Debe superar a "contactar a todos"
  (**18,76 millones** por 1.000 asociados con los supuestos); el techo teórico es 42,68. Control técnico principal:
  **PR-AUC** (al azar ≈ 0,21), más ROC-AUC, recall y precisión con su umbral, lift y calibración. No se usa exactitud:
  responder siempre "no radica" acierta el 79,1 %.

- **business_cost_note:**
  - Los retiros pesan mucho: en 2025 radicaron 9.422 asociados y se llevaron **16.121 millones de pesos** en aportes. El
    20 % de quienes radican con más saldo concentra el **61 %** de los aportes perdidos.
  - **Falso negativo** (no contactar a quien radica): se deja de conservar, en promedio, 20 % × 1,166 millones − 0,03 =
    **0,203 millones** por asociado. Es el error caro.
  - **Falso positivo** (contactar a quien no iba a radicar): **0,03 millones** (el contacto). Proporción **6,8 a 1**.
  - Supuestos de referencia, **por confirmar**: costo por contacto $30.000 y efectividad de retención 20 %. Con ellos,
    una campaña masiva deja de ser rentable por encima de **$48.760** por contacto.

- **status:** borrador

- **reviewer_notes:** <!-- las llena el instructor -->
