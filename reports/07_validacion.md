# Validación e interpretación

> Resultado de `notebooks/07_validacion_interpretacion.ipynb`. Cifras de `reports/07_resultados.json`,
> `07_errores_por_segmento.csv` y `07_importancia_shap.csv`. Código: `src/proyecto/models/validation.py`.

## 1. Decisiones congeladas (escritas ANTES de abrir la prueba)

| Decisión | Valor | Cómo se decidió (con qué datos) |
|---|---|---|
| Modelo final | Random Forest | Primero del torneo (notebook 05, validación cruzada por asociado) |
| Parámetros (optimizados o de partida) | Optimizados: 200 árboles, `max_depth` 14, `min_samples_leaf` 15, `max_features` 0,269, `max_samples` 0,926 | Notebook 06: igual PR-AUC y beneficio que los de partida en 38.815 asociados reservados, con la brecha de sobreajuste de 0,30 a 0,07 (regla escrita antes) |
| ¿Calibrar las probabilidades? | No | Asociados reservados (sección 2); la regla pedía mejorar el log loss en más de 0,005 |
| Métrica principal | Beneficio neto por cada 1.000 asociados | Notebook 01; controles: PR-AUC, AUC, Brier |
| Regla de decisión | Contactar si probabilidad × 20 % × saldo de aportes > $30.000 | Notebook 01 (regla de valor esperado); supuestos de costo y efectividad por validar |
| **Fecha y hora en que se congelaron** | **2026-11-05 20:35:20** | `reports/07_decisiones_congeladas.json` |

## 2. Calibración (con entidades reservadas, no con la prueba)

- ¿Se evaluó? ☒ sí, con los mismos 38.815 asociados reservados del notebook 06 (modelo ajustado con los 116.443 de búsqueda;
  calibración con 3 pliegues por asociado).
- Resultado y decisión:

| Variante | Log loss | Brier | Beneficio neto por 1.000 | Diferencia de beneficio vs sin calibrar (IC 95 %) |
|---|---|---|---|---|
| Sin calibrar | 0,3243 | 0,1022 | 33,44 | — |
| Sigmoide | 0,3292 | 0,1022 | 33,37 | −0,067 [−0,244 ; +0,114] |
| Isotónica | 0,3205 | 0,1012 | 33,66 | +0,226 [+0,071 ; +0,356] |

  **No se calibra.** La sigmoide empeora el log loss; la isotónica lo mejora en 0,0038, por debajo del umbral de 0,005 fijado
  antes de mirar, aunque sube el beneficio 0,23 millones por cada 1.000 asociados. Queda como mejora candidata para una
  versión siguiente; no se adopta ahora para no cambiar la regla después de ver los números.

## 3. La prueba, abierta **una sola vez**

- Fecha y hora en que se abrió: **2026-11-05 20:36:06** (`reports/prueba_abierta.txt`), 46 segundos después de congelar las
  decisiones. 38.815 asociados, 8.118 radicaron (20,9 %).
- ¿Se cambió algo del modelo después de verla? ☒ no.

| Métrica | Modelo final | Referencia | Intervalo del 95 % (asociados) |
|---|---|---|---|
| Beneficio neto por 1.000 (principal) | **33,60** | Contactar a todos: 18,79 · Baseline: 0 | [31,81 ; 35,40] |
| PR-AUC (control) | **0,7004** | Baseline: 0,2091 | [0,6901 ; 0,7113] |
| AUC (control) | **0,8958** | Baseline: 0,5000 | [0,8927 ; 0,8990] |
| Brier (control) | 0,1014 | Baseline: 0,2091 | [0,0998 ; 0,1031] |

**¿Mejora al baseline con confianza?** Sí. Modelo − baseline en PR-AUC: +0,4913 [0,4819 ; 0,5003]. Modelo − contactar a todos
en beneficio: **+14,81 millones por cada 1.000 asociados** [14,61 ; 15,00]. Ningún intervalo toca el cero.

**Generalización:** PR-AUC 0,6966 en validación cruzada, 0,6971 en reservados y 0,7004 en la prueba; AUC 0,8931, 0,8932 y
0,8958; beneficio 33,51, 33,44 y 33,60. Brecha de sobreajuste en la prueba: 0,0666 (tope 0,10).

## 4. Errores y carga de trabajo

**La campaña en la prueba (regla de valor esperado):** contacta a 15.930 de 38.815 asociados (41 %); encuentra a 6.142 de los
8.118 que radican (recall 75,7 %, precisión 38,6 %); cubre el 94,1 % del saldo de aportes en riesgo; beneficio neto de 1.304
millones con los supuestos. Para detectar al 75 % bastarían 10.523 contactos (precisión 57,9 %): la regla contacta unos 5.400
más porque prioriza a asociados de saldo alto.

| Segmento | Casos | Recall | % que el modelo contacta | % real que radica |
|---|---|---|---|---|
| Edad 14–25 | 3.842 | 71,1 % | 33,8 % | 22,7 % |
| Edad 26–35 / 36–50 / 51–65 | 10.642 / 12.862 / 8.463 | 75,7 % / 76,1 % / 75,9 % | 39,1 % / 41,2 % / 44,7 % | 21,7 % / 19,8 % / 20,9 % |
| Edad 66–100 | 3.001 | 79,5 % | 46,3 % | 21,0 % |
| Femenino / masculino | 19.717 / 19.098 | 75,2 % / 76,2 % | 41,5 % / 40,5 % | 22,2 % / 19,6 % |
| Medellín / otras zonas | 9.421 / 29.394 | 76,9 % / 75,2 % | 40,5 % / 41,2 % | 22,6 % / 20,4 % |
| Educación superior / otra | 15.763 / 23.052 | 77,9 % / 74,0 % | 44,0 % / 39,0 % | 22,4 % / 19,9 % |
| **Antigüedad 0–6 meses** | 4.777 | **49,2 %** | 14,0 % | 13,3 % |
| Antigüedad 7–24 / 25–60 meses | 9.887 / 14.294 | 74,0 % / 82,2 % | 42,6 % / 49,2 % | 31,0 % / 22,6 % |
| Antigüedad 61 meses o más | 9.857 | 76,3 % | 40,7 % | 12,0 % |

- ¿Hay un segmento al que le va peor? **Los asociados nuevos (0–6 meses):** el modelo encuentra solo a la mitad de quienes
  radican. Tienen poca historia y poco saldo, así que la regla les asigna poco valor esperado. Necesitan otra estrategia
  (bienvenida y fidelización temprana). Entre los grupos sociodemográficos las diferencias son pequeñas (recall de 71 % a
  80 %); el más joven queda 4,6 puntos por debajo del total.
- Carga que genera frente a la carga real: contacta al 41 % de los asociados cuando radica el 21 %; el 61 % de los contactos
  son a quienes no iban a radicar. Mercadeo debe confirmar esa capacidad. La regla contacta al 26,1 % de los retiros
  unilaterales (contactos que no evitan nada) frente al 32,9 % de los activos que no radican.
- Calibración en la prueba: creíble en el rango medio; subestima el tramo de mayor riesgo (predice 71 %, radica 79 %) y
  sobreestima un poco los bajos. Desviación máxima: 8 puntos.

## 5. Interpretación

- **Variables que más pesan** (SHAP, 600 asociados del entrenamiento): visitas a sucursal (15,5 % de la influencia, sube el
  riesgo), saldo de aportes (12,3 %, efecto no lineal), capitalización de aportes (11,5 %, sube), cartera mínima/máxima
  (11,4 %, baja), ingreso (8,6 %), antigüedad (7,9 %, baja) y ahorro fuera de los aportes (7,4 %, baja). Para el negocio:
  quien usa la cooperativa de forma transaccional y ve los aportes como ahorro disponible tiene más riesgo; quien tiene
  crédito y otros productos, menos.
- **Límite de la explicación:** explica al modelo, no al mundo; no dice si acertó, no es causalidad (no autoriza a concluir
  que ofrecer crédito retiene) y las variables correlacionadas se reparten el crédito.
- **Atributos sensibles:** edad, sexo, zona, educación y ocupación suman el 2,9 % de la influencia y el error por segmento es
  parecido entre grupos. Se declaran en la ficha del modelo y se revisan con el área jurídica antes de producción.
- **Sospechas de fuga:** suman el 40 % de la influencia. El equipo confirmó que no incluyen el mes de radicación; el peor caso
  sin ellas (notebook 05) seguía superando a contactar a todos.

## 6. Conclusión

- **¿Sirve para la decisión del caso?** Sí: en asociados nuevos para el modelo, deja 14,8 millones más por cada 1.000
  asociados que contactar a todos y cubre el 94 % del dinero en riesgo con el 41 % de los contactos. **Condiciones:** validar
  costo y efectividad con un piloto con grupo de control; confirmar la capacidad de contacto; aclarar la ventana de las
  variables sospechosas; complementar a los asociados de 0–6 meses con otra estrategia; recalibrar cada seis meses.
- **Qué NO debe hacerse con él:** negar productos o servicios; leer sus variables como causas; decidir sobre grupos
  sociodemográficos; volver a abrir esta prueba para mejorarlo.
