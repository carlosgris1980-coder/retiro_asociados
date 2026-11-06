# Despliegue de la app

> Plantilla en blanco. Complétala durante la clase 5 (viernes 6 de noviembre), después de desplegar la app.
> Guía paso a paso: `material-del-curso/guias/streamlit-despliegue.md` y `material-del-curso/kit/guia-despliegue-y-presentacion.md`.

## 1. Enlace y estado

| Dato | Valor |
|---|---|
| Enlace público de la app | |
| Plataforma | Streamlit Community Cloud |
| Repo, rama y archivo principal | `carlosgris1980-coder/retiro_asociados`, `main`, `app/streamlit_app.py` |
| Versión de Python elegida | ☐ 3.12  ☐ otra → motivo: _______________ |
| ¿Se comprobó en una **ventana privada**? | ☐ sí  ☐ no |
| ¿Se desplegó? | ☐ sí  ☐ no → **plan B** (apartado 4) |

## 2. Decisiones de despliegue y su motivo

| Decisión | Motivo |
|---|---|
| La app carga `models/model.joblib` desde el repo y **no consulta Databricks** | Sin credenciales en la nube ni dependencia de la cuota de otra plataforma. El archivo es el `champion` exportado en el notebook 09 y da las mismas probabilidades que la prueba del 07 |
| Versiones fijadas en `requirements.txt` | El modelo se guardó con scikit-learn 1.9.1: otra versión podría no cargarlo o dar otras probabilidades |
| Qué NO se subió al repo (y por qué) | Los datos (`data/raw`, `data/processed`: información de asociados), `.env`, el registro local de MLflow (`mlflow_local.db`) y el pipeline sin comprimir de 54,6 MB (el comprimido pesa 17,6 MB) |
| La app pide 19 datos, no 22 | Los 3 indicadores de tenencia de producto se calculan a partir de los saldos (`predict.prepare_inputs`) |

## 3. Pruebas del enlace público

| Caso | Datos usados | Resultado esperado | Resultado real |
|---|---|---|---|
| Normal | Perfil de ejemplo «riesgo intermedio» | Contactar · 29,7 % · valor esperado 0,193 millones | |
| Límite (cambia la decisión) | Perfil «alto riesgo» con saldo de aportes 0,25 y luego 0,20 | 0,25 → contactar (valor esperado 0,0375); 0,20 → no contactar (0,0298 < 0,03) | |
| Inválido (debe rechazarse con un mensaje claro) | Edad 200 (el formulario limita a 14–100; `predict` también la rechaza) | «edad: debe estar entre 14 y 100» | |
| Con un campo en «No sé» | Perfil «riesgo intermedio» con ingresos y saldo de crédito en «No sé» | Contactar · 33,2 % | |

## 4. Problemas encontrados y plan B

- Errores de despliegue que aparecieron y cómo se resolvieron (qué decía el registro y qué se cambió): _______________
- **Plan B** (si la app falla en vivo): captura con fecha y repo → `_______________`; app en local → `_______________`
- Si **no se desplegó**: **plan de despliegue escrito** (qué se intentó, qué error dio, qué harían): _______________

## 5. Ensayo cronometrado (clase 5)

| Parte | Tiempo previsto | Tiempo real |
|---|---|---|
| Caso de negocio | 2 min | |
| App (demo en vivo) | 4 min | |
| Resultados | 3 min | |
| Cierre | 1 min | |
| **Total** | **10 min** | |

- ¿Se pasó de 10 minutos? ☐ no  ☐ sí → qué se recortó: _______________
- Retroalimentación recibida (lo mejor y la corrección que propusieron): _______________
- **Corrección concreta que hicimos por ella:** _______________
- Quién habla de qué y quién responde las preguntas técnicas: _______________
