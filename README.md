# credit-risk-scoring

[![CI](https://github.com/Joseverav02/credit-risk-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/Joseverav02/credit-risk-scoring/actions/workflows/ci.yml)

Modelo de **probabilidad de default (PD)** para préstamos, evaluado con criterios de negocio: no solo AUC, sino también **calibración** y el **umbral de aprobación** que maximiza el beneficio esperado.

> Datos **sintéticos** generados en `src/credit_risk/data.py`. No contiene información real de clientes. Ver la [model card](docs/model_card.md) para límites y uso previsto.

## Resultados de referencia

Configuración: `gbm`, calibrado, 5000 préstamos, semilla 42 (`reports/metrics.json`).

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,795 |
| PR-AUC | 0,622 |
| Brier | 0,140 |
| Tasa de default observada / predicha | 24,6 % / 24,1 % |
| Umbral óptimo (beneficio esperado) | 0,14 |
| Tasa de aprobación con ese umbral | 39 % |

## Qué demuestra
- Pipeline reproducible con scikit-learn: escalado, modelos, calibración isotónica y semillas fijas.
- Comparación entre un modelo interpretable (regresión logística) y gradient boosting.
- Métricas de discriminación (ROC-AUC, PR-AUC) y de calibración (Brier, tasa predicha frente a observada).
- Decisión de negocio: umbral según margen y pérdida en caso de default (LGD).
- Tests con `pytest` (14 casos) y CI en GitHub Actions para Python 3.11 y 3.12.

## Uso

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src python -m credit_risk.train --model gbm --n 5000 --out reports/metrics.json
```

Opciones de la CLI: `--model {logreg,gbm}`, `--n`, `--seed`, `--no-calibrate`, `--out`.

## Estructura

```
src/credit_risk/
  data.py     # generador sintético y variables derivadas
  model.py    # pipelines, división, métricas y umbral de decisión
  train.py    # CLI que escribe métricas en JSON
docs/
  model_card.md
reports/
  metrics.json
tests/
  test_model.py
.github/workflows/ci.yml
```

## Decisiones de diseño
- **Calibración isotónica** en lugar de confiar en la salida cruda: en riesgo de crédito importa que la probabilidad sea interpretable, no solo que ordene bien.
- **Umbral por beneficio esperado** y no por una métrica genérica: el coste de un falso negativo (rechazar un buen cliente) y de un falso positivo (aprobar un mal cliente) son distintos.
- **Datos sintéticos con semilla fija**: el proyecto es reproducible y no expone datos reales.

## Próximas mejoras
- Explicabilidad por préstamo con SHAP.
- Análisis de equidad por grupos (edad, zona) antes de cualquier uso real.
- Monitorización de drift con PSI.
- API con FastAPI para servir predicciones.

## Licencia
MIT
