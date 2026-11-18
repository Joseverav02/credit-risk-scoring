# credit-risk-scoring

Modelo de **probabilidad de default** para préstamos, con evaluación orientada a negocio: no solo AUC, sino también calibración y el umbral de aprobación que maximiza el beneficio esperado.

> Datos **sintéticos** generados en `src/credit_risk/data.py`. No contiene información real de clientes.

## Qué demuestra
- Pipeline reproducible de scikit-learn (escalado, modelos, semillas fijas).
- Comparación entre modelo interpretable (regresión logística) y gradient boosting.
- Métricas de riesgo: ROC-AUC, PR-AUC y Brier score (calibración).
- Decisión de negocio: umbral de aprobación según margen y pérdida esperada.
- Tests automatizados con `pytest`.

## Uso

```bash
pip install -r requirements.txt
pytest -q
PYTHONPATH=src python -m credit_risk.train --model gbm
```

## Estructura

```
src/credit_risk/
  data.py    # generador sintético y features
  model.py   # entrenamiento, métricas y umbral de decisión
  train.py   # CLI
tests/
```

## Próximas mejoras
- Explicabilidad con SHAP.
- Monitorización de drift (PSI) en producción.
- API con FastAPI para servir predicciones.

## Licencia
MIT
