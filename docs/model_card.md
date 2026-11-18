# Model card: modelo de probabilidad de default (PD)

## Resumen
Modelo de clasificación binaria que estima la probabilidad de que un préstamo entre en default.
Se usa para decidir qué préstamos aprobar mediante un umbral de PD.

**Estado:** prototipo con datos sintéticos. No apto para decisiones reales de crédito.

## Datos
- **Origen:** generador sintético (`src/credit_risk/data.py`), semilla fija.
- **Variables:** ingresos, edad, importe, años de empleo, deuda previa, pagos atrasados.
  Derivadas: `debt_to_income`, `loan_to_income`.
- **Objetivo:** `default` (1 = impago).
- **Tasa de default observada (test):** ~25 %.

## Modelos comparados
| Modelo | Descripción | Ventaja | Limitación |
|---|---|---|---|
| `logreg` | Regresión logística con escalado y `class_weight="balanced"` | Interpretable | Captura mal las no linealidades |
| `gbm` | Gradient boosting de histogramas, calibrado con isotónica | Mejor discriminación | Menos interpretable |

## Métricas (semilla 42, 5000 préstamos, test 25 %)
Ver `reports/metrics.json`, que se regenera con el comando de la sección de uso.

| Métrica | Valor | Qué significa |
|---|---|---|
| ROC-AUC | ~0,80 | Capacidad de ordenar buenos y malos pagadores |
| PR-AUC | ~0,62 | Precisión frente a exhaustividad en la clase default |
| Brier | ~0,14 | Error cuadrático de las probabilidades (menor es mejor) |
| Tasa observada vs. predicha | 24,6 % vs. 24,1 % | Calibración global aceptable |

## Decisión
El umbral se elige maximizando el beneficio esperado sobre test, con margen del 8 % y pérdida en caso de default del 45 % (LGD).
Resultado con la configuración por defecto: umbral 0,14, aprobación del ~39 % de las solicitudes.

## Limitaciones y riesgos
- Datos sintéticos: las relaciones son las del generador, no las de una cartera real.
- No hay análisis de sesgo por edad, género o zona. Es obligatorio antes de usarlo con clientes reales.
- No hay monitorización de drift. En producción haría falta PSI por variable y recalibración periódica.
- Los parámetros de negocio (margen, LGD) son supuestos, no datos reales.

## Uso previsto y no previsto
- **Previsto:** demostración técnica, docencia y punto de partida para experimentos.
- **No previsto:** decisiones automáticas de concesión de crédito a personas.
