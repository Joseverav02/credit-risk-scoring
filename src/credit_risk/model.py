"""Entrenamiento, evaluación y decisión de un modelo de probabilidad de default (PD).

Flujo:
    datos -> features -> pipeline (escalado + modelo + calibración) -> métricas -> umbral de decisión
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import FEATURES, TARGET, add_engineered_features

MODEL_FEATURES = FEATURES + ["debt_to_income", "loan_to_income"]
MODEL_KINDS = ("logreg", "gbm")


@dataclass
class TrainResult:
    model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    metrics: dict = field(default_factory=dict)


def build_model(kind: str = "gbm", calibrate: bool = True, seed: int = 0) -> Pipeline:
    """Construye el pipeline.

    - ``logreg``: regresión logística con escalado. Interpretable y rápida.
    - ``gbm``: gradient boosting de histogramas. Suele captar mejor las no linealidades.
    - ``calibrate``: calibración isotónica para que las probabilidades sean fiables
      (una PD de 0,10 debe corresponder a ~10 % de defaults observados).
    """
    if kind == "logreg":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
        steps = [("scale", StandardScaler()), ("clf", estimator)]
    elif kind == "gbm":
        estimator = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed
        )
        steps = [("clf", estimator)]
    else:
        raise ValueError(f"Tipo de modelo desconocido: {kind!r}. Opciones: {MODEL_KINDS}")

    if calibrate:
        steps[-1] = ("clf", CalibratedClassifierCV(steps[-1][1], method="isotonic", cv=3))
    return Pipeline(steps)


def split_data(df: pd.DataFrame, seed: int = 42, test_size: float = 0.25):
    """Divide en train/test de forma estratificada y reproducible."""
    if not 0 < test_size < 1:
        raise ValueError("test_size debe estar entre 0 y 1")
    data = add_engineered_features(df)
    return train_test_split(
        data[MODEL_FEATURES], data[TARGET],
        test_size=test_size, stratify=data[TARGET], random_state=seed,
    )


def evaluate_proba(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Métricas de discriminación (AUC, PR-AUC) y calibración (Brier)."""
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "default_rate_observed": float(np.mean(y_true)),
        "default_rate_predicted": float(np.mean(proba)),
    }


def train_and_evaluate(df: pd.DataFrame, kind: str = "gbm", calibrate: bool = True,
                       seed: int = 42) -> TrainResult:
    X_train, X_test, y_train, y_test = split_data(df, seed=seed)
    model = build_model(kind, calibrate=calibrate, seed=seed)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_proba(y_test.to_numpy(), proba)
    return TrainResult(model=model, X_test=X_test, y_test=y_test, metrics=metrics)


def expected_profit_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    loan_amount: np.ndarray,
    loss_given_default: float = 0.45,
    margin: float = 0.08,
    grid: int = 99,
) -> dict:
    """Busca el umbral de aprobación que maximiza el beneficio esperado.

    Se aprueba un préstamo si PD < umbral. Para cada préstamo aprobado:
        beneficio ≈ (1 - default_real) * importe * margen - default_real * importe * LGD
    Se usan los defaults *observados* (y_true) para evaluar la política sobre datos de test.
    """
    if len(proba) == 0:
        raise ValueError("No hay datos para elegir el umbral")
    if not 0 <= loss_given_default <= 1:
        raise ValueError("loss_given_default debe estar entre 0 y 1")

    best = {"threshold": 0.0, "profit": 0.0, "approval_rate": 0.0}
    for t in np.linspace(0.01, 0.99, grid):
        approved = proba < t
        if not approved.any():
            continue
        y = y_true[approved]
        amount = loan_amount[approved]
        profit = float(np.sum((1 - y) * amount * margin - y * amount * loss_given_default))
        if profit > best["profit"]:
            best = {"threshold": float(t), "profit": profit, "approval_rate": float(approved.mean())}
    return best
