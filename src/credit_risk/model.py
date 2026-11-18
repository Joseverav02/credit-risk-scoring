"""Entrenamiento, evaluación y decisión de un modelo de probabilidad de default."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.data import FEATURES, TARGET, add_engineered_features


@dataclass
class TrainResult:
    model: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    metrics: dict


def build_model(kind: str = "gbm") -> Pipeline:
    """Devuelve un pipeline. 'logreg' es interpretable; 'gbm' suele ser más preciso."""
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if kind == "gbm":
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        return Pipeline([("clf", clf)])
    raise ValueError(f"Tipo de modelo desconocido: {kind}")


def train_and_evaluate(df: pd.DataFrame, kind: str = "gbm", seed: int = 42) -> TrainResult:
    data = add_engineered_features(df)
    features = FEATURES + ["debt_to_income", "loan_to_income"]

    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data[TARGET], test_size=0.25, stratify=data[TARGET], random_state=seed
    )

    model = build_model(kind)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "default_rate_test": float(y_test.mean()),
    }
    return TrainResult(model=model, X_test=X_test, y_test=y_test, metrics=metrics)


def expected_loss_threshold(
    proba: np.ndarray, y_true: np.ndarray, loan_amount: np.ndarray,
    loss_given_default: float = 0.45, margin: float = 0.08,
) -> dict:
    """Elige el umbral de aprobación que maximiza el beneficio esperado.

    Aprobar un préstamo de importe L con probabilidad de default p:
      beneficio ≈ (1 - p) * L * margin - p * L * loss_given_default
    """
    best = {"threshold": None, "profit": -np.inf}
    for t in np.linspace(0.01, 0.99, 99):
        approved = proba < t  # aprobamos si el riesgo estimado es bajo
        profit = np.sum(
            (1 - y_true[approved]) * loan_amount[approved] * margin
            - y_true[approved] * loan_amount[approved] * loss_given_default
        )
        if profit > best["profit"]:
            best = {"threshold": float(t), "profit": float(profit), "approval_rate": float(approved.mean())}
    return best
