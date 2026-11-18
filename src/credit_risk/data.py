"""Generación de datos sintéticos de préstamos para entrenar y evaluar el modelo.

Los datos son sintéticos: no contienen información real de clientes.
"""

import numpy as np
import pandas as pd


def make_synthetic_loans(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Crea un dataset de préstamos con una relación realista entre variables y default."""
    rng = np.random.default_rng(seed)

    income = rng.lognormal(mean=10.2, sigma=0.5, size=n)          # ingresos anuales (€)
    age = rng.integers(21, 70, size=n)
    loan_amount = rng.lognormal(mean=9.5, sigma=0.6, size=n)      # importe solicitado (€)
    years_employed = np.clip(rng.normal(8, 6, size=n), 0, 40)
    existing_debt = rng.uniform(0, 1, size=n) * income * 0.8     # deuda previa (€)
    late_payments = rng.poisson(0.4, size=n)                      # pagos atrasados históricos

    debt_to_income = (existing_debt + loan_amount) / income
    logit = (
        -3.0
        + 2.2 * debt_to_income
        + 0.6 * late_payments
        - 0.08 * np.clip(years_employed, 0, 25)
        - 0.000015 * income
    )
    p_default = 1 / (1 + np.exp(-logit))
    default = rng.uniform(size=n) < p_default

    return pd.DataFrame(
        {
            "income": income.round(2),
            "age": age,
            "loan_amount": loan_amount.round(2),
            "years_employed": years_employed.round(1),
            "existing_debt": existing_debt.round(2),
            "late_payments": late_payments,
            "default": default.astype(int),
        }
    )


FEATURES = ["income", "age", "loan_amount", "years_employed", "existing_debt", "late_payments"]
TARGET = "default"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade ratios de endeudamiento, variables típicas en scoring de crédito."""
    out = df.copy()
    out["debt_to_income"] = (out["existing_debt"] + out["loan_amount"]) / out["income"]
    out["loan_to_income"] = out["loan_amount"] / out["income"]
    return out
