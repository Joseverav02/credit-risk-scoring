import numpy as np

from credit_risk.data import FEATURES, TARGET, add_engineered_features, make_synthetic_loans
from credit_risk.model import expected_loss_threshold, train_and_evaluate


def test_synthetic_data_has_expected_columns_and_rate():
    df = make_synthetic_loans(n=2000, seed=1)
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert 0.02 < df[TARGET].mean() < 0.9


def test_engineered_features_are_computed():
    df = make_synthetic_loans(n=100, seed=2)
    out = add_engineered_features(df)
    expected = (df["existing_debt"] + df["loan_amount"]) / df["income"]
    assert np.allclose(out["debt_to_income"], expected)


def test_model_beats_random_on_synthetic_data():
    df = make_synthetic_loans(n=3000, seed=3)
    result = train_and_evaluate(df, kind="logreg")
    assert result.metrics["roc_auc"] > 0.65


def test_threshold_respects_profit_logic():
    proba = np.array([0.05, 0.10, 0.60, 0.90])
    y = np.array([0, 0, 1, 1])
    amount = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    best = expected_loss_threshold(proba, y, amount)
    assert best["profit"] > 0
    assert best["threshold"] is not None
