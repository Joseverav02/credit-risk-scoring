import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import FEATURES, TARGET, add_engineered_features, make_synthetic_loans
from credit_risk.model import (
    build_model,
    evaluate_proba,
    expected_profit_threshold,
    split_data,
    train_and_evaluate,
)
from credit_risk.train import main


@pytest.fixture(scope="module")
def loans():
    return make_synthetic_loans(n=3000, seed=3)


def test_synthetic_data_schema_and_positive_rate(loans):
    assert set(FEATURES + [TARGET]).issubset(loans.columns)
    assert 0.05 < loans[TARGET].mean() < 0.6
    assert loans[FEATURES].isna().sum().sum() == 0


def test_synthetic_data_is_reproducible():
    a = make_synthetic_loans(n=200, seed=11)
    b = make_synthetic_loans(n=200, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_engineered_features_match_definition(loans):
    out = add_engineered_features(loans)
    expected = (loans["existing_debt"] + loans["loan_amount"]) / loans["income"]
    assert np.allclose(out["debt_to_income"], expected)


def test_split_is_stratified_and_disjoint(loans):
    X_train, X_test, y_train, y_test = split_data(loans, seed=0)
    assert len(X_train) + len(X_test) == len(loans)
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_split_rejects_invalid_test_size(loans):
    with pytest.raises(ValueError):
        split_data(loans, test_size=1.5)


@pytest.mark.parametrize("kind", ["logreg", "gbm"])
def test_models_beat_random(loans, kind):
    result = train_and_evaluate(loans, kind=kind, calibrate=False)
    assert result.metrics["roc_auc"] > 0.7
    assert 0 < result.metrics["brier"] < 0.25


def test_calibrated_model_predicts_sane_probabilities(loans):
    result = train_and_evaluate(loans, kind="gbm", calibrate=True)
    proba = result.model.predict_proba(result.X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    gap = abs(result.metrics["default_rate_predicted"] - result.metrics["default_rate_observed"])
    assert gap < 0.05


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        build_model("random_forest")


def test_evaluate_proba_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    metrics = evaluate_proba(y, np.array([0.0, 0.1, 0.9, 1.0]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["brier"] < 0.01


def test_threshold_maximises_profit_on_toy_example():
    proba = np.array([0.05, 0.10, 0.60, 0.90])
    y = np.array([0, 0, 1, 1])
    amount = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    best = expected_profit_threshold(proba, y, amount)
    # Aprobar solo los dos de bajo riesgo: 2 * 1000 * 0.08 = 160
    assert best["profit"] == pytest.approx(160.0)
    assert best["approval_rate"] == pytest.approx(0.5)


def test_threshold_rejects_bad_lgd():
    with pytest.raises(ValueError):
        expected_profit_threshold(np.array([0.1]), np.array([0]), np.array([1.0]), loss_given_default=2)


def test_cli_writes_json(tmp_path):
    out = tmp_path / "metrics.json"
    code = main(["--model", "logreg", "--n", "800", "--seed", "1", "--out", str(out)])
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"] == "logreg"
    assert "roc_auc" in data["metrics"]


def test_cli_rejects_tiny_sample():
    with pytest.raises(SystemExit):
        main(["--n", "10"])
