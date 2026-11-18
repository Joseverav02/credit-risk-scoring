"""CLI: entrena el modelo sobre datos sintéticos e imprime métricas y umbral de decisión."""

import argparse
import json

from credit_risk.data import make_synthetic_loans
from credit_risk.model import expected_loss_threshold, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el modelo de riesgo de crédito.")
    parser.add_argument("--model", choices=["logreg", "gbm"], default="gbm")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_synthetic_loans(n=args.n, seed=args.seed)
    result = train_and_evaluate(df, kind=args.model, seed=args.seed)

    proba = result.model.predict_proba(result.X_test)[:, 1]
    decision = expected_loss_threshold(
        proba, result.y_test.to_numpy(), result.X_test["loan_amount"].to_numpy()
    )

    report = {"model": args.model, "metrics": result.metrics, "decision": decision}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
