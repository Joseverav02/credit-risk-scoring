"""CLI: entrena el modelo sobre datos sintéticos y escribe métricas en JSON.

Ejemplo:
    python -m credit_risk.train --model gbm --n 5000 --out reports/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from credit_risk.data import make_synthetic_loans
from credit_risk.model import MODEL_KINDS, expected_profit_threshold, train_and_evaluate

logger = logging.getLogger("credit_risk")


def run(model: str, n: int, seed: int, calibrate: bool) -> dict:
    df = make_synthetic_loans(n=n, seed=seed)
    result = train_and_evaluate(df, kind=model, calibrate=calibrate, seed=seed)
    proba = result.model.predict_proba(result.X_test)[:, 1]
    decision = expected_profit_threshold(
        proba, result.y_test.to_numpy(), result.X_test["loan_amount"].to_numpy()
    )
    return {
        "model": model,
        "calibrated": calibrate,
        "n_rows": n,
        "seed": seed,
        "metrics": result.metrics,
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Entrena el modelo de riesgo de crédito.")
    parser.add_argument("--model", choices=MODEL_KINDS, default="gbm")
    parser.add_argument("--n", type=int, default=5000, help="número de préstamos sintéticos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-calibrate", action="store_true", help="desactiva la calibración")
    parser.add_argument("--out", type=Path, help="ruta del JSON de salida (opcional)")
    args = parser.parse_args(argv)

    if args.n < 100:
        parser.error("--n debe ser al menos 100")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = run(args.model, args.n, args.seed, calibrate=not args.no_calibrate)
    text = json.dumps(report, indent=2, ensure_ascii=False)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        logger.info("Métricas escritas en %s", args.out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
