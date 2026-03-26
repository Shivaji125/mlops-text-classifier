"""Evaluation pipeline with quality gates.
Compare new model against production baseline.
Blocks deployment if new model regresses.
"""
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import mlflow
from src.logging_config import get_logger

logger = get_logger(__name__, log_file="logs/evaluate.log")

# Minimum thresholds for any model to pass
# Set to 0.60 fo 500 dataset; raise to 0.8 for larger datasets
ABSOLUTE_THRESHOLDS = {
    "test_f1": 0.60,
    "test_accuracy": 0.60,
}

# Maximum allowed regression vs production model
MAX_REGRESSION = 0.02   # 2% drop allowed

def get_production_metrics(model_name: str = "sentiment-classifier") -> dict | None:
    """Fetch metrics from the current production model
    (the model tagged  'production' in MLflow registry).
    returns None if no production model exists yet.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return {k: v for k,v in run.data.metrics.items() if k.startswith("test_")}
    except Exception:
        return None

def evaluate_model(run_id: str, experiment_name: str = "sentiment-classifier") -> bool:
    """
    Run quality gates on a trained model

    Gates:
    1. Absolute thresholds - model must meet minimum F1 and accuracy
    2. Regression check - model must not drop > 2% vs production model
    3. Consistency check - train/val/test metrics should not diverge wildly

    Returns True if all gates pass.
    """
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    logger.info(f"Evaluating run: {run_id}")
    logger.info(f"Run metrics: { {k: round(v,4) for k, v in metrics.items() if k.startswith('test_')}}")

    gate_results = []

    # ---- Gate 1: Absolute thresholds -----------------
    logger.info("Gate 1: Checking absolute thresholds...") 
    for metric_name, threshold in ABSOLUTE_THRESHOLDS.items():
        value = metrics.get(metric_name, 0)
        passed = value >= threshold
        gate_results.append({
            "gate": f"absolute_{metric_name}",
            "threshold": threshold,
            "actual": round(value, 4),
            "passed": passed,
        })
        if passed:
            logger.info(f" PASSED: {metric_name} = {value:.4f} >= {threshold}")
        else:
            logger.warning(f" FAILED: {metric_name} = {value:.4f} < {threshold}")


    # --- Gate 2: Regression vs production -------------
    logger.info("Gate 2: Checking regression vs production model...")  
    prod_metrics = get_production_metrics()
    if prod_metrics:
        for metric_name, prod_value in prod_metrics.items():
            new_value = metrics.get(metric_name, 0)
            regression = prod_value - new_value
            passed = regression <= MAX_REGRESSION
            gate_results.append({
                "gate": f"regression_{metric_name}",
                "production_value": round(prod_value, 4),
                "new_value": round(new_value, 4),
                "regression": round(regression, 4),
                "max_allowed": MAX_REGRESSION,
                "passed": passed,
            })
            if passed:
                logger.info(
                    f" PASSED: {metric_name} regression = {regression:.4f}"
                    f"(prod_value:.4f) -> new={new_value:.4f}"
                )
            else:
                logger.warning(
                    f" FAILED: {metric_name} regressed by {regression:.4f}"
                    f"(prod_value:.4f) -> new={new_value:.4f}"
                )
    else:
        logger.info(" No production model found - skipping regression checks")
    
    
    # --- Gate 3: Overfitting check --------------------
    logger.info("Gate 3: Checking for overfitting...")
    test_f1 = metrics.get("test_f1", 0)
    val_f1 = metrics.get("val_f1", metrics.get("best_val_f1", 0))
    overfit_gap = abs(val_f1 -  test_f1)
    overfit_passed = overfit_gap < 0.05   # val-test f1 gap should be < 5%
    gate_results.append({
        "gate": "overfitting_check",
        "val_f1": round(val_f1, 4),
        "test_f1": round(test_f1, 4),
        "gap": round(overfit_gap, 4),
        "max_gap": 0.05,
        "passed": overfit_passed,
    })
    if overfit_passed:
        logger.info(f" PASSED: val-test F1 gap = {overfit_gap:.4f} < 0.05")
    else:
        logger.warning(f" FAILED: val-test F1 gap = {overfit_gap:.4f} >= 0.05")
    
    # ------- Results------------------
    all_passed = all(g["passed"] for g in gate_results)
    report= {
        "run_id": run_id,
        "overall_passed": all_passed,
        "gates": gate_results,
    }

    # Save report
    report_path = Path("evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Log report to MLflow
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(report_path))
    except Exception as e:
        logger.warning(f"Couldn not log artifact to Mlflow: {e}")
    
    #------- Final Verdict--------------------
    logger.info("=" * 50)
    if all_passed:
        logger.info("ALL QUALITY GATES PASSED - model is ready for deployment")
        passed_count = len(gate_results)
        logger.info(f"Gates passed: {passed_count} / {passed_count}")
    else:
        failed = [g["gate"] for g in gate_results if not g["passed"]]
        passed_count = sum(1 for g in gate_results if g["passed"])
        logger.error(f"QUALITY GATES FAILED - {len(failed)} gates failed")
        logger.error(f"Failed gates: {failed}")
        logger.error("Model will NOT be promoted to production")
    logger.info("=" * 50)

    return all_passed

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quality gates on trained model")
    parser.add_argument("--run-id", required=True, help="MLflow run ID to evaluate")
    args = parser.parse_args()

    passed = evaluate_model(args.run_id)
    if not passed:
        sys.exit(1)
    else:
        print("\nQuality gates passed1 Safe to deploy.")