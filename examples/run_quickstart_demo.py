"""
Quickstart Verification and Synthetic FBHP Benchmark Demo.

This self-contained script verifies that the Python environment, data I/O,
cross-validation splitting, regression metrics, and JSON/CSV export routines
work as expected without modifying any manuscript outputs in `tab/` or `img/`.

Usage:
    python examples/run_quickstart_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


FEATURE_COLUMNS = ["Qo", "Qg", "Qw", "Id", "Depth", "Api", "St", "Bt", "Pwh"]
TARGET_COLUMN = "Pwf"


def generate_synthetic_fbhp_dataset(
    n_samples: int = 220, random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate a physics-inspired synthetic multiphase well dataset for testing.

    The synthetic target (Pwf in psia) combines wellhead pressure (Pwh),
    mixture hydrostatic head proportional to true vertical depth and fluid
    density, and nonlinear multiphase frictional pressure losses.
    """
    rng = np.random.default_rng(random_seed)

    qo = rng.uniform(300.0, 18000.0, size=n_samples)       # Oil rate (bbl/day)
    qg = rng.uniform(50.0, 12000.0, size=n_samples)        # Gas rate (mscf/day)
    qw = rng.uniform(0.0, 9000.0, size=n_samples)          # Water rate (bbl/day)
    tubing_id = rng.choice([2.441, 2.992, 3.81, 3.96, 4.0], size=n_samples)  # (in)
    depth = rng.uniform(4500.0, 7500.0, size=n_samples)    # Well depth (ft)
    api = rng.uniform(28.0, 38.0, size=n_samples)          # Oil gravity (deg API)
    st = rng.uniform(75.0, 155.0, size=n_samples)          # Surface temp (deg F)
    bt = st + rng.uniform(55.0, 115.0, size=n_samples)     # Bottom-hole temp (deg F)
    pwh = rng.uniform(100.0, 850.0, size=n_samples)        # Wellhead pressure (psia)

    # Approximate mixture gradient and frictional pressure drop (psia)
    total_liquid = qo + qw + 1e-6
    water_cut = qw / total_liquid
    oil_sg = 141.5 / (api + 131.5)
    mix_sg = (1.0 - water_cut) * oil_sg + water_cut * 1.04
    gas_lift_factor = 1.0 / (1.0 + 0.00018 * (qg / (total_liquid / 1000.0 + 1.0)))

    hydrostatic_dp = 0.433 * mix_sg * depth * gas_lift_factor
    friction_dp = (
        0.00045
        * (total_liquid / 1000.0) ** 1.75
        * (depth / 1000.0)
        / (tubing_id ** 4.8)
        * 120.0
    )
    thermal_correction = -0.15 * (bt - st)
    noise = rng.normal(0.0, 28.0, size=n_samples)

    pwf = pwh + hydrostatic_dp + friction_dp + thermal_correction + noise

    return pd.DataFrame(
        {
            "Qo": np.round(qo, 2),
            "Qg": np.round(qg, 2),
            "Qw": np.round(qw, 2),
            "Id": np.round(tubing_id, 3),
            "Depth": np.round(depth, 1),
            "Api": np.round(api, 2),
            "St": np.round(st, 1),
            "Bt": np.round(bt, 1),
            "Pwh": np.round(pwh, 2),
            "Pwf": np.round(pwf, 2),
        }
    )


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute standard regression evaluation metrics for FBHP prediction."""
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {"R2": r2, "RMSE_psia": rmse, "MAE_psia": mae, "MAPE_pct": mape}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"
    output_dir = repo_root / "examples" / "demo_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("paper_fbhp_automl_open :: Quickstart Environment & Pipeline Verification")
    print("=" * 72)

    # Check existing repository data directory (read-only check)
    if data_dir.exists():
        existing_files = sorted(
            p.name for p in data_dir.iterdir() if not p.name.startswith(".")
        )
        print(f"[INFO] Detected data/ directory with files: {existing_files}")
    else:
        print("[WARN] data/ directory not found; proceeding with synthetic test set.")

    # Create synthetic benchmark dataset for self-contained testing
    df = generate_synthetic_fbhp_dataset(n_samples=220, random_seed=42)
    sample_csv_path = output_dir / "synthetic_fbhp_sample.csv"
    df.to_csv(sample_csv_path, index=False)
    print(f"[INFO] Saved synthetic test dataset to: {sample_csv_path}")

    X = df[FEATURE_COLUMNS].to_numpy()
    y = df[TARGET_COLUMN].to_numpy()

    models = {
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    summary_rows = []
    json_records = {}

    for model_name, estimator in models.items():
        start_time = time.perf_counter()
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            estimator.fit(X[train_idx], y[train_idx])
            y_pred = estimator.predict(X[test_idx])
            metrics = compute_regression_metrics(y[test_idx], y_pred)
            metrics["fold"] = fold_idx
            fold_metrics.append(metrics)

        elapsed_sec = time.perf_counter() - start_time
        fold_df = pd.DataFrame(fold_metrics)

        summary_rows.append(
            {
                "Model": model_name,
                "R2_mean": round(float(fold_df["R2"].mean()), 4),
                "R2_std": round(float(fold_df["R2"].std()), 4),
                "RMSE_mean_psia": round(float(fold_df["RMSE_psia"].mean()), 2),
                "MAE_mean_psia": round(float(fold_df["MAE_psia"].mean()), 2),
                "MAPE_mean_pct": round(float(fold_df["MAPE_pct"].mean()), 2),
                "Runtime_sec": round(elapsed_sec, 2),
            }
        )
        json_records[model_name] = {
            "random_seed": 42,
            "n_splits": 5,
            "runtime_sec": round(elapsed_sec, 3),
            "folds": fold_metrics,
        }

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "demo_summary.csv"
    summary_json_path = output_dir / "demo_metrics.json"

    summary_df.to_csv(summary_csv_path, index=False)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2)

    print("\n[RESULTS] 5-Fold Cross-Validation Summary (Synthetic FBHP Benchmark):")
    print(summary_df.to_string(index=False))
    print(f"\n[INFO] Saved CSV summary to:  {summary_csv_path}")
    print(f"[INFO] Saved JSON metrics to: {summary_json_path}")
    print("\nQuickstart verification completed successfully.")


if __name__ == "__main__":
    main()