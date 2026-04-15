from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from models.kalshi_timeseries_model import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    RidgeARXRegressor,
    add_future_target,
    add_past_only_features,
    chronological_event_split,
    evaluate,
    leakage_columns,
    read_final_data,
    resolve_data_path,
)


EXPERIMENT_DIR = DEFAULT_OUTPUT_DIR / "experiments"


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    feature_group: str
    target_mode: str
    alpha: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep BTC Kalshi time-series forecasting/trading experiments."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 30, 60, 120, 300],
        help="Forecast horizons in seconds.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.05],
        help="Trade when abs(predicted future price - current price) >= threshold.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.01,
        help="Minimum trade coverage for selecting a best trading threshold.",
    )
    return parser.parse_args()


def feature_columns(df: pl.DataFrame, target_col: str, feature_group: str) -> list[str]:
    dropped = set(leakage_columns(df, target_col))
    numeric = [
        col
        for col, dtype in zip(df.columns, df.dtypes)
        if col not in dropped and dtype.is_numeric()
    ]

    def has_prefix(col: str, prefixes: tuple[str, ...]) -> bool:
        return any(col.startswith(prefix) for prefix in prefixes)

    if feature_group == "kalshi_core":
        return [
            col
            for col in numeric
            if not has_prefix(col, ("ETH_", "XRP_", "SOL_", "btc_spot_"))
        ]

    if feature_group == "kalshi_cross_coin":
        return [
            col
            for col in numeric
            if not has_prefix(col, ("btc_spot_",))
        ]

    if feature_group == "kalshi_spot":
        return [
            col
            for col in numeric
            if not has_prefix(col, ("ETH_", "XRP_", "SOL_"))
        ]

    if feature_group == "spot_only":
        return [
            col
            for col in numeric
            if col == "yes_mid_dollars" or col.startswith("btc_spot_")
        ]

    if feature_group == "all":
        return numeric

    raise ValueError(f"Unknown feature_group: {feature_group}")


def build_specs() -> list[ExperimentSpec]:
    specs = [
        ExperimentSpec("persistence", "none", "persistence", 0.0),
        ExperimentSpec("trend_5s", "none", "trend_5s", 0.0),
    ]

    for target_mode in ["price", "delta"]:
        for feature_group in ["kalshi_core", "kalshi_spot", "kalshi_cross_coin", "all"]:
            for alpha in [0.1, 10.0, 1000.0]:
                specs.append(
                    ExperimentSpec(
                        name=f"ridge_{target_mode}_{feature_group}_alpha_{alpha:g}",
                        feature_group=feature_group,
                        target_mode=target_mode,
                        alpha=alpha,
                    )
                )

    return specs


def clean_feature_frame(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    needed = [*feature_cols, target_col, "yes_mid_dollars"]
    if "yes_spread_dollars" in df.columns:
        needed.append("yes_spread_dollars")

    clean_df = df.select(list(dict.fromkeys(needed))).drop_nulls()
    x = clean_df.select(feature_cols).to_numpy()
    target = clean_df.select(target_col).to_numpy().ravel()
    current = clean_df.select("yes_mid_dollars").to_numpy().ravel()

    if "yes_spread_dollars" in clean_df.columns:
        spread = clean_df.select("yes_spread_dollars").to_numpy().ravel()
        spread = np.nan_to_num(spread, nan=0.0, posinf=0.0, neginf=0.0)
        spread = np.maximum(spread, 0.0)
    else:
        spread = np.zeros_like(current)

    return x, target, current, spread, np.arange(len(target))


def persistence_predictions(test_df: pl.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_df = test_df.select([target_col, "yes_mid_dollars", "yes_spread_dollars"]).drop_nulls()
    actual = clean_df.select(target_col).to_numpy().ravel()
    current = clean_df.select("yes_mid_dollars").to_numpy().ravel()
    spread = clean_df.select("yes_spread_dollars").to_numpy().ravel()
    return actual, current, np.nan_to_num(np.maximum(spread, 0.0), nan=0.0)


def trend_predictions(
    test_df: pl.DataFrame,
    target_col: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = [target_col, "yes_mid_dollars", "yes_mid_lag_5s", "yes_spread_dollars"]
    clean_df = test_df.select(required).drop_nulls()
    actual = clean_df.select(target_col).to_numpy().ravel()
    current = clean_df.select("yes_mid_dollars").to_numpy().ravel()
    lag = clean_df.select("yes_mid_lag_5s").to_numpy().ravel()
    spread = clean_df.select("yes_spread_dollars").to_numpy().ravel()
    slope_per_second = (current - lag) / 5.0
    pred = np.clip(current + slope_per_second * horizon, 0.0, 1.0)
    return actual, pred, current, np.nan_to_num(np.maximum(spread, 0.0), nan=0.0)


def trading_metrics(
    actual: np.ndarray,
    pred: np.ndarray,
    current: np.ndarray,
    spread: np.ndarray,
    thresholds: list[float],
    min_coverage: float,
) -> dict[str, float]:
    actual_delta = actual - current
    pred_delta = pred - current
    pred_direction = np.sign(pred_delta)
    actual_direction = np.sign(actual_delta)
    nonzero = actual_direction != 0
    directional_accuracy = (
        float(np.mean(pred_direction[nonzero] == actual_direction[nonzero]))
        if np.any(nonzero)
        else 0.0
    )
    corr = (
        float(np.corrcoef(pred_delta, actual_delta)[0, 1])
        if np.std(pred_delta) > 0 and np.std(actual_delta) > 0
        else 0.0
    )

    threshold_rows = []
    for threshold in thresholds:
        position = np.where(np.abs(pred_delta) >= threshold, np.sign(pred_delta), 0.0)
        traded = position != 0
        coverage = float(np.mean(traded))

        if np.any(traded):
            gross_pnl = position[traded] * actual_delta[traded]
            half_spread_round_trip_pnl = gross_pnl - spread[traded]
            one_cent_pnl = gross_pnl - 0.01
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "coverage": coverage,
                    "trade_count": int(np.sum(traded)),
                    "hit_rate": float(np.mean(gross_pnl > 0)),
                    "avg_gross_pnl": float(np.mean(gross_pnl)),
                    "total_gross_pnl": float(np.sum(gross_pnl)),
                    "avg_pnl_after_1c": float(np.mean(one_cent_pnl)),
                    "avg_pnl_after_current_spread": float(np.mean(half_spread_round_trip_pnl)),
                }
            )
        else:
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "coverage": coverage,
                    "trade_count": 0,
                    "hit_rate": 0.0,
                    "avg_gross_pnl": 0.0,
                    "total_gross_pnl": 0.0,
                    "avg_pnl_after_1c": 0.0,
                    "avg_pnl_after_current_spread": 0.0,
                }
            )

    eligible = [row for row in threshold_rows if row["coverage"] >= min_coverage]
    if not eligible:
        eligible = threshold_rows
    best_by_spread = max(eligible, key=lambda row: row["avg_pnl_after_current_spread"])
    best_by_1c = max(eligible, key=lambda row: row["avg_pnl_after_1c"])

    payload = {
        "directional_accuracy": directional_accuracy,
        "pred_actual_delta_corr": corr,
        "best_threshold_by_spread_cost": best_by_spread["threshold"],
        "best_coverage_by_spread_cost": best_by_spread["coverage"],
        "best_hit_rate_by_spread_cost": best_by_spread["hit_rate"],
        "best_avg_gross_pnl_by_spread_cost": best_by_spread["avg_gross_pnl"],
        "best_avg_pnl_after_current_spread": best_by_spread["avg_pnl_after_current_spread"],
        "best_threshold_by_1c_cost": best_by_1c["threshold"],
        "best_coverage_by_1c_cost": best_by_1c["coverage"],
        "best_avg_pnl_after_1c": best_by_1c["avg_pnl_after_1c"],
    }

    for row in threshold_rows:
        suffix = str(row["threshold"]).replace(".", "p")
        payload[f"coverage_thr_{suffix}"] = row["coverage"]
        payload[f"avg_gross_pnl_thr_{suffix}"] = row["avg_gross_pnl"]
        payload[f"avg_pnl_after_current_spread_thr_{suffix}"] = row["avg_pnl_after_current_spread"]

    return payload


def run_spec(
    spec: ExperimentSpec,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col: str,
    horizon: int,
    thresholds: list[float],
    min_coverage: float,
) -> tuple[dict[str, float | int | str], object | None, list[str]]:
    started = time.monotonic()

    if spec.target_mode == "persistence":
        actual, current, spread = persistence_predictions(test_df, target_col)
        pred = current.copy()
        feature_cols: list[str] = []
        fitted = None
    elif spec.target_mode == "trend_5s":
        actual, pred, current, spread = trend_predictions(test_df, target_col, horizon)
        feature_cols = ["yes_mid_dollars", "yes_mid_lag_5s"]
        fitted = None
    else:
        feature_cols = feature_columns(train_df, target_col, spec.feature_group)
        x_train, train_target, train_current, _, _ = clean_feature_frame(
            train_df,
            feature_cols,
            target_col,
        )
        x_test, actual, current, spread, _ = clean_feature_frame(
            test_df,
            feature_cols,
            target_col,
        )

        if spec.target_mode == "delta":
            y_train = train_target - train_current
        elif spec.target_mode == "price":
            y_train = train_target
        else:
            raise ValueError(f"Unknown target_mode: {spec.target_mode}")

        fitted = RidgeARXRegressor(alpha=spec.alpha).fit(x_train, y_train)
        raw_pred = fitted.predict(x_test)
        pred = current + raw_pred if spec.target_mode == "delta" else raw_pred
        pred = np.clip(pred, 0.0, 1.0)

    regression = evaluate(actual, pred)
    trading = trading_metrics(actual, pred, current, spread, thresholds, min_coverage)
    elapsed = time.monotonic() - started

    row = {
        "horizon_seconds": horizon,
        "experiment": spec.name,
        "feature_group": spec.feature_group,
        "target_mode": spec.target_mode,
        "alpha": spec.alpha,
        "n_test_rows": int(len(actual)),
        "n_features": len(feature_cols),
        "elapsed_seconds": round(elapsed, 3),
        **regression,
        **trading,
    }
    return row, fitted, feature_cols


def write_summary(
    output_dir: Path,
    results: pl.DataFrame,
    best_overall: dict,
    best_model_path: Path,
) -> None:
    top_by_horizon = (
        results.sort(
            ["horizon_seconds", "best_avg_pnl_after_current_spread", "directional_accuracy"],
            descending=[False, True, True],
        )
        .group_by("horizon_seconds", maintain_order=True)
        .head(3)
    )
    summary_cols = [
        "horizon_seconds",
        "experiment",
        "rmse",
        "mae",
        "directional_accuracy",
        "best_threshold_by_spread_cost",
        "best_coverage_by_spread_cost",
        "best_avg_pnl_after_current_spread",
    ]
    table_lines = [
        "| " + " | ".join(summary_cols) + " |",
        "| " + " | ".join(["---"] * len(summary_cols)) + " |",
    ]
    for row in top_by_horizon.select(summary_cols).iter_rows(named=True):
        values = []
        for col in summary_cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        table_lines.append("| " + " | ".join(values) + " |")

    summary_path = output_dir / "experiment_summary.md"
    lines = [
        "# Kalshi Time-Series Experiment Summary",
        "",
        "Selection objective: maximize average per-contract PnL after an approximate current-spread round-trip cost,",
        "requiring at least the configured minimum test-set trade coverage.",
        "",
        "## Best Overall",
        "",
        f"- Horizon: {best_overall['horizon_seconds']} seconds",
        f"- Experiment: {best_overall['experiment']}",
        f"- Feature group: {best_overall['feature_group']}",
        f"- Target mode: {best_overall['target_mode']}",
        f"- Alpha: {best_overall['alpha']}",
        f"- RMSE: {best_overall['rmse']:.6f}",
        f"- MAE: {best_overall['mae']:.6f}",
        f"- Directional accuracy: {best_overall['directional_accuracy']:.4f}",
        f"- Best threshold: {best_overall['best_threshold_by_spread_cost']}",
        f"- Coverage at best threshold: {best_overall['best_coverage_by_spread_cost']:.4f}",
        f"- Hit rate at best threshold: {best_overall['best_hit_rate_by_spread_cost']:.4f}",
        f"- Avg gross PnL at best threshold: {best_overall['best_avg_gross_pnl_by_spread_cost']:.6f}",
        f"- Avg PnL after current-spread cost: {best_overall['best_avg_pnl_after_current_spread']:.6f}",
        f"- Saved model: `{best_model_path}`",
        "",
        "## Top 3 By Horizon",
        "",
        "\n".join(table_lines),
        "",
        "## Notes",
        "",
        "- `persistence` means predicting the current YES midpoint as the future YES midpoint.",
        "- `trend_5s` extrapolates the last 5 seconds of YES midpoint movement.",
        "- `ridge_price_*` directly predicts the future price.",
        "- `ridge_delta_*` predicts the future price change and adds it back to the current price.",
        "- Spread-adjusted PnL is only an approximation because the future executable spread is not in the target.",
    ]
    summary_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_data_path(args.data_path)
    base_df = add_past_only_features(read_final_data(data_path))
    specs = build_specs()
    all_rows = []
    best_score = -np.inf
    best_bundle = None

    for horizon in args.horizons:
        print(f"\n=== horizon {horizon}s ===")
        target_col = f"yes_mid_dollars_t_plus_{horizon}s"
        horizon_df = add_future_target(base_df, horizon)
        train_df, test_df = chronological_event_split(horizon_df, args.test_size)

        for spec in specs:
            row, fitted, feature_cols = run_spec(
                spec,
                train_df,
                test_df,
                target_col,
                horizon,
                args.thresholds,
                args.min_coverage,
            )
            all_rows.append(row)
            print(
                f"{spec.name}: rmse={row['rmse']:.4f}, "
                f"dir={row['directional_accuracy']:.3f}, "
                f"spread_pnl={row['best_avg_pnl_after_current_spread']:.5f}, "
                f"thr={row['best_threshold_by_spread_cost']}"
            )

            score = row["best_avg_pnl_after_current_spread"]
            if row["best_coverage_by_spread_cost"] >= args.min_coverage and score > best_score:
                best_score = score
                best_bundle = {
                    "row": row,
                    "model": fitted,
                    "feature_cols": feature_cols,
                    "target_col": target_col,
                }

    results = pl.DataFrame(all_rows)
    results_path = args.output_dir / "experiment_results.csv"
    results.write_csv(results_path)

    if best_bundle is None:
        raise RuntimeError("No eligible best model found")

    best_row = best_bundle["row"]
    best_model_path = args.output_dir / "best_model.pkl"
    best_metadata_path = args.output_dir / "best_model_metadata.json"
    with best_model_path.open("wb") as model_file:
        pickle.dump(best_bundle, model_file)

    best_metadata = {
        "data_path": str(data_path),
        "selection_metric": "best_avg_pnl_after_current_spread",
        "min_coverage": args.min_coverage,
        "best": best_row,
        "feature_cols": best_bundle["feature_cols"],
        "target_col": best_bundle["target_col"],
    }
    best_metadata_path.write_text(json.dumps(best_metadata, indent=2))
    write_summary(args.output_dir, results, best_row, best_model_path)

    print(f"\nSaved results to {results_path}")
    print(f"Saved best model to {best_model_path}")
    print(f"Saved best metadata to {best_metadata_path}")
    print(f"Saved summary to {args.output_dir / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
