from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from models.kalshi_timeseries_experiments import (
    clean_feature_frame,
    feature_columns,
    trading_metrics,
)
from models.kalshi_timeseries_model import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    RidgeARXRegressor,
    add_future_target,
    add_past_only_features,
    chronological_event_split,
    evaluate,
    read_final_data,
    resolve_data_path,
)

try:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, SGDRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, StandardScaler
except ModuleNotFoundError:
    ExtraTreesRegressor = None
    HistGradientBoostingRegressor = None
    RandomForestRegressor = None
    SimpleImputer = None
    ElasticNet = None
    SGDRegressor = None
    MLPRegressor = None
    Pipeline = None
    RobustScaler = None
    StandardScaler = None


OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "model_family_sweep"


@dataclass(frozen=True)
class FamilySpec:
    name: str
    family: str
    target_mode: str
    feature_group: str
    params: dict[str, Any]
    max_train_rows: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare time-series model families with realized executable PnL."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        choices=["coarse", "fine", "all"],
        default="all",
        help="Run the broad family comparison, the winner fine-tune, or both.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--coarse-horizons",
        type=int,
        nargs="+",
        default=[10, 30, 60, 120, 300],
    )
    parser.add_argument(
        "--fine-horizons",
        type=int,
        nargs="+",
        default=[30, 60, 120, 300],
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2],
    )
    parser.add_argument("--min-coverage", type=float, default=0.01)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def require_sklearn() -> None:
    if Pipeline is None:
        raise ModuleNotFoundError("scikit-learn is required for model family experiments")


def coarse_specs() -> list[FamilySpec]:
    specs = [
        FamilySpec("ridge_delta_all_alpha_10", "ridge", "delta", "all", {"alpha": 10.0}),
        FamilySpec("ridge_price_all_alpha_10", "ridge", "price", "all", {"alpha": 10.0}),
        FamilySpec("ridge_delta_cross_alpha_1000", "ridge", "delta", "kalshi_cross_coin", {"alpha": 1000.0}),
    ]
    if Pipeline is None:
        return specs

    # EDA notes: features are heavy-tailed and correlated, so include robust linear
    # models plus nonlinear tree models that can pick up threshold effects.
    specs.extend(
        [
            FamilySpec(
                "elastic_net_delta_all",
                "elastic_net",
                "delta",
                "all",
                {"alpha": 0.0005, "l1_ratio": 0.15},
                max_train_rows=300_000,
            ),
            FamilySpec(
                "sgd_huber_delta_all",
                "sgd_huber",
                "delta",
                "all",
                {"alpha": 0.0001, "epsilon": 0.01},
                max_train_rows=400_000,
            ),
            FamilySpec(
                "hist_gbr_delta_all",
                "hist_gbr",
                "delta",
                "all",
                {"max_iter": 160, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.1},
                max_train_rows=300_000,
            ),
            FamilySpec(
                "hist_gbr_price_all",
                "hist_gbr",
                "price",
                "all",
                {"max_iter": 160, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 0.1},
                max_train_rows=300_000,
            ),
            FamilySpec(
                "extra_trees_delta_all",
                "extra_trees",
                "delta",
                "all",
                {"n_estimators": 80, "max_depth": 12, "min_samples_leaf": 50},
                max_train_rows=160_000,
            ),
            FamilySpec(
                "random_forest_delta_all",
                "random_forest",
                "delta",
                "all",
                {"n_estimators": 60, "max_depth": 12, "min_samples_leaf": 50},
                max_train_rows=120_000,
            ),
            FamilySpec(
                "mlp_delta_all",
                "mlp",
                "delta",
                "all",
                {"hidden_layer_sizes": (48, 24), "alpha": 0.0001, "learning_rate_init": 0.001},
                max_train_rows=160_000,
            ),
        ]
    )
    return specs


def fine_specs(winning_family: str) -> list[FamilySpec]:
    if winning_family == "hist_gbr":
        return [
            FamilySpec(
                f"hist_gbr_{mode}_{group}_iter{max_iter}_lr{lr}_leaf{leaf}_l2{l2}",
                "hist_gbr",
                mode,
                group,
                {
                    "max_iter": max_iter,
                    "learning_rate": lr,
                    "max_leaf_nodes": leaf,
                    "l2_regularization": l2,
                },
                max_train_rows=500_000,
            )
            for mode in ["delta", "price"]
            for group in ["all"]
            for max_iter in [160, 320]
            for lr in [0.03, 0.06]
            for leaf in [15, 31]
            for l2 in [0.1]
        ]

    if winning_family == "extra_trees":
        return [
            FamilySpec(
                f"extra_trees_{mode}_{group}_trees{trees}_depth{depth}_leaf{leaf}",
                "extra_trees",
                mode,
                group,
                {"n_estimators": trees, "max_depth": depth, "min_samples_leaf": leaf},
                max_train_rows=250_000,
            )
            for mode in ["delta", "price"]
            for group in ["kalshi_spot", "kalshi_cross_coin", "all"]
            for trees in [100, 180]
            for depth in [10, 16, None]
            for leaf in [20, 80]
        ]

    if winning_family == "elastic_net":
        return [
            FamilySpec(
                f"elastic_net_{mode}_{group}_alpha{alpha}_l1{l1}",
                "elastic_net",
                mode,
                group,
                {"alpha": alpha, "l1_ratio": l1},
                max_train_rows=500_000,
            )
            for mode in ["delta", "price"]
            for group in ["kalshi_spot", "kalshi_cross_coin", "all"]
            for alpha in [0.0001, 0.0005, 0.001, 0.005]
            for l1 in [0.05, 0.15, 0.5]
        ]

    return [
        FamilySpec(
            f"ridge_{mode}_{group}_alpha{alpha:g}",
            "ridge",
            mode,
            group,
            {"alpha": alpha},
        )
        for mode in ["delta", "price"]
        for group in ["kalshi_core", "kalshi_spot", "kalshi_cross_coin", "all"]
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0]
    ]


def sample_training_rows(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if max_rows is None or len(y) <= max_rows:
        return x, y, len(y)

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(y), size=max_rows, replace=False))
    return x[idx], y[idx], len(idx)


def build_estimator(spec: FamilySpec):
    if spec.family == "ridge":
        return RidgeARXRegressor(alpha=float(spec.params["alpha"]))

    require_sklearn()
    if spec.family == "elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(quantile_range=(5, 95))),
                (
                    "model",
                    ElasticNet(
                        alpha=float(spec.params["alpha"]),
                        l1_ratio=float(spec.params["l1_ratio"]),
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        )

    if spec.family == "sgd_huber":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(quantile_range=(5, 95))),
                (
                    "model",
                    SGDRegressor(
                        loss="huber",
                        penalty="elasticnet",
                        alpha=float(spec.params["alpha"]),
                        epsilon=float(spec.params["epsilon"]),
                        max_iter=1500,
                        tol=1e-4,
                        random_state=42,
                        early_stopping=True,
                    ),
                ),
            ]
        )

    if spec.family == "hist_gbr":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            **spec.params,
        )

    if spec.family == "extra_trees":
        return ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_samples=0.8,
            **spec.params,
        )

    if spec.family == "random_forest":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_samples=0.8,
            **spec.params,
        )

    if spec.family == "mlp":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        random_state=42,
                        max_iter=80,
                        early_stopping=True,
                        validation_fraction=0.1,
                        **spec.params,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown family: {spec.family}")


def run_spec(
    spec: FamilySpec,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    target_col: str,
    horizon: int,
    thresholds: list[float],
    min_coverage: float,
    seed: int,
) -> tuple[dict[str, Any], object, list[str]]:
    started = time.monotonic()
    feature_cols = feature_columns(train_df, target_col, spec.feature_group)
    x_train, train_target, train_current, _, _, _, _ = clean_feature_frame(
        train_df,
        feature_cols,
        target_col,
    )
    x_test, actual, current, yes_ask, no_ask, future_yes_bid, future_no_bid = clean_feature_frame(
        test_df,
        feature_cols,
        target_col,
    )

    y_train = train_target - train_current if spec.target_mode == "delta" else train_target
    x_fit, y_fit, n_fit_rows = sample_training_rows(x_train, y_train, spec.max_train_rows, seed)

    estimator = build_estimator(spec)
    estimator.fit(x_fit, y_fit)
    raw_pred = estimator.predict(x_test)
    pred = current + raw_pred if spec.target_mode == "delta" else raw_pred
    pred = np.clip(pred, 0.0, 1.0)

    regression = evaluate(actual, pred)
    trading = trading_metrics(
        actual,
        pred,
        current,
        yes_ask,
        no_ask,
        future_yes_bid,
        future_no_bid,
        thresholds,
        min_coverage,
    )
    elapsed = time.monotonic() - started

    row = {
        "horizon_seconds": horizon,
        "experiment": spec.name,
        "family": spec.family,
        "target_mode": spec.target_mode,
        "feature_group": spec.feature_group,
        "params_json": json.dumps(spec.params, sort_keys=True),
        "n_features": len(feature_cols),
        "n_train_rows": int(len(y_train)),
        "n_fit_rows": int(n_fit_rows),
        "n_test_rows": int(len(actual)),
        "elapsed_seconds": round(elapsed, 3),
        **regression,
        **trading,
    }
    return row, estimator, feature_cols


def run_grid(
    base_df: pl.DataFrame,
    horizons: list[int],
    specs: list[FamilySpec],
    thresholds: list[float],
    min_coverage: float,
    test_size: float,
    seed: int,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_bundle: dict[str, Any] = {}

    for horizon in horizons:
        print(f"\n=== model-family horizon {horizon}s ===")
        target_col = f"yes_mid_dollars_t_plus_{horizon}s"
        horizon_df = add_future_target(base_df, horizon)
        train_df, test_df = chronological_event_split(horizon_df, test_size)

        for spec_idx, spec in enumerate(specs):
            row, estimator, feature_cols = run_spec(
                spec,
                train_df,
                test_df,
                target_col,
                horizon,
                thresholds,
                min_coverage,
                seed + spec_idx + horizon,
            )
            rows.append(row)
            print(
                f"{spec.name}: rmse={row['rmse']:.4f}, "
                f"realized_pnl={row['best_avg_realized_pnl']:.5f}, "
                f"cov={row['best_coverage_by_realized_pnl']:.3f}, "
                f"thr={row['best_threshold_by_realized_pnl']}"
            )

            score = row["best_avg_realized_pnl"]
            if row["best_coverage_by_realized_pnl"] >= min_coverage and score > best_score:
                best_score = score
                best_bundle = {
                    "row": row,
                    "model": estimator,
                    "feature_cols": feature_cols,
                    "target_col": target_col,
                }

    return pl.DataFrame(rows), best_bundle


def write_outputs(
    output_dir: Path,
    stage: str,
    results: pl.DataFrame,
    best_bundle: dict[str, Any],
    data_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{stage}_results.csv"
    results.write_csv(results_path)

    if not best_bundle:
        raise RuntimeError(f"No eligible model found for {stage}")

    best_path = output_dir / f"{stage}_best_model.pkl"
    metadata_path = output_dir / f"{stage}_best_metadata.json"
    with best_path.open("wb") as model_file:
        pickle.dump(best_bundle, model_file)

    best_metadata = {
        "data_path": str(data_path),
        "stage": stage,
        "selection_metric": "best_avg_realized_pnl",
        "pnl_method": "enter current ask, exit future bid at forecast horizon",
        "best": best_bundle["row"],
        "feature_cols": best_bundle["feature_cols"],
        "target_col": best_bundle["target_col"],
    }
    metadata_path.write_text(json.dumps(best_metadata, indent=2))

    top_cols = [
        "horizon_seconds",
        "experiment",
        "family",
        "target_mode",
        "feature_group",
        "rmse",
        "mae",
        "directional_accuracy",
        "best_threshold_by_realized_pnl",
        "best_coverage_by_realized_pnl",
        "best_hit_rate_by_realized_pnl",
        "best_avg_realized_pnl",
        "best_total_realized_pnl",
    ]
    top = results.sort(
        ["best_avg_realized_pnl", "best_coverage_by_realized_pnl"],
        descending=[True, True],
    ).head(25)
    lines = [
        f"# Kalshi Model Family {stage.title()} Results",
        "",
        "Selection metric: average realized executable PnL per 1-contract covered trade.",
        "",
        "PnL method: enter at current ask, exit at future bid.",
        "",
        "EDA-informed choices: include spot/cross-coin features, use robust linear models for heavy-tailed features,",
        "and compare nonlinear tree models for threshold/interactions.",
        "",
        "## Best",
        "",
    ]
    for key, value in best_bundle["row"].items():
        if key in top_cols or key in {"params_json", "best_yes_trade_count", "best_no_trade_count"}:
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top 25", ""])
    lines.append("| " + " | ".join(top_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(top_cols)) + " |")
    for row in top.select(top_cols).iter_rows(named=True):
        vals = [f"{row[col]:.6f}" if isinstance(row[col], float) else str(row[col]) for col in top_cols]
        lines.append("| " + " | ".join(vals) + " |")
    (output_dir / f"{stage}_summary.md").write_text("\n".join(lines))

    print(f"Saved {stage} results to {results_path}")
    print(f"Saved {stage} best model to {best_path}")
    print(f"Saved {stage} metadata to {metadata_path}")


def best_family_from_results(results: pl.DataFrame) -> str:
    best = (
        results.sort(
            ["best_avg_realized_pnl", "best_coverage_by_realized_pnl"],
            descending=[True, True],
        )
        .head(1)
        .to_dicts()[0]
    )
    return str(best["family"])


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_df = add_past_only_features(read_final_data(data_path))

    coarse_results = None
    winning_family = "ridge"
    if args.stage in {"coarse", "all"}:
        coarse_results, coarse_best = run_grid(
            base_df,
            args.coarse_horizons,
            coarse_specs(),
            args.thresholds,
            args.min_coverage,
            args.test_size,
            args.sample_seed,
        )
        write_outputs(args.output_dir, "coarse", coarse_results, coarse_best, data_path)
        winning_family = best_family_from_results(coarse_results)

    if args.stage in {"fine", "all"}:
        if args.stage == "fine":
            coarse_path = args.output_dir / "coarse_results.csv"
            if coarse_path.exists():
                coarse_results = pl.read_csv(coarse_path)
                winning_family = best_family_from_results(coarse_results)
        fine_results, fine_best = run_grid(
            base_df,
            args.fine_horizons,
            fine_specs(winning_family),
            args.thresholds,
            args.min_coverage,
            args.test_size,
            args.sample_seed + 10_000,
        )
        write_outputs(args.output_dir, "fine", fine_results, fine_best, data_path)


if __name__ == "__main__":
    main()
