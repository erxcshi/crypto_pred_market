from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import polars as pl

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    ColumnTransformer = None
    HistGradientBoostingRegressor = None
    SimpleImputer = None
    Pipeline = None
    StandardScaler = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data_gather" / "final_data" / "final_data.csv"
LEGACY_DATA_PATH = PROJECT_ROOT / "final_data" / "final_data.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "artifacts" / "kalshi_timeseries"

EVENT_COLUMNS = ["open_time", "close_time"]
TIME_COLUMNS = ["curr_time", "open_time", "close_time", "prev_time"]
KNOWN_LEAKAGE_COLUMNS = {
    "outcome",
    "next_price_dollars_lead1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a BTC Kalshi YES-contract time-series regressor for the "
            "question: what will yes_mid_dollars cost n seconds from now?"
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--horizon-seconds", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument(
        "--model-type",
        choices=["auto", "hist_gradient_boosting", "ridge_arx"],
        default="auto",
        help=(
            "auto uses sklearn HistGradientBoostingRegressor when installed; "
            "otherwise it falls back to a NumPy ridge autoregressive model."
        ),
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    return parser.parse_args()


def resolve_data_path(path: Path) -> Path:
    if not path.exists():
        if path == DEFAULT_DATA_PATH and LEGACY_DATA_PATH.exists():
            return LEGACY_DATA_PATH
        else:
            raise FileNotFoundError(
                f"Could not find {path}. Build it first with: python -m data_gather.filter"
            )
    return path


def read_final_data(path: Path) -> pl.DataFrame:
    path = resolve_data_path(path)
    return pl.read_csv(
        path,
        try_parse_dates=True,
        infer_schema_length=10000,
    ).sort("curr_time")


def add_future_target(df: pl.DataFrame, horizon_seconds: int) -> pl.DataFrame:
    target_col = f"yes_mid_dollars_t_plus_{horizon_seconds}s"

    missing = [col for col in [*EVENT_COLUMNS, "yes_mid_dollars"] if col not in df.columns]
    if missing:
        raise ValueError(f"final_data.csv is missing required columns: {missing}")

    base = (
        df.sort([*EVENT_COLUMNS, "curr_time"])
        .with_columns(
            (pl.col("curr_time") + pl.duration(seconds=horizon_seconds)).alias("target_time")
        )
    )
    future_prices = (
        df.select(
            [
                *EVENT_COLUMNS,
                pl.col("curr_time").alias("future_time"),
                pl.col("yes_mid_dollars").alias(target_col),
            ]
        )
        .sort([*EVENT_COLUMNS, "future_time"])
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sortedness of columns cannot be checked when 'by' groups provided",
            category=UserWarning,
        )
        return (
            base.join_asof(
                future_prices,
                left_on="target_time",
                right_on="future_time",
                by=EVENT_COLUMNS,
                strategy="forward",
                tolerance="1s",
            )
            .filter(pl.col(target_col).is_not_null())
            .drop(["target_time", "future_time"])
        )


def add_past_only_features(df: pl.DataFrame) -> pl.DataFrame:
    event_cols = [col for col in EVENT_COLUMNS if col in df.columns]

    feature_exprs = []
    if "yes_mid_dollars" in df.columns:
        feature_exprs.extend(
            [
                pl.col("yes_mid_dollars").shift(1).over(event_cols).alias("yes_mid_lag_1s"),
                pl.col("yes_mid_dollars").shift(5).over(event_cols).alias("yes_mid_lag_5s"),
                pl.col("yes_mid_dollars").shift(15).over(event_cols).alias("yes_mid_lag_15s"),
                pl.col("yes_mid_dollars").shift(60).over(event_cols).alias("yes_mid_lag_60s"),
                pl.col("yes_mid_dollars")
                .shift(1)
                .rolling_mean(window_size=15)
                .over(event_cols)
                .alias("yes_mid_mean_prev_15s"),
                pl.col("yes_mid_dollars")
                .shift(1)
                .rolling_mean(window_size=60)
                .over(event_cols)
                .alias("yes_mid_mean_prev_60s"),
            ]
        )

    if "btc_spot_return_1s" in df.columns:
        feature_exprs.extend(
            [
                pl.col("btc_spot_return_1s")
                .shift(1)
                .rolling_mean(window_size=15)
                .over(event_cols)
                .alias("btc_spot_return_mean_prev_15s"),
                pl.col("btc_spot_return_1s")
                .shift(1)
                .rolling_std(window_size=60)
                .over(event_cols)
                .alias("btc_spot_return_std_prev_60s"),
            ]
        )

    if not feature_exprs:
        return df

    return df.with_columns(feature_exprs)


def chronological_event_split(
    df: pl.DataFrame,
    test_size: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    event_cols = [col for col in EVENT_COLUMNS if col in df.columns]
    if len(event_cols) != len(EVENT_COLUMNS):
        split_idx = int(df.height * (1 - test_size))
        return df[:split_idx], df[split_idx:]

    events = (
        df.select(EVENT_COLUMNS)
        .unique()
        .sort(EVENT_COLUMNS)
        .with_row_index("event_idx")
    )
    split_idx = max(1, int(events.height * (1 - test_size)))
    train_events = events[:split_idx].drop("event_idx")
    test_events = events[split_idx:].drop("event_idx")

    if test_events.height == 0:
        raise ValueError("Not enough events to create a non-empty test set")

    train_df = df.join(train_events, on=EVENT_COLUMNS, how="inner")
    test_df = df.join(test_events, on=EVENT_COLUMNS, how="inner")
    return train_df.sort("curr_time"), test_df.sort("curr_time")


def leakage_columns(df: pl.DataFrame, target_col: str) -> list[str]:
    drop_cols = set(TIME_COLUMNS)
    drop_cols.update(KNOWN_LEAKAGE_COLUMNS)
    drop_cols.add(target_col)

    for col in df.columns:
        lowered = col.lower()
        if lowered.startswith("next_") or "lead" in lowered or "t_plus" in lowered:
            drop_cols.add(col)

    return [col for col in df.columns if col in drop_cols]


def make_xy(
    df: pl.DataFrame,
    target_col: str,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if feature_cols is None:
        dropped = set(leakage_columns(df, target_col))
        feature_cols = [
            col
            for col, dtype in zip(df.columns, df.dtypes)
            if col not in dropped and dtype.is_numeric()
        ]

    if not feature_cols:
        raise ValueError("No numeric feature columns available after leakage filtering")

    clean_df = df.select([*feature_cols, target_col]).drop_nulls()
    x = clean_df.select(feature_cols).to_numpy()
    y = clean_df.select(target_col).to_numpy().ravel()
    return x, y, feature_cols


class RidgeARXRegressor:
    """Small dependency-free autoregressive-with-exogenous-features regressor."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.feature_medians_: np.ndarray | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeARXRegressor":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.feature_medians_ = np.nanmedian(x, axis=0)
        x = np.where(np.isnan(x), self.feature_medians_, x)

        self.feature_means_ = x.mean(axis=0)
        self.feature_stds_ = x.std(axis=0)
        self.feature_stds_[self.feature_stds_ == 0] = 1.0
        x_scaled = (x - self.feature_means_) / self.feature_stds_

        design = np.c_[np.ones(x_scaled.shape[0]), x_scaled]
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if (
            self.feature_medians_ is None
            or self.feature_means_ is None
            or self.feature_stds_ is None
            or self.coef_ is None
        ):
            raise RuntimeError("Model must be fitted before predict")

        x = np.asarray(x, dtype=float)
        x = np.where(np.isnan(x), self.feature_medians_, x)
        x_scaled = (x - self.feature_means_) / self.feature_stds_
        design = np.c_[np.ones(x_scaled.shape[0]), x_scaled]
        return design @ self.coef_


def build_model(args: argparse.Namespace, n_features: int):
    use_sklearn = args.model_type in {"auto", "hist_gradient_boosting"}
    if use_sklearn and HistGradientBoostingRegressor is None:
        if args.model_type == "hist_gradient_boosting":
            raise ModuleNotFoundError(
                "scikit-learn is not installed. Install it or use --model-type ridge_arx."
            )
        return RidgeARXRegressor(alpha=args.ridge_alpha)

    if args.model_type == "ridge_arx":
        return RidgeARXRegressor(alpha=args.ridge_alpha)

    numeric_features = list(range(n_features))
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    total_var = float(np.sum((y_true - np.mean(y_true)) ** 2))
    residual_var = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1 - residual_var / total_var if total_var > 0 else 0.0
    baseline = np.full_like(y_true, fill_value=np.mean(y_true), dtype=float)
    baseline_rmse = float(np.sqrt(np.mean((y_true - baseline) ** 2)))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "baseline_mean_rmse": baseline_rmse,
        "rmse_vs_baseline_improvement": float(1 - rmse / baseline_rmse)
        if baseline_rmse > 0
        else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_col = f"yes_mid_dollars_t_plus_{args.horizon_seconds}s"

    data_path = resolve_data_path(args.data_path)
    df = read_final_data(data_path)
    df = add_future_target(df, args.horizon_seconds)
    df = add_past_only_features(df)

    train_df, test_df = chronological_event_split(df, args.test_size)
    x_train, y_train, feature_cols = make_xy(train_df, target_col)
    x_test, y_test, _ = make_xy(test_df, target_col, feature_cols)

    model = build_model(args, n_features=len(feature_cols))
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = evaluate(y_test, predictions)

    metrics_payload = {
        "target": target_col,
        "horizon_seconds": args.horizon_seconds,
        "data_path": str(data_path),
        "n_rows": df.height,
        "n_train_rows": int(len(y_train)),
        "n_test_rows": int(len(y_test)),
        "n_features": len(feature_cols),
        "model_type": type(model).__name__,
        "features": feature_cols,
        "metrics": metrics,
    }

    metrics_path = args.output_dir / f"metrics_{args.horizon_seconds}s.json"
    model_path = args.output_dir / f"model_{args.horizon_seconds}s.pkl"
    predictions_path = args.output_dir / f"predictions_{args.horizon_seconds}s.csv"

    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    with model_path.open("wb") as model_file:
        pickle.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "horizon_seconds": args.horizon_seconds,
            },
            model_file,
        )

    pl.DataFrame(
        {
            "actual": y_test,
            "predicted": predictions,
            "error": predictions - y_test,
        }
    ).write_csv(predictions_path)

    print(json.dumps(metrics_payload, indent=2))
    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved predictions to {predictions_path}")


if __name__ == "__main__":
    main()
