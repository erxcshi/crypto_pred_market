"""
Bayesian logistic regression model — inference only.

Load artifacts saved by save_model.py, then call predict() per observation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from trading.config import MODEL_ARTIFACTS_PATH


FEATURE_NAMES: list[str] = [
    "time_to_close",
    "yes_spread_dollars",
    "distance_from_strike",
    "yes_mid_change_1s",
    "yes_mid_change_5s",
    "yes_mid_change_std_30s",
    "yes_mid_change_std_60s",
    "yes_spread_mean_30s",
    "ETH_last_price_dollars",
    "ETH_yes_mid_dollars",
    "ETH_yes_spread_dollars",
    "ETH_distance_from_strike",
    "ETH_yes_mid_change_1s",
    "ETH_yes_mid_change_5s",
    "ETH_yes_mid_change_std_30s",
    "ETH_yes_mid_change_std_60s",
    "ETH_yes_spread_mean_30s",
    "ETH_yes_ask_dollars",
    "ETH_yes_bid_dollars",
    "ETH_no_ask_dollars",
    "ETH_no_bid_dollars",
    "XRP_last_price_dollars",
    "XRP_yes_mid_dollars",
    "XRP_yes_spread_dollars",
    "XRP_distance_from_strike",
    "XRP_yes_mid_change_1s",
    "XRP_yes_mid_change_5s",
    "XRP_yes_mid_change_std_30s",
    "XRP_yes_mid_change_std_60s",
    "XRP_yes_spread_mean_30s",
    "XRP_yes_ask_dollars",
    "XRP_yes_bid_dollars",
    "XRP_no_ask_dollars",
    "XRP_no_bid_dollars",
    "SOL_last_price_dollars",
    "SOL_yes_mid_dollars",
    "SOL_yes_spread_dollars",
    "SOL_distance_from_strike",
    "SOL_yes_mid_change_1s",
    "SOL_yes_mid_change_5s",
    "SOL_yes_mid_change_std_30s",
    "SOL_yes_mid_change_std_60s",
    "SOL_yes_spread_mean_30s",
    "SOL_yes_ask_dollars",
    "SOL_yes_bid_dollars",
    "SOL_no_ask_dollars",
    "SOL_no_bid_dollars",
    "btc_spot_price",
    "btc_spot_size_1s",
    "btc_spot_signed_size_1s",
    "btc_spot_return_1s",
    "btc_spot_return_5s",
    "btc_spot_return_15s",
    "btc_spot_return_60s",
    "btc_spot_return_vol_30s",
    "btc_spot_return_vol_5m",
    "btc_spot_signed_flow_mean_30s",
    "btc_spot_size_mean_30s",
]

N_FEATURES = len(FEATURE_NAMES)  # 58 (matches X_train.columns from log_reg.ipynb)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class BayesLogRegModel:
    """Vectorised posterior-predictive inference over MCMC draws."""

    def __init__(
        self,
        posterior_draws: np.ndarray,   # shape (n_draws, n_features + 1)
        feature_means: np.ndarray,     # shape (n_features,)
        feature_stds: np.ndarray,      # shape (n_features,)
    ) -> None:
        assert posterior_draws.shape[1] == N_FEATURES + 1, (
            f"Expected {N_FEATURES + 1} cols (intercept + features), "
            f"got {posterior_draws.shape[1]}"
        )
        self._draws = posterior_draws          # (n_draws, 65)
        self._means = feature_means            # (64,)
        self._stds = np.where(feature_stds == 0, 1.0, feature_stds)  # avoid /0

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str = MODEL_ARTIFACTS_PATH) -> "BayesLogRegModel":
        data = np.load(path)
        return cls(
            posterior_draws=data["posterior_draws"],
            feature_means=data["feature_means"],
            feature_stds=data["feature_stds"],
        )

    # ------------------------------------------------------------------
    def predict(self, features: dict[str, float]) -> tuple[float, float]:
        """
        Returns (p_hat, p_std) for a single observation.

        p_hat  — posterior mean probability of YES outcome
        p_std  — posterior std (use as uncertainty filter)
        """
        x_raw = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float64)
        x_scaled = (x_raw - self._means) / self._stds
        x = np.concatenate([[1.0], x_scaled])               # (65,)
        p_samples = _sigmoid(self._draws @ x)               # (n_draws,)
        return float(p_samples.mean()), float(p_samples.std())

    def predict_batch(self, feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """feature_matrix: (n_obs, 64) raw (unscaled) values."""
        x_scaled = (feature_matrix - self._means) / self._stds
        X = np.column_stack([np.ones(len(x_scaled)), x_scaled])   # (n, 65)
        p_samples = _sigmoid(X @ self._draws.T)                    # (n, n_draws)
        return p_samples.mean(axis=1), p_samples.std(axis=1)
