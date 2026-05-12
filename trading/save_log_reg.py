"""
Save sklearn LogisticRegression artifacts into model_artifacts.npz format.

After training in log_reg.ipynb, run:

    from trading.save_log_reg import save_log_reg_artifacts
    save_log_reg_artifacts(log_reg, X_train.columns)

This stores the fitted coefficients as a single-draw "posterior" so the existing
trading.model.BayesLogRegModel inference code can load it unchanged (p_std will
be 0 — the SIGMA filter effectively disables itself).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from trading.config import MODEL_ARTIFACTS_PATH
from trading.model import FEATURE_NAMES, N_FEATURES


def save_log_reg_artifacts(
    pipeline,
    feature_names,
    output_path: Path | str = MODEL_ARTIFACTS_PATH,
) -> None:
    """
    pipeline: fitted sklearn Pipeline(StandardScaler + LogisticRegression)
    feature_names: iterable of column names from X_train (any sequence type)
    """
    output_path = Path(output_path)

    scaler = pipeline.named_steps["standardscaler"]
    logit  = pipeline.named_steps["logisticregression"]

    feature_names = list(feature_names)
    n = len(feature_names)

    # Feature-order check: must exactly match FEATURE_NAMES
    if feature_names != FEATURE_NAMES:
        mismatches = [
            (i, a, b) for i, (a, b) in enumerate(zip(feature_names, FEATURE_NAMES)) if a != b
        ]
        extra = set(feature_names) - set(FEATURE_NAMES)
        missing = set(FEATURE_NAMES) - set(feature_names)
        msg = [
            f"Feature mismatch between notebook ({n}) and trader FEATURE_NAMES ({N_FEATURES}).",
            f"First 5 positional diffs: {mismatches[:5]}",
            f"In notebook but not trader: {sorted(extra)[:10]}",
            f"In trader but not notebook: {sorted(missing)[:10]}",
            "Fix: update FEATURE_NAMES in trading/model.py to match training columns.",
        ]
        raise ValueError("\n".join(msg))

    coef = logit.coef_[0].astype(np.float64)        # (n_features,)
    intercept = float(logit.intercept_[0])
    draws = np.concatenate([[intercept], coef]).reshape(1, -1)  # (1, n_features+1)

    means = scaler.mean_.astype(np.float64)
    stds  = scaler.scale_.astype(np.float64)

    assert draws.shape == (1, N_FEATURES + 1), f"draws shape {draws.shape}"
    assert means.shape == (N_FEATURES,), f"means shape {means.shape}"
    assert stds.shape == (N_FEATURES,), f"stds shape {stds.shape}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        posterior_draws=draws,
        feature_means=means,
        feature_stds=stds,
    )
    print(f"Saved log_reg artifacts to {output_path}")
    print(f"  draws:  {draws.shape}  (1 draw = deterministic prediction, p_std=0)")
    print(f"  means:  {means.shape}")
    print(f"  stds:   {stds.shape}")
    print(f"  intercept: {intercept:+.4f}")
    print(f"  |coef| max={np.abs(coef).max():.4f}  mean={np.abs(coef).mean():.4f}")
