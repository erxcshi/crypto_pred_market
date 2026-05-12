"""
Save Bayesian model artifacts from the notebook into a .npz file for the trader.

Add this cell at the END of your notebook and run it once after MCMC finishes:

    from trading.save_model import save_model_artifacts
    save_model_artifacts(
        posterior_draws=posterior_draws,   # (10000, 65) from metropolis_sample()
        feature_means=feature_means,       # (64,) computed from X_raw_train
        feature_stds=feature_stds,         # (64,) computed from X_raw_train
        feature_names=list(feature_names), # list of 64 column names from X_train_df
    )

Or run this script directly pointing at a previously saved npz that has the raw
arrays (e.g. for inspection):

    python -m trading.save_model --inspect
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from trading.config import MODEL_ARTIFACTS_PATH
from trading.model import FEATURE_NAMES, N_FEATURES


def save_model_artifacts(
    posterior_draws: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    feature_names: list[str],
    output_path: Path | str = MODEL_ARTIFACTS_PATH,
) -> None:
    """
    Validate shapes and feature order, then write model_artifacts.npz.

    Called from inside the notebook after MCMC.
    """
    output_path = Path(output_path)

    # --- shape checks ---
    assert posterior_draws.ndim == 2 and posterior_draws.shape[1] == N_FEATURES + 1, (
        f"posterior_draws must be (n_draws, {N_FEATURES + 1}). "
        f"Got {posterior_draws.shape}. Did you run metropolis_sample() with the full design matrix?"
    )
    assert feature_means.shape == (N_FEATURES,), (
        f"feature_means must be ({N_FEATURES},). Got {feature_means.shape}."
    )
    assert feature_stds.shape == (N_FEATURES,), (
        f"feature_stds must be ({N_FEATURES},). Got {feature_stds.shape}."
    )

    # --- feature order check ---
    if list(feature_names) != FEATURE_NAMES:
        mismatches = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(feature_names, FEATURE_NAMES))
            if a != b
        ]
        if mismatches:
            msg = "\n".join(f"  [{i}] notebook={a!r}  trader={b!r}" for i, a, b in mismatches[:5])
            raise ValueError(
                f"Feature name mismatch between notebook and trader config:\n{msg}\n"
                "Update FEATURE_NAMES in trading/model.py to match your notebook."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        posterior_draws=posterior_draws,
        feature_means=feature_means,
        feature_stds=feature_stds,
    )
    print(f"Model artifacts saved to {output_path}")
    print(f"  posterior_draws: {posterior_draws.shape}")
    print(f"  feature_means:   {feature_means.shape}")
    print(f"  feature_stds:    {feature_stds.shape}")
    print(f"  acceptance rate (if known): use sampler_stats['acceptance_rate']")


def inspect(path: Path | str = MODEL_ARTIFACTS_PATH) -> None:
    data = np.load(path)
    draws = data["posterior_draws"]
    means = data["feature_means"]
    stds = data["feature_stds"]
    print(f"File: {path}")
    print(f"  posterior_draws: {draws.shape}  (dtype={draws.dtype})")
    print(f"  feature_means:   {means.shape}")
    print(f"  feature_stds:    {stds.shape}")
    print(f"\nIntercept (mean ± sd): {draws[:, 0].mean():.4f} ± {draws[:, 0].std():.4f}")
    print(f"\nTop 5 features by |mean weight|:")
    import pandas as pd
    summary = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "mean": draws[:, 1:].mean(axis=0),
        "sd": draws[:, 1:].std(axis=0),
    })
    summary["abs_mean"] = summary["mean"].abs()
    top = summary.nlargest(5, "abs_mean")[["feature", "mean", "sd"]]
    print(top.to_string(index=False))


if __name__ == "__main__":
    if "--inspect" in sys.argv:
        inspect()
    else:
        print(__doc__)
