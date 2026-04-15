from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ARTIFACT_DIR = Path("models/artifacts/kalshi_timeseries/model_family_sweep")
OLD_RIDGE_DIR = Path("models/artifacts/kalshi_timeseries/experiments_extended_thresholds")
FIG_DIR = ARTIFACT_DIR / "presentation_figures"


def cents(x: pd.Series) -> pd.Series:
    return x * 100.0


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def annotate_bars(ax) -> None:
    for patch in ax.patches:
        value = patch.get_width()
        ax.text(
            value,
            patch.get_y() + patch.get_height() / 2,
            f" {value:.1f}c",
            va="center",
            fontsize=9,
        )


def plot_coarse_family_comparison(coarse: pd.DataFrame) -> None:
    best_by_family = (
        coarse.sort_values("best_avg_realized_pnl", ascending=False)
        .groupby("family", as_index=False)
        .first()
        .sort_values("best_avg_realized_pnl", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(best_by_family["family"], cents(best_by_family["best_avg_realized_pnl"]), color="#2A9D8F")
    ax.set_title("Best Realized PnL by Model Family")
    ax.set_xlabel("Average realized PnL per 1-contract trade (cents)")
    ax.set_ylabel("Model family")
    annotate_bars(ax)
    savefig("01_best_pnl_by_model_family.png")


def plot_top_fine_models(fine: pd.DataFrame) -> None:
    top = fine.sort_values("best_avg_realized_pnl", ascending=False).head(10).copy()
    top = top.sort_values("best_avg_realized_pnl", ascending=True)
    labels = top["experiment"].str.replace("hist_gbr_", "", regex=False).str.replace("_all_", "\n", regex=False)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(labels, cents(top["best_avg_realized_pnl"]), color="#457B9D")
    ax.set_title("Top Fine-Tuned Histogram Gradient Boosting Models")
    ax.set_xlabel("Average realized PnL per 1-contract trade (cents)")
    ax.set_ylabel("")
    annotate_bars(ax)
    savefig("02_top_fine_models.png")


def plot_edge_vs_coverage(coarse: pd.DataFrame, fine: pd.DataFrame) -> None:
    combined = pd.concat(
        [
            coarse.assign(stage="coarse"),
            fine.assign(stage="fine"),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {
        "ridge": "#E76F51",
        "elastic_net": "#F4A261",
        "sgd_huber": "#E9C46A",
        "hist_gbr": "#2A9D8F",
        "extra_trees": "#457B9D",
        "random_forest": "#1D3557",
        "mlp": "#6D597A",
    }
    for family, grp in combined.groupby("family"):
        ax.scatter(
            grp["best_coverage_by_realized_pnl"] * 100,
            cents(grp["best_avg_realized_pnl"]),
            s=70,
            alpha=0.75,
            label=family,
            color=colors.get(family),
        )
    ax.set_title("Tradeoff: Edge vs Coverage")
    ax.set_xlabel("Coverage (% of test rows traded)")
    ax.set_ylabel("Average realized PnL per trade (cents)")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="1% min coverage")
    ax.legend(ncol=2, fontsize=8)
    savefig("03_edge_vs_coverage.png")


def plot_horizon_comparison(coarse: pd.DataFrame) -> None:
    best_by_horizon = (
        coarse.sort_values("best_avg_realized_pnl", ascending=False)
        .groupby("horizon_seconds", as_index=False)
        .first()
        .sort_values("horizon_seconds")
    )

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(
        best_by_horizon["horizon_seconds"],
        cents(best_by_horizon["best_avg_realized_pnl"]),
        marker="o",
        color="#2A9D8F",
        label="Avg PnL",
    )
    ax2.plot(
        best_by_horizon["horizon_seconds"],
        best_by_horizon["best_coverage_by_realized_pnl"] * 100,
        marker="s",
        color="#E76F51",
        label="Coverage",
    )
    ax1.set_title("Best Model at Each Forecast Horizon")
    ax1.set_xlabel("Forecast horizon (seconds)")
    ax1.set_ylabel("Average realized PnL per trade (cents)", color="#2A9D8F")
    ax2.set_ylabel("Coverage (% test rows)", color="#E76F51")
    ax1.set_xticks(best_by_horizon["horizon_seconds"])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    savefig("04_best_by_horizon.png")


def plot_ridge_vs_winner(fine: pd.DataFrame, ridge: pd.DataFrame) -> None:
    winner = fine.sort_values("best_avg_realized_pnl", ascending=False).iloc[0]
    ridge_60 = ridge[
        (ridge["horizon_seconds"] == 60)
        & (ridge["experiment"] == "ridge_price_all_alpha_10")
    ].iloc[0]

    compare = pd.DataFrame(
        [
            {
                "model": "Old ridge\n60s",
                "avg_pnl_cents": ridge_60["best_avg_realized_pnl"] * 100,
                "coverage_pct": ridge_60["best_coverage_by_realized_pnl"] * 100,
                "hit_rate_pct": ridge_60["best_hit_rate_by_realized_pnl"] * 100,
            },
            {
                "model": "New HistGBR\n60s",
                "avg_pnl_cents": winner["best_avg_realized_pnl"] * 100,
                "coverage_pct": winner["best_coverage_by_realized_pnl"] * 100,
                "hit_rate_pct": winner["best_hit_rate_by_realized_pnl"] * 100,
            },
        ]
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("avg_pnl_cents", "Avg PnL\n(cents)", "#2A9D8F"),
        ("coverage_pct", "Coverage\n(%)", "#457B9D"),
        ("hit_rate_pct", "Hit Rate\n(%)", "#E76F51"),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        ax.bar(compare["model"], compare[col], color=color)
        ax.set_title(title)
        for i, value in enumerate(compare[col]):
            ax.text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Old Ridge Baseline vs New Winning Model", y=1.05)
    savefig("05_old_ridge_vs_new_winner.png")


def plot_threshold_curve(fine: pd.DataFrame) -> None:
    winner = fine.sort_values("best_avg_realized_pnl", ascending=False).iloc[0]
    threshold_cols = [
        col for col in fine.columns if col.startswith("avg_realized_pnl_thr_")
    ]
    suffixes = [col.replace("avg_realized_pnl_thr_", "") for col in threshold_cols]
    thresholds = [float(suffix.replace("p", ".")) for suffix in suffixes]
    avg_pnl = [winner[f"avg_realized_pnl_thr_{suffix}"] * 100 for suffix in suffixes]
    total_pnl = [winner[f"total_realized_pnl_thr_{suffix}"] for suffix in suffixes]
    coverage = [winner[f"coverage_thr_{suffix}"] * 100 for suffix in suffixes]

    order = sorted(range(len(thresholds)), key=lambda i: thresholds[i])
    thresholds = [thresholds[i] for i in order]
    avg_pnl = [avg_pnl[i] for i in order]
    total_pnl = [total_pnl[i] for i in order]
    coverage = [coverage[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(thresholds, avg_pnl, marker="o", color="#2A9D8F")
    axes[0].set_title("Avg PnL by Threshold")
    axes[0].set_ylabel("Cents per trade")
    axes[0].set_xlabel("Predicted move threshold")

    axes[1].plot(thresholds, total_pnl, marker="o", color="#457B9D")
    axes[1].set_title("Total PnL by Threshold")
    axes[1].set_ylabel("1-contract PnL units")
    axes[1].set_xlabel("Predicted move threshold")

    axes[2].plot(thresholds, coverage, marker="o", color="#E76F51")
    axes[2].set_title("Coverage by Threshold")
    axes[2].set_ylabel("% test rows traded")
    axes[2].set_xlabel("Predicted move threshold")

    fig.suptitle("Winning Model Threshold Behavior", y=1.04)
    savefig("06_winner_threshold_curve.png")


def main() -> None:
    coarse = pd.read_csv(ARTIFACT_DIR / "coarse_results.csv")
    fine = pd.read_csv(ARTIFACT_DIR / "fine_results.csv")
    ridge = pd.read_csv(OLD_RIDGE_DIR / "experiment_results.csv")

    plot_coarse_family_comparison(coarse)
    plot_top_fine_models(fine)
    plot_edge_vs_coverage(coarse, fine)
    plot_horizon_comparison(coarse)
    plot_ridge_vs_winner(fine, ridge)
    plot_threshold_curve(fine)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
