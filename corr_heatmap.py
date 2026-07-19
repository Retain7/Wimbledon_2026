"""
Generate a feature-correlation heatmap from the training data used by wimbledon_rf.py.

Run from the repo root:
    python make_corr_heatmap.py

Outputs: correlation_heatmap.png
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wimbledon_rf as W

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "correlation_heatmap.png")


def main():
    matches = W.load_matches(W.matches_path)
    player_stats = W.Player_Stats()
    training_rows = W.build_training_rows(matches, player_stats)
    train_df = pd.DataFrame(training_rows)

    corr = train_df[W.FEATURES].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        annot_kws={"size": 7},
    )
    plt.title("Feature Correlation Matrix (grass-court training rows)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200)
    print(f"Saved heatmap to {OUT_PATH}")


if __name__ == "__main__":
    main()