"""
postmortem.py - score the pre-tournament model against what actually happened
at Wimbledon 2026.

The tournament finished on 12 July 2026, so the forecast is now falsifiable.
This scores it: match-level log-loss and Brier against the pairwise matrix
frozen before round one, a calibration table, and the championship probability
the model had assigned to the eventual winner.

Inputs:
    results/pretournament.json          written by wimbledon_rf.py
    data/wimbledon_2026_results.json    actual results, one entry per match:
        [{"round": 1, "winner": "SINNER, Jannik",
          "loser": "KECMANOVIC, Miomir"}, ...]
        Names must match wimbledon_2026_draw.json exactly.

Usage:
    python postmortem.py
    python postmortem.py --results data/wimbledon_2026_results.json
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

from wimbledon_rf import data_dir, results_dir, SEED

PRE_PATH = os.path.join(results_dir, "pretournament.json")
DEFAULT_RESULTS = os.path.join(data_dir, "wimbledon_2026_results.json")

ROUND_NAMES = {1: "R128", 2: "R64", 3: "R32", 4: "R16",
               5: "QF", 6: "SF", 7: "F"}


def load_pretournament(path):
    with open(path) as f:
        pre = json.load(f)
    idx = {n: i for i, n in enumerate(pre["names"])}
    return np.array(pre["pairwise"]), idx, pre["championship_probs"], pre


def score_matches(results, P, idx, seed=SEED):
    """
    One row per match. Orientation is randomised with a fixed seed so the
    labels are not all ones, which would make log-loss degenerate.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for m in results:
        w, l = m["winner"], m["loser"]
        if w not in idx or l not in idx:
            raise KeyError(f"{w!r} or {l!r} not in the pre-tournament draw.")
        p_winner = float(P[idx[w], idx[l]])
        flip = rng.random() < 0.5
        rows.append({
            "round": m.get("round"),
            "winner": w,
            "loser": l,
            "p_winner": p_winner,
            "label": 0 if flip else 1,
            "p": (1 - p_winner) if flip else p_winner,
        })
    return pd.DataFrame(rows)


def calibration_table(df, bins=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    """Predicted versus realised frequency, the check the README flagged as absent."""
    cut = pd.cut(df["p"], bins=bins, include_lowest=True)
    tab = df.groupby(cut, observed=True).agg(
        n=("p", "size"), predicted=("p", "mean"), realised=("label", "mean"))
    tab["gap"] = tab["realised"] - tab["predicted"]
    return tab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--pre", default=PRE_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if not os.path.exists(args.pre):
        raise SystemExit(f"{args.pre} not found. Run wimbledon_rf.py first.")
    if not os.path.exists(args.results):
        raise SystemExit(
            f"{args.results} not found. Create it as a JSON list of\n"
            f'  {{"round": 1, "winner": "...", "loser": "..."}}\n'
            f"using the exact name strings from wimbledon_2026_draw.json.")

    P, idx, champ_probs, pre = load_pretournament(args.pre)
    with open(args.results) as f:
        results = json.load(f)

    df = score_matches(results, P, idx, args.seed)
    y = df["label"].to_numpy()
    ll = log_loss(y, df["p"], labels=[0, 1])
    br = brier_score_loss(y, df["p"])
    acc = ((df["p_winner"] >= 0.5)).mean()

    finals = [m for m in results if m.get("round") == 7]
    champion = finals[0]["winner"] if finals else None
    ranked = sorted(champ_probs.items(), key=lambda kv: -kv[1])
    champ_rank = next((i for i, (n, _) in enumerate(ranked, 1) if n == champion), None)

    print(f"Scored {len(df)} matches from Wimbledon 2026 against the "
          f"pre-tournament model (as of {pre['as_of_date']}).")
    print(f"  log-loss {ll:.4f}   brier {br:.4f}   "
          f"favourite-correct {acc:.1%}")
    if champion:
        print(f"  champion: {champion}")
        print(f"  model had him at {champ_probs.get(champion, 0):.1%} "
              f"(rank {champ_rank} of {len(ranked)})")

    by_round = df.groupby("round").apply(lambda g: pd.Series({
        "n": len(g),
        "logloss": log_loss(g["label"], g["p"], labels=[0, 1]),
        "favourite_correct": (g["p_winner"] >= 0.5).mean(),
    }))
    by_round.index = [ROUND_NAMES.get(i, i) for i in by_round.index]
    print("\nBy round:")
    print(by_round.to_string(float_format=lambda v: f"{v:.4f}"))

    calib = calibration_table(df)
    print("\nCalibration:")
    print(calib.to_string(float_format=lambda v: f"{v:.3f}"))

    biggest = df.assign(surprise=-np.log(df["p_winner"])).nlargest(5, "surprise")
    print("\nBiggest surprises:")
    for _, r in biggest.iterrows():
        print(f"  {ROUND_NAMES.get(r['round'], r['round']):<5} "
              f"{r['winner']} d. {r['loser']}  (model gave winner {r['p_winner']:.1%})")

    write_report(df, ll, br, acc, champion, champ_probs, champ_rank,
                 by_round, calib, biggest, pre)


def write_report(df, ll, br, acc, champion, champ_probs, champ_rank,
                 by_round, calib, biggest, pre):
    os.makedirs(results_dir, exist_ok=True)
    out = ["<!-- generated by postmortem.py; do not edit by hand -->", ""]
    out.append(f"Pre-tournament model frozen as of {pre['as_of_date']}, scored "
               f"against all {len(df)} completed matches.\n")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Log-loss | {ll:.4f} |")
    out.append(f"| Brier | {br:.4f} |")
    out.append(f"| Favourite correct | {acc:.1%} |")
    if champion:
        out.append(f"| Champion | {champion} |")
        out.append(f"| Pre-tournament probability assigned | "
                   f"{champ_probs.get(champion, 0):.1%} (rank {champ_rank}) |")

    out.append("\n### By round\n")
    out.append("| Round | n | Log-loss | Favourite correct |")
    out.append("|---|---|---|---|")
    for rnd, r in by_round.iterrows():
        out.append(f"| {rnd} | {int(r['n'])} | {r['logloss']:.4f} | "
                   f"{r['favourite_correct']:.1%} |")

    out.append("\n### Calibration\n")
    out.append("| Predicted bucket | n | Mean predicted | Realised | Gap |")
    out.append("|---|---|---|---|---|")
    for bucket, r in calib.iterrows():
        out.append(f"| {bucket} | {int(r['n'])} | {r['predicted']:.3f} | "
                   f"{r['realised']:.3f} | {r['gap']:+.3f} |")

    out.append("\n### Biggest surprises\n")
    out.append("| Round | Result | Model probability for winner |")
    out.append("|---|---|---|")
    for _, r in biggest.iterrows():
        out.append(f"| {ROUND_NAMES.get(r['round'], r['round'])} | "
                   f"{r['winner']} d. {r['loser']} | {r['p_winner']:.1%} |")

    path = os.path.join(results_dir, "postmortem_report.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()