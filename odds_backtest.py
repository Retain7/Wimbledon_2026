"""
odds_backtest.py - compare model log-loss against historical market log-loss
on past Wimbledons.

This is the headline evaluation. Unlike the cross-validation in
wimbledon_rf.py, it is strictly chronological: the model is fit once on data
before the backtest window, then walked forward match by match, so no future
information reaches any prediction. The market is a genuine adversary, which
makes this the number worth quoting.

Requires data/odds/{year}.xlsx from tennis-data.co.uk.

Usage:
    python odds_backtest.py --years 2018 2019 2021 2022 2023 2024 2025
"""

import os
import re
import argparse
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

from wimbledon_rf import (
    Player_Stats, load_matches, build_training_rows, train_model,
    apply_feature_medians, FEATURES, matches_path, data_dir, results_dir, SEED,
)

ODDS_DIR = os.path.join(data_dir, "odds")


# ---------------------------------------------------------------------------
# Name reconciliation
# ---------------------------------------------------------------------------
# tennis-data.co.uk writes "Del Potro J."; Sackmann writes "Juan Martin Del
# Potro". Splitting on the last token works for simple names and silently
# fails for every compound surname (del potro, de minaur, bautista agut,
# davidovich fokina, van de zandschulp, carreno busta). Those failures used to
# drop out of the join without a warning, which biased the comparison towards
# whichever players happened to have one-word surnames.
#
# Fix: canonicalise the market side to (surname, first initial), and generate
# every plausible surname split for the Sackmann side, matching on the first
# candidate that exists in the market index. Coverage is then reported so a
# residual join failure is visible instead of silent.
# ---------------------------------------------------------------------------
def _clean(name):
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace(".", " ").replace("-", " ").replace("'", "")
    return re.sub(r"[^a-z ]", " ", s).split()


def market_key(name):
    """'Del Potro J.' -> ('del potro', 'j')"""
    parts = _clean(name)
    if len(parts) < 2:
        return None
    return (" ".join(parts[:-1]), parts[-1][0])


def player_key_candidates(name):
    """
    'Juan Martin Del Potro' -> [('martin del potro', 'j'),
                                ('del potro', 'j'),
                                ('potro', 'j')]
    Longest surname first, so 'del potro' is preferred over 'potro'.
    """
    parts = _clean(name)
    if len(parts) < 2:
        return []
    initial = parts[0][0]
    return [(" ".join(parts[k:]), initial) for k in range(1, len(parts))]


def resolve(name, index):
    for key in player_key_candidates(name):
        if key in index:
            return key
    return None


# ---------------------------------------------------------------------------
# Odds loading
# ---------------------------------------------------------------------------
def load_wimbledon_odds(years):
    frames = []
    for yr in years:
        path = os.path.join(ODDS_DIR, f"{yr}.xlsx")
        if not os.path.exists(path):
            print(f"  [skip] missing {path}")
            continue
        df = pd.read_excel(path)
        df = df[df["Tournament"].str.contains("Wimbledon", case=False, na=False)]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No odds files in data/odds/. Download from tennis-data.co.uk first.")

    odds = pd.concat(frames, ignore_index=True)

    # Prefer Pinnacle (sharpest), fall back to Bet365, then the market average.
    source = None
    for wcol, lcol, tag in [("PSW", "PSL", "pinnacle"),
                            ("B365W", "B365L", "bet365"),
                            ("AvgW", "AvgL", "average")]:
        if wcol in odds.columns:
            odds["odds_w"], odds["odds_l"], source = odds[wcol], odds[lcol], tag
            break
    if source is None:
        raise ValueError("No recognised odds columns found.")
    print(f"  odds source: {source}")

    odds["_wk"] = odds["Winner"].apply(market_key)
    odds["_lk"] = odds["Loser"].apply(market_key)
    odds["_year"] = pd.to_datetime(odds["Date"]).dt.year
    odds = odds.dropna(subset=["odds_w", "odds_l", "_wk", "_lk"])
    return odds[["_wk", "_lk", "_year", "odds_w", "odds_l"]]


def devig(odds_w, odds_l):
    """Decimal odds -> no-vig implied probability that the actual winner wins."""
    p_w, p_l = 1.0 / odds_w, 1.0 / odds_l
    return p_w / (p_w + p_l)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def per_row_logloss(y, p, eps=1e-15):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(y, p_model, p_market, n_boot=10000, seed=SEED):
    """
    Paired bootstrap on the per-match log-loss difference. Answers whether the
    gap between model and market is distinguishable from noise at this sample
    size, which is the first question anyone will ask about it.
    """
    d = per_row_logloss(y, p_model) - per_row_logloss(y, p_market)
    rng = np.random.default_rng(seed)
    n = len(d)
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return d.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def backtest(years, seed=SEED, n_boot=10000):
    years = sorted(years)
    odds = load_wimbledon_odds(years)
    odds["market_prob_winner"] = devig(odds["odds_w"], odds["odds_l"])
    lookup = odds.set_index(["_wk", "_lk", "_year"])["market_prob_winner"]
    market_index = set(odds["_wk"]).union(odds["_lk"])

    matches = load_matches(matches_path)
    cutoff = pd.Timestamp(f"{years[0]}-01-01")

    train_matches = matches[matches["date"] < cutoff]
    print(f"\nFitting on {len(train_matches):,} matches before {cutoff:%Y-%m-%d}")
    fit_stats = Player_Stats()
    train_df = pd.DataFrame(build_training_rows(train_matches, fit_stats))
    model, medians = train_model(train_df, verbose=False)
    print(f"  {len(train_df):,} training rows")

    # Walk forward over the full history with a fresh accumulator, so each
    # prediction sees only matches that preceded it.
    running = Player_Stats()
    records, unmatched = [], []

    for _, m in matches.sort_values("date").iterrows():
        w, l, yr = m["_w"], m["_l"], m["_year"]

        if yr in years and m["_is_wimb"]:
            wk, lk = resolve(w, market_index), resolve(l, market_index)
            key = (wk, lk, yr)
            if wk and lk and key in lookup.index:
                row = running.build_feature_vector(
                    w, l, m["winner_rank"], m["loser_rank"], m["date"],
                    year=yr, medians=medians, surface="Grass", best_of=m.get("best_of", 5))
                X = apply_feature_medians(pd.DataFrame([row]), medians)[FEATURES]
                p_market = lookup.loc[key]
                if isinstance(p_market, pd.Series):
                    p_market = p_market.iloc[0]
                records.append({
                    "year": yr,
                    "round": m.get("round"),
                    "p_model_winner": float(model.predict_proba(X)[0, 1]),
                    "p_market_winner": float(p_market),
                })
            else:
                unmatched.append((yr, m["winner_name"], m["loser_name"]))

        running.update_after_match(
            m, w, l, m["date"], yr, m["_is_grass"], m["_is_wimb"],
            m["_is_gs"] and m.get("round") == "F", m["winner_rank"], m["loser_rank"])
        running.accumulate_serve(w, m.get("w_ace"), m.get("w_svpt"), m.get("w_1stIn"),
                                 m.get("w_1stWon"), m.get("w_2ndWon"), m["_is_grass"])
        running.accumulate_serve(l, m.get("l_ace"), m.get("l_svpt"), m.get("l_1stIn"),
                                 m.get("l_1stWon"), m.get("l_2ndWon"), m["_is_grass"])

    df = pd.DataFrame(records)
    total_wimb = int(((matches["_year"].isin(years)) & matches["_is_wimb"]).sum())
    coverage = len(df) / total_wimb if total_wimb else 0.0

    print(f"\nJoin coverage: {len(df):,} of {total_wimb:,} Wimbledon matches "
          f"({coverage:.1%})")
    if unmatched:
        print(f"  {len(unmatched)} unmatched. First 15:")
        for yr, wn, ln in unmatched[:15]:
            print(f"    {yr}  {wn} d. {ln}")

    # Every row is stored winner-first, so labels would otherwise be all ones
    # and the evaluation degenerate. Randomise the orientation with a fixed
    # seed: flipped rows become label 0 with both probabilities complemented.
    rng = np.random.default_rng(seed)
    flip = rng.random(len(df)) < 0.5
    df["label"] = np.where(flip, 0, 1)
    df["p_model"] = np.where(flip, 1 - df["p_model_winner"], df["p_model_winner"])
    df["p_market"] = np.where(flip, 1 - df["p_market_winner"], df["p_market_winner"])

    y = df["label"].to_numpy()
    model_ll = log_loss(y, df["p_model"], labels=[0, 1])
    market_ll = log_loss(y, df["p_market"], labels=[0, 1])
    model_br = brier_score_loss(y, df["p_model"])
    market_br = brier_score_loss(y, df["p_market"])
    mean_d, lo, hi = paired_bootstrap(y, df["p_model"], df["p_market"], n_boot, seed)

    print(f"\nWimbledon {years} - {len(df):,} matches")
    print(f"  model  log-loss {model_ll:.4f}   brier {model_br:.4f}")
    print(f"  market log-loss {market_ll:.4f}   brier {market_br:.4f}")
    print(f"  gap (model - market): {mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  {'significant' if lo > 0 or hi < 0 else 'not distinguishable from noise'}")

    by_round = df.groupby("round").apply(lambda g: pd.Series({
        "n": len(g),
        "model": log_loss(g["label"], g["p_model"], labels=[0, 1]),
        "market": log_loss(g["label"], g["p_market"], labels=[0, 1]),
    }))
    by_round["gap"] = by_round["model"] - by_round["market"]
    by_round = by_round.sort_values("n", ascending=False)
    print("\nBy round:")
    print(by_round.to_string(float_format=lambda v: f"{v:.4f}"))

    write_report(years, df, coverage, total_wimb, model_ll, market_ll,
                 model_br, market_br, mean_d, lo, hi, by_round, len(train_df), cutoff)
    return df


def write_report(years, df, coverage, total_wimb, model_ll, market_ll,
                 model_br, market_br, mean_d, lo, hi, by_round, n_train, cutoff):
    os.makedirs(results_dir, exist_ok=True)
    out = ["<!-- generated by odds_backtest.py; do not edit by hand -->", ""]
    out.append(f"Model fit once on {n_train:,} rows from matches before "
               f"{cutoff:%Y-%m-%d}, then walked forward chronologically. "
               f"Evaluated on **{len(df):,} of {total_wimb:,}** Wimbledon matches "
               f"across {years} ({coverage:.1%} odds coverage).\n")
    out.append("| | Log-loss | Brier |")
    out.append("|---|---|---|")
    out.append(f"| Model | {model_ll:.4f} | {model_br:.4f} |")
    out.append(f"| Market (devigged) | {market_ll:.4f} | {market_br:.4f} |")
    out.append(f"| **Gap** | **{mean_d:+.4f}** | {model_br - market_br:+.4f} |")
    out.append(f"\nPaired bootstrap 95% CI on the log-loss gap: "
               f"[{lo:+.4f}, {hi:+.4f}]. "
               f"{'The gap is significant at this sample size.' if lo > 0 or hi < 0 else 'The gap is not distinguishable from noise at this sample size.'}\n")
    out.append("### By round\n")
    out.append("| Round | n | Model | Market | Gap |")
    out.append("|---|---|---|---|---|")
    for rnd, r in by_round.iterrows():
        out.append(f"| {rnd} | {int(r['n'])} | {r['model']:.4f} | "
                   f"{r['market']:.4f} | {r['gap']:+.4f} |")

    path = os.path.join(results_dir, "backtest_report.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    backtest(args.years, seed=args.seed, n_boot=args.n_boot)