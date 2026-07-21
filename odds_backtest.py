"""
odds_backtest.py — compare model log-loss vs. historical market
log-loss on past Wimbledons.

Requires data/odds/{year}.xlsx files from tennis-data.co.uk
(see module-level instructions in the chat message this came with).

Usage:
    python odds_backtest.py --years 2018 2019 2021 2022 2023 2024 2025
"""

import os
import argparse
import pandas as pd
from sklearn.metrics import log_loss

from wimbledon_rf import Player_Stats, load_matches, build_training_rows, train_model, normalise, matches_path, data_dir

ODDS_DIR = os.path.join(data_dir, "odds")


def canon_key(name):
    """
    Canonicalize a player name to 'lastname firstinitial', reconciling
    tennis-data.co.uk's 'Federer R.' style with Sackmann's full-name
    'Roger Federer' style. Assumes both inputs are already lowercased
    (call after normalise()), or handles raw strings directly.
    """
    s = str(name).strip().lower().replace(".", "")
    parts = s.split()
    if len(parts) < 2:
        return s
    if len(parts[-1]) == 1:
        # already "lastname f" shape (tennis-data)
        return s
    # "firstname lastname" -> "lastname f" (Sackmann)
    return f"{parts[-1]} {parts[0][0]}"


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
        raise FileNotFoundError("No odds files found in data/odds/. Download from tennis-data.co.uk first.")
    odds = pd.concat(frames, ignore_index=True)

    # Prefer Pinnacle (sharpest book), fall back to Bet365, then market average
    for wcol, lcol, tag in [("PSW", "PSL", "pinnacle"), ("B365W", "B365L", "bet365"), ("AvgW", "AvgL", "average")]:
        if wcol in odds.columns:
            odds["odds_w"], odds["odds_l"], odds["odds_source"] = odds[wcol], odds[lcol], tag
            break

    odds["_w"] = odds["Winner"].apply(canon_key)
    odds["_l"] = odds["Loser"].apply(canon_key)
    odds["_year"] = pd.to_datetime(odds["Date"]).dt.year
    return odds[["_w", "_l", "_year", "odds_w", "odds_l"]].dropna(subset=["odds_w", "odds_l"])


def devig(odds_w, odds_l):
    """Decimal odds -> no-vig implied probability the actual winner wins."""
    p_w_raw = 1.0 / odds_w
    p_l_raw = 1.0 / odds_l
    return p_w_raw / (p_w_raw + p_l_raw)


def backtest(years):
    odds = load_wimbledon_odds(years)
    odds["market_prob_winner"] = devig(odds["odds_w"], odds["odds_l"])
    odds_lookup = odds.set_index(["_w", "_l", "_year"])["market_prob_winner"]

    matches = load_matches(matches_path)
    cutoff = pd.Timestamp(f"{min(years)}-01-01")

    # Fit once on pre-backtest-window data only, to keep the headline
    # comparison honest (no future grass form leaking into the model).
    train_matches = matches[matches["date"] < cutoff]
    fit_stats = Player_Stats()
    train_df = pd.DataFrame(build_training_rows(train_matches, fit_stats))
    model, feature_medians = train_model(train_df)

    # Walk forward chronologically, updating a *running* Player_Stats as we
    # go, so each Wimbledon prediction only ever sees data before that match.
    running_stats = Player_Stats()
    y_true, model_probs, market_probs = [], [], []

    for _, m in matches.sort_values("date").iterrows():
        w, l, yr = m["_w"], m["_l"], m["_year"]
        if yr in years and m["_is_wimb"]:
            key = (canon_key(w), canon_key(l), yr)
            if key in odds_lookup.index:
                row = running_stats.build_feature_vector(
                    w, l, m["winner_rank"], m["loser_rank"], m["date"], year=yr, medians=feature_medians
                )
                p_model = model.predict_proba(pd.DataFrame([row])[model.feature_names_in_])[0, 1]
                p_market = odds_lookup.loc[key]
                if isinstance(p_market, pd.Series):
                    p_market = p_market.iloc[0]
                y_true.append(1)  # historical winner, by construction
                model_probs.append(p_model)
                market_probs.append(p_market)

        running_stats.update_after_match(
            m, w, l, m["date"], yr, m["_is_grass"], m["_is_wimb"],
            m["_is_gs"] and m.get("round") == "F", m["winner_rank"], m["loser_rank"],
        )
        running_stats.accumulate_serve(w, m.get("w_ace"), m.get("w_svpt"), m.get("w_1stIn"), m.get("w_1stWon"), m.get("w_2ndWon"), m["_is_grass"])
        running_stats.accumulate_serve(l, m.get("l_ace"), m.get("l_svpt"), m.get("l_1stIn"), m.get("l_1stWon"), m.get("l_2ndWon"), m["_is_grass"])

    model_ll = log_loss(y_true, model_probs, labels=[0, 1])
    market_ll = log_loss(y_true, market_probs, labels=[0, 1])
    print(f"\nBacktest over Wimbledon {sorted(years)} ({len(y_true)} matches with odds coverage)")
    print(f"  model  log-loss: {model_ll:.4f}")
    print(f"  market log-loss: {market_ll:.4f}")
    winner = "Model" if model_ll < market_ll else "Market"
    print(f"  {winner} wins by {abs(model_ll - market_ll):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", required=True)
    args = parser.parse_args()
    backtest(args.years)