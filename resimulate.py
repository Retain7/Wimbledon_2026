"""
resimulate.py — conditional resimulation of the Wimbledon draw as real
results come in, layered on top of wimbledon_rf.py.

Usage (after each completed round of real matches):
    python resimulate.py --results r1_results.json --round 1

Each results file is a JSON list of that round's completed matches, using
the exact "name" strings from wimbledon_2026_draw.json:
[
  {"winner": "SINNER, Jannik", "loser": "KECMANOVIC, Miomir"},
  {"winner": "RUUD, Casper",  "loser": "HURKACZ, Hubert"},
  ...
]

DESIGN
------
The bracket is collapsed to reflect real results. Eliminated players
are dropped entirely; only survivors are resimulated forward. The
fitted model is untouched.

State (which players are eliminated, what's been fed in so far) is
persisted in data/tournament_state.json so you can run this once per
round without re-supplying earlier rounds' results each time.
"""

import os
import json
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from wimbledon_rf import (
    Player_Stats, load_matches, build_training_rows, train_model,
    precompute_win_probs, simulate_tournament, normalise,
    matches_path, profiles_path, draw_path, data_dir,
)

STATE_PATH = os.path.join(data_dir, "tournament_state.json")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"completed_rounds": 0, "eliminated": [], "match_log": []}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def apply_round_results(state, results, round_num):
    for r in results:
        state["eliminated"].append(r["loser"])
        state["match_log"].append({"round": round_num, **r})
    state["completed_rounds"] = max(state["completed_rounds"], round_num)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="JSON file with this round's completed matches")
    parser.add_argument("--round", type=int, required=True, help="Round number just completed (1=R128 ... 7=F)")
    parser.add_argument("--n_sims", type=int, default=50000)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    state = load_state()
    state = apply_round_results(state, results, args.round)
    save_state(state)

    matches = load_matches(matches_path)
    profiles = pd.read_csv(profiles_path)
    prof_idx = profiles.set_index("name")
    player_stats = Player_Stats()
    training_rows = build_training_rows(matches, player_stats)
    train_df = pd.DataFrame(training_rows)
    model, feature_medians = train_model(train_df)

    with open(draw_path) as f:
        full_draw = json.load(f)

    # Collapse the bracket to survivors only. Original draw order
    # already encodes bracket structure, so filtering preserves correct
    # pairings for whatever rounds remain.
    eliminated = set(state["eliminated"])
    live_draw = [p for p in full_draw if p["name"] not in eliminated]

    as_of_date = matches["date"].max()
    as_of_year = int(as_of_date.year)
    win_prob = precompute_win_probs(
        live_draw, model, player_stats, prof_idx, as_of_date, as_of_year, feature_medians
    )

    results_ctr = Counter()
    for _ in range(args.n_sims):
        champion = simulate_tournament(live_draw, win_prob)
        results_ctr[champion["name"]] += 1

    print(f"\n--- Updated Championship Probabilities after Round {args.round} ---")
    for player, wins in results_ctr.most_common():
        p = wins / args.n_sims * 100
        if p >= 1:
            print(f"  {player:<35} {p:.1f}%")


if __name__ == "__main__":
    main()