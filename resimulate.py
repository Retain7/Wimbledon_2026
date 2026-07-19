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
Layer A (every call, essentially free): the bracket is collapsed to
reflect real results. Eliminated players are dropped entirely; only
survivors are resimulated forward. The fitted model is untouched.

Layer B (once per round, after ALL of that round's matches are in):
each surviving player's grass_win_rate is nudged toward their
Wimbledon-2026 performance so far, blended with their career grass
rate. The blend weight grows round-over-round (BLEND_SCHEDULE) — a
single extra match shouldn't overwrite years of history, but by the
semis it should matter more. This is intentionally NOT a retrain:
refitting a 1000-tree forest on a handful of new rows would overfit
badly, so only the *inputs* to the already-fitted model move.

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

# Weight given to in-tournament form vs. career history, applied once a
# round is fully complete. R1 barely moves the needle; by the SF/F the
# live signal is weighted much more heavily.
BLEND_SCHEDULE = {
    1: 0.05,
    2: 0.10,
    3: 0.15,
    4: 0.20,  # Round of 16
    5: 0.30,  # QF
    6: 0.40,  # SF
    7: 0.40,  # F
}


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


def blend_in_tournament_form(player_stats, state, prof_idx):
    """Layer B: nudge grass_win_rate toward Wimbledon-2026-so-far form."""
    weight = BLEND_SCHEDULE.get(state["completed_rounds"], 0.0)
    if weight == 0:
        return player_stats

    wins, losses = Counter(), Counter()
    for m in state["match_log"]:
        wins[normalise(prof_idx.loc[m["winner"], "sackmann_name"])] += 1
        losses[normalise(prof_idx.loc[m["loser"], "sackmann_name"])] += 1

    for player in set(list(wins) + list(losses)):
        played = wins[player] + losses[player]
        if played == 0:
            continue
        live_rate = wins[player] / played
        n = player_stats.grass_t[player]
        if n == 0:
            continue
        prior_rate = player_stats.grass_w[player] / n
        blended_rate = (1 - weight) * prior_rate + weight * live_rate
        # keep the original sample size, just shift the implied rate
        player_stats.grass_w[player] = blended_rate * n
    return player_stats


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

    player_stats = blend_in_tournament_form(player_stats, state, prof_idx)

    with open(draw_path) as f:
        full_draw = json.load(f)

    # Layer A: collapse the bracket to survivors only. Original draw order
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