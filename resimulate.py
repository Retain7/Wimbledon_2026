"""
resimulate.py - conditional resimulation of the Wimbledon draw as results
come in, layered on top of wimbledon_rf.py.

Usage (after each completed round):
    python resimulate.py --results r1_results.json --round 1
    python resimulate.py --results r2_results.json --round 2 --refit

Each results file is a JSON list of that round's completed matches, using the
exact "name" strings from wimbledon_2026_draw.json:
[
  {"winner": "SINNER, Jannik", "loser": "KECMANOVIC, Miomir"},
  ...
]

DESIGN
------
The bracket is collapsed to survivors and resimulated forward. The fitted
model really is left alone between rounds: it is trained once, cached to
data/model.joblib, and reloaded on subsequent runs. Pass --refit to retrain
deliberately. (An earlier version claimed the model was untouched while
calling train_model on every invocation.)

What this does NOT do is update player form from the tournament in progress.
Player_Stats is built from the historical CSV only, so a player who has just
won three matches carries the same features into round four that he had on day
one. The update here is bracket collapse, not information update. Feeding
in-tournament results into the accumulators is listed under Going Forward.
"""

import os
import json
import argparse
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from wimbledon_rf import (
    Player_Stats, load_matches, build_training_rows, train_model,
    precompute_win_probs, championship_probabilities, set_seed,
    matches_path, profiles_path, draw_path, data_dir, SEED,
)

STATE_PATH = os.path.join(data_dir, "tournament_state.json")
MODEL_PATH = os.path.join(data_dir, "model.joblib")


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"completed_rounds": 0, "eliminated": [], "match_log": []}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def validate_results(results, full_draw, state):
    """Fail loudly rather than silently producing a malformed bracket."""
    names = {p["name"] for p in full_draw}
    already = set(state["eliminated"])
    problems = []

    for r in results:
        for role in ("winner", "loser"):
            if r[role] not in names:
                problems.append(f"{r[role]!r} is not in the draw")
            elif r[role] in already:
                problems.append(f"{r[role]!r} was already eliminated")

    losers = [r["loser"] for r in results]
    dupes = [n for n, c in Counter(losers).items() if c > 1]
    if dupes:
        problems.append(f"duplicate losers: {dupes}")

    if problems:
        raise ValueError("Bad results file:\n  " + "\n  ".join(problems))


def get_model(matches, refit=False):
    """
    Load the cached model if present, otherwise fit once and cache. Returns
    (model, medians, player_stats). player_stats has to be rebuilt each run
    because it is a rolling accumulator, not a fitted object, but that pass is
    cheap relative to fitting a 1000-tree forest.
    """
    stats = Player_Stats()
    train_df = pd.DataFrame(build_training_rows(matches, stats))

    if not refit and os.path.exists(MODEL_PATH):
        model, medians = joblib.load(MODEL_PATH)
        print(f"Loaded cached model from {MODEL_PATH}")
        return model, medians, stats

    print("Fitting model (first run or --refit)...")
    model, medians = train_model(train_df, verbose=False)
    joblib.dump((model, medians), MODEL_PATH)
    print(f"Cached model to {MODEL_PATH}")
    return model, medians, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True,
                        help="JSON file with this round's completed matches")
    parser.add_argument("--round", type=int, required=True,
                        help="Round number just completed (1=R128 ... 7=F)")
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--refit", action="store_true",
                        help="Retrain and overwrite the cached model")
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.results) as f:
        results = json.load(f)
    with open(draw_path) as f:
        full_draw = json.load(f)

    state = load_state()
    validate_results(results, full_draw, state)

    # Apply to a candidate copy and validate the resulting bracket BEFORE
    # persisting. A bad results file should fail without leaving
    # tournament_state.json half-updated, which would compound on the next run.
    candidate = json.loads(json.dumps(state))
    for r in results:
        candidate["eliminated"].append(r["loser"])
        candidate["match_log"].append({"round": args.round, **r})
    candidate["completed_rounds"] = max(candidate["completed_rounds"], args.round)

    # Draw order encodes bracket structure, so filtering to survivors
    # preserves the correct pairings for the remaining rounds. That only holds
    # if each completed round was applied in full, which this check enforces.
    eliminated = set(candidate["eliminated"])
    live_draw = [p for p in full_draw if p["name"] not in eliminated]
    n = len(live_draw)
    if n & (n - 1) != 0:
        raise ValueError(
            f"{n} survivors is not a power of two. A round was applied "
            f"partially, or a result is missing. Fix {args.results} and rerun. "
            f"State on disk is unchanged.")

    save_state(candidate)
    print(f"{n} players remain after round {args.round}")

    matches = load_matches(matches_path)
    profiles = pd.read_csv(profiles_path)
    prof_idx = profiles.set_index("name")
    model, medians, player_stats = get_model(matches, refit=args.refit)

    as_of_date = matches["date"].max()
    P, _, asym = precompute_win_probs(
        live_draw, model, player_stats, prof_idx, as_of_date,
        int(as_of_date.year), medians)

    rng = np.random.default_rng(args.seed)
    champ_probs = championship_probabilities(live_draw, P, args.n_sims, rng)

    print(f"\n--- Championship probabilities after round {args.round} "
          f"({args.n_sims:,} sims, seed {args.seed}) ---")
    for player, p in champ_probs.items():
        if p >= 0.01:
            print(f"  {player:<35} {p:.1%}")


if __name__ == "__main__":
    main()