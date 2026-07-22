"""
wimbledon_rf.py - core pipeline for the Wimbledon 2026 win-probability model.

Trains a random forest on ATP match data across all surfaces, evaluates it on
Wimbledon matches only, and simulates the 2026 gentlemen's singles draw.

Usage:
    python wimbledon_rf.py                  # train, evaluate, simulate
    python wimbledon_rf.py --n-sims 100000
    python wimbledon_rf.py --skip-cv        # simulate only
    python wimbledon_rf.py --seed 7

Outputs (written to ./results/):
    model_report.md     markdown block, paste-ready for the README
    pretournament.json  championship probabilities + pairwise matrix,
                        consumed by postmortem.py
"""

import os
import json
import random
import argparse
import warnings
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Paths and global config
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
results_dir = os.path.join(script_dir, "results")
matches_path = os.path.join(data_dir, "atp_matches_all.csv")
profiles_path = os.path.join(data_dir, "player_profiles.csv")
draw_path = os.path.join(data_dir, "wimbledon_2026_draw.json")

SEED = 42


FEATURES = [
    # Ranking
    "rank_diff",
    "p1_peak_rank",
    "p2_peak_rank",
    # Grass / Wimbledon formula
    "wimb_formula_diff",
    "p1_grass_win_rate",
    "p2_grass_win_rate",
    "p1_peak_wimb_rate",
    "p2_peak_wimb_rate",
    # Form
    "p1_ytd_win_rate",
    "p2_ytd_win_rate",
    # Quality of competition
    "p1_top10_win_rate",
    "p2_top10_win_rate",
    # Grass-isolated serve metrics
    "p1_grass_serve_quality",
    "p2_grass_serve_quality",
    "p1_ace_rate",
    "p2_ace_rate",
    "p1_first_serve_pct",
    "p2_first_serve_pct",
    "p1_second_serve_won_pct",
    "p2_second_serve_won_pct",
    # Match context. The model trains on all surfaces, so it needs to know
    # which surface it is looking at. Without this it cannot tell a clay
    # match from a grass one while the grass-specific features above sit
    # median-imputed.
    "is_grass_match",
    "is_clay_match",
    "is_best_of_5",
    # Sample-size signal for the grass-specific features. log1p(grass matches
    # played) encodes both "has grass data at all" and "how much", so the
    # forest can discount an imputed median instead of treating it as an
    # observed value.
    "p1_grass_n",
    "p2_grass_n",
]

RF_PARAMS = dict(
    n_estimators=1000,
    max_depth=15,
    min_samples_leaf=25,
    max_features="sqrt",
    criterion="log_loss",
    random_state=SEED,
    n_jobs=-1,
)


def set_seed(seed):
    """Seed every RNG the pipeline touches, so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    RF_PARAMS["random_state"] = seed


def log_rank_diff(r1, r2):
    return np.log(r1) - np.log(r2)


def normalise(name):
    return str(name).strip().lower()


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------
class Player_Stats:
    """
    Rolling per-player accumulators. Every getter reflects only matches
    already passed to update_after_match / accumulate_serve, which is what
    keeps the feature build free of lookahead.
    """

    def __init__(self):
        self.grass_w = defaultdict(int)
        self.grass_t = defaultdict(int)
        self.peak_rank = defaultdict(lambda: np.inf)
        self.gs_champ = defaultdict(int)
        self.top10_w = defaultdict(int)
        self.top10_t = defaultdict(int)
        self.ytd = defaultdict(list)
        self.srv_won = defaultdict(float)
        self.srv_tot = defaultdict(float)
        self.ace_tot = defaultdict(float)
        self.fs_in = defaultdict(float)
        self.fs_tot = defaultdict(float)
        self.ss_won = defaultdict(float)
        self.ss_tot = defaultdict(float)
        self.wimb_seas = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        self.grass_history = defaultdict(list)

    def safe_rate(self, wins, total):
        return wins / total if total > 0 else np.nan

    def ytd_rate(self, player, year):
        entries = [won for y, won in self.ytd[player] if y == year]
        return sum(entries) / len(entries) if entries else np.nan

    def top10_rate(self, player):
        return self.top10_w[player] / self.top10_t[player] if self.top10_t[player] > 0 else np.nan

    def best_wimb(self, player):
        seasons = self.wimb_seas[player]
        if not seasons:
            return np.nan
        rates = [v[0] / v[1] for v in seasons.values() if v[1] > 0]
        return max(rates) if rates else np.nan

    def serve_qual(self, player):
        return self.srv_won[player] / self.srv_tot[player] if self.srv_tot[player] > 0 else np.nan

    def ace_rate(self, player):
        return self.ace_tot[player] / self.srv_tot[player] if self.srv_tot[player] > 0 else np.nan

    def first_serve_pct(self, player):
        return self.fs_in[player] / self.fs_tot[player] if self.fs_tot[player] > 0 else np.nan

    def second_serve_won_pct(self, player):
        return self.ss_won[player] / self.ss_tot[player] if self.ss_tot[player] > 0 else np.nan

    def grass_n(self, player):
        return float(np.log1p(self.grass_t[player]))

    def wimbledon_formula_score(self, player, current_date, current_rank):
        """Proxy for the Wimbledon seeding formula using rolling grass history."""
        base_pts = 10000 / (current_rank + 1) if (pd.notna(current_rank) and current_rank > 0) else 0
        past_12m = current_date - pd.Timedelta(days=365)
        past_24m = current_date - pd.Timedelta(days=730)

        recent_grass_wins = sum(1 for d, _, w in self.grass_history[player] if w and d >= past_12m)
        recent_grass_pts = recent_grass_wins * 50

        old_grass = [t for d, t, w in self.grass_history[player] if w and past_24m <= d < past_12m]
        best_old_tourney_pts = (Counter(old_grass).most_common(1)[0][1] * 50) if old_grass else 0

        return base_pts + recent_grass_pts + (0.75 * best_old_tourney_pts)

    def accumulate_serve(self, player, ace, svpt, fst_in, fst_won, snd_won, is_grass):
        if not is_grass:
            return
        if not (pd.notna(svpt) and svpt > 0 and pd.notna(ace) and pd.notna(fst_in)
                and pd.notna(fst_won) and pd.notna(snd_won)):
            return
        self.srv_won[player] += fst_won + snd_won
        self.srv_tot[player] += svpt
        self.ace_tot[player] += ace
        self.fs_in[player] += fst_in
        self.fs_tot[player] += svpt
        snd_faced = svpt - fst_in
        if snd_faced > 0:
            self.ss_won[player] += snd_won
            self.ss_tot[player] += snd_faced

    def update_after_match(self, match, w, l, dt, yr, is_grass, is_wimb, is_gf, wr, lr):
        if is_grass:
            self.grass_w[w] += 1
            self.grass_t[w] += 1
            self.grass_t[l] += 1
            self.grass_history[w].append((dt, match["tourney_name"], True))
            self.grass_history[l].append((dt, match["tourney_name"], False))

        if is_wimb and pd.notna(yr):
            self.wimb_seas[w][yr][0] += 1
            self.wimb_seas[w][yr][1] += 1
            self.wimb_seas[l][yr][1] += 1

        if is_gf:
            self.gs_champ[w] = 1

        if pd.notna(lr) and lr <= 10:
            self.top10_w[w] += 1
            self.top10_t[w] += 1
        if pd.notna(wr) and wr <= 10:
            self.top10_t[l] += 1

        if pd.notna(wr):
            self.peak_rank[w] = min(self.peak_rank[w], wr)
        if pd.notna(lr):
            self.peak_rank[l] = min(self.peak_rank[l], lr)

        if pd.notna(yr):
            self.ytd[w].append((yr, True))
            self.ytd[l].append((yr, False))

    def build_feature_vector(self, p1_name, p2_name, p1_rank, p2_rank, current_date,
                             year=None, medians=None, surface="Grass", best_of=5):
        row = {
            "rank_diff": log_rank_diff(p1_rank, p2_rank),
            "wimb_formula_diff": (
                self.wimbledon_formula_score(p1_name, current_date, p1_rank)
                - self.wimbledon_formula_score(p2_name, current_date, p2_rank)
            ),
            "p1_peak_rank": self.peak_rank[p1_name] if self.peak_rank[p1_name] != np.inf else np.nan,
            "p2_peak_rank": self.peak_rank[p2_name] if self.peak_rank[p2_name] != np.inf else np.nan,
            "p1_grass_win_rate": self.safe_rate(self.grass_w[p1_name], self.grass_t[p1_name]),
            "p2_grass_win_rate": self.safe_rate(self.grass_w[p2_name], self.grass_t[p2_name]),
            "p1_peak_wimb_rate": self.best_wimb(p1_name),
            "p2_peak_wimb_rate": self.best_wimb(p2_name),
            "p1_ytd_win_rate": self.ytd_rate(p1_name, year) if year is not None else np.nan,
            "p2_ytd_win_rate": self.ytd_rate(p2_name, year) if year is not None else np.nan,
            "p1_top10_win_rate": self.top10_rate(p1_name),
            "p2_top10_win_rate": self.top10_rate(p2_name),
            "p1_grass_serve_quality": self.serve_qual(p1_name),
            "p2_grass_serve_quality": self.serve_qual(p2_name),
            "p1_ace_rate": self.ace_rate(p1_name),
            "p2_ace_rate": self.ace_rate(p2_name),
            "p1_first_serve_pct": self.first_serve_pct(p1_name),
            "p2_first_serve_pct": self.first_serve_pct(p2_name),
            "p1_second_serve_won_pct": self.second_serve_won_pct(p1_name),
            "p2_second_serve_won_pct": self.second_serve_won_pct(p2_name),
            "is_grass_match": 1.0 if surface == "Grass" else 0.0,
            "is_clay_match": 1.0 if surface == "Clay" else 0.0,
            "is_best_of_5": 1.0 if (pd.notna(best_of) and int(best_of) == 5) else 0.0,
            "p1_grass_n": self.grass_n(p1_name),
            "p2_grass_n": self.grass_n(p2_name),
        }

        if medians is not None:
            for feature, value in row.items():
                if pd.isna(value):
                    row[feature] = medians.get(feature, value)
        return row


# ---------------------------------------------------------------------------
# Data loading and training-row construction
# ---------------------------------------------------------------------------
def load_matches(path):
    matches = pd.read_csv(path, low_memory=False)
    matches["date"] = pd.to_datetime(matches["tourney_date"].astype(str),
                                     format="%Y%m%d", errors="coerce")
    matches = matches.sort_values("date").reset_index(drop=True)

    for col in ["winner_rank", "loser_rank", "best_of",
                "w_ace", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
                "l_ace", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    if "best_of" not in matches.columns:
        matches["best_of"] = np.nan

    matches["_w"] = matches["winner_name"].apply(normalise)
    matches["_l"] = matches["loser_name"].apply(normalise)
    matches["_is_grass"] = matches["surface"] == "Grass"
    matches["_is_wimb"] = matches["tourney_name"].str.contains("Wimbledon", na=False)
    matches["_is_gs"] = matches["tourney_level"] == "G"
    matches["_year"] = matches["date"].dt.year
    return matches


def build_training_rows(matches, player_stats):
    """
    Every match with valid ranks and date produces two rows, one from each
    player's perspective, so the label is balanced by construction and the
    model cannot learn a positional prior on p1.

    Features are built from Player_Stats state *before* the match is folded
    in, so no row ever sees its own outcome. Grass-specific features are NaN
    for a player with no grass history and get median-filled downstream;
    p1_grass_n / p2_grass_n carry the sample size so the forest can tell an
    imputed value from an observed one.
    """
    training_rows = []
    for midx, match in matches.iterrows():
        w, l = match["_w"], match["_l"]
        dt, yr = match["date"], match["_year"]
        is_grass, is_wimb, is_gs = match["_is_grass"], match["_is_wimb"], match["_is_gs"]
        is_gf = is_gs and match.get("round") == "F"
        wr, lr = match["winner_rank"], match["loser_rank"]
        surface, best_of = match.get("surface"), match.get("best_of")

        if pd.notna(wr) and pd.notna(lr) and pd.notna(dt):
            winner_row = player_stats.build_feature_vector(
                w, l, wr, lr, dt, year=yr, surface=surface, best_of=best_of)
            loser_row = player_stats.build_feature_vector(
                l, w, lr, wr, dt, year=yr, surface=surface, best_of=best_of)
            meta = {"match_id": midx, "is_wimb": bool(is_wimb), "year": yr,
                    "round": match.get("round")}
            winner_row.update({**meta, "label": 1})
            loser_row.update({**meta, "label": 0})
            training_rows.append(winner_row)
            training_rows.append(loser_row)

        player_stats.update_after_match(match, w, l, dt, yr, is_grass, is_wimb, is_gf, wr, lr)
        player_stats.accumulate_serve(
            w, match.get("w_ace", np.nan), match.get("w_svpt", np.nan),
            match.get("w_1stIn", np.nan), match.get("w_1stWon", np.nan),
            match.get("w_2ndWon", np.nan), is_grass)
        player_stats.accumulate_serve(
            l, match.get("l_ace", np.nan), match.get("l_svpt", np.nan),
            match.get("l_1stIn", np.nan), match.get("l_1stWon", np.nan),
            match.get("l_2ndWon", np.nan), is_grass)
    return training_rows


# ---------------------------------------------------------------------------
# Imputation, kept separate so it can be fit per CV fold
# ---------------------------------------------------------------------------
def fit_feature_medians(df, feature_cols=FEATURES):
    return {c: df[c].median() for c in feature_cols}


def apply_feature_medians(df, medians, feature_cols=FEATURES):
    """Return a filled copy. Does not mutate the caller's frame."""
    out = df.copy()
    for c in feature_cols:
        out[c] = out[c].fillna(medians[c])
    return out


def aggregate_importances(model, feature_cols=FEATURES):
    """Collapse p1_x / p2_x importance pairs into one number per concept."""
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    agg = defaultdict(float)
    for feat, imp in importances.items():
        base = feat[3:] if feat.startswith(("p1_", "p2_")) else feat
        agg[base] += imp
    return pd.Series(agg).sort_values(ascending=False)


def train_model(train_df, feature_cols=FEATURES, verbose=True):
    missing = set(feature_cols) - set(train_df.columns)
    assert not missing, f"FEATURES declared but never populated: {sorted(missing)}"

    medians = fit_feature_medians(train_df, feature_cols)
    filled = apply_feature_medians(train_df, medians, feature_cols)

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(filled[feature_cols], filled["label"])

    if verbose:
        print("\nFeature importances (p1/p2 combined, per concept):")
        for base, imp in aggregate_importances(model, feature_cols).items():
            print(f"  {base:<25} {imp:.3f}")

    return model, medians


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cv_evaluate(train_df, feature_cols=FEATURES, n_splits=5):
    """
    Train on all surfaces, score on Wimbledon only.

    GroupKFold splits the FULL frame on match_id, so each training fold keeps
    the whole all-surface sample minus the held-out matches, and a match's
    winner-row and loser-row can never straddle the split. The test fold is
    then restricted to its Wimbledon rows before scoring, which is the
    evaluation slice we actually care about.

    Median imputation is fit inside each training fold, not on the full frame,
    so held-out rows contribute nothing to their own fill values.

    NOTE: GroupKFold is not time-aware. It shuffles matches across seasons, so
    this number is optimistic relative to the strictly chronological
    walk-forward evaluation in odds_backtest.py. Treat the backtest as the
    headline result; this is a sanity check, not a claim of live performance.
    """
    gkf = GroupKFold(n_splits=n_splits)
    y = train_df["label"].to_numpy()
    groups = train_df["match_id"].to_numpy()
    is_wimb = train_df["is_wimb"].to_numpy(dtype=bool)

    rows = []
    for fold, (tr, te) in enumerate(gkf.split(train_df, y, groups), start=1):
        te_wimb = te[is_wimb[te]]
        if len(te_wimb) == 0:
            continue

        tr_df = train_df.iloc[tr]
        medians = fit_feature_medians(tr_df, feature_cols)
        tr_X = apply_feature_medians(tr_df, medians, feature_cols)[feature_cols]
        te_X = apply_feature_medians(train_df.iloc[te_wimb], medians, feature_cols)[feature_cols]
        y_tr, y_te = y[tr], y[te_wimb]

        fm = RandomForestClassifier(**RF_PARAMS)
        fm.fit(tr_X, y_tr)
        proba = fm.predict_proba(te_X)[:, 1]

        # Rank-only floor: a one-parameter logistic on log rank difference,
        # fit on the same training fold. This answers "what did you beat",
        # it is not an argument about model class.
        base = LogisticRegression()
        base.fit(tr_X[["rank_diff"]], y_tr)
        base_proba = base.predict_proba(te_X[["rank_diff"]])[:, 1]

        rows.append({
            "fold": fold,
            "n_train": len(tr),
            "n_eval": len(te_wimb),
            "brier": brier_score_loss(y_te, proba),
            "logloss": log_loss(y_te, proba, labels=[0, 1]),
            "acc": accuracy_score(y_te, (proba >= 0.5).astype(int)),
            "base_brier": brier_score_loss(y_te, base_proba),
            "base_logloss": log_loss(y_te, base_proba, labels=[0, 1]),
            "base_acc": accuracy_score(y_te, (base_proba >= 0.5).astype(int)),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pairwise probability matrix
# ---------------------------------------------------------------------------
def build_matchup_rows(player_stats, prof_idx, pairs, as_of_date, as_of_year, medians):
    """Build feature rows for a list of (name_a, name_b) draw pairs."""
    rows = []
    for a, b in pairs:
        pa, pb = prof_idx.loc[a], prof_idx.loc[b]
        na, nb = normalise(pa["sackmann_name"]), normalise(pb["sackmann_name"])
        rows.append(player_stats.build_feature_vector(
            na, nb, pa["rank"], pb["rank"], as_of_date,
            year=as_of_year, medians=medians, surface="Grass", best_of=5))
    return pd.DataFrame(rows)[FEATURES]


def precompute_win_probs(draw, model, player_stats, prof_idx, as_of_date, as_of_year, medians):
    """
    Return (P, names, asym) where P[i, j] is the probability player i beats
    player j, and asym is the mean absolute antisymmetry violation.

    The forest trains on both orderings but is not exactly antisymmetric, so
    P(a beats b) and 1 - P(b beats a) generally disagree. The previous version
    keyed the cache on alphabetically sorted names, which meant the simulation
    silently used whichever of the two estimates happened to sort first. Here
    both orderings are evaluated and averaged, which makes P exactly
    antisymmetric and removes the dependence on name order. The size of the
    disagreement is reported rather than hidden.
    """
    names = [p["name"] for p in draw]
    if len(set(names)) != len(names):
        raise ValueError("Draw contains duplicate player names.")

    idx = {n: i for i, n in enumerate(names)}
    pairs = list(combinations(names, 2))

    fwd = model.predict_proba(build_matchup_rows(
        player_stats, prof_idx, pairs, as_of_date, as_of_year, medians))[:, 1]
    rev = model.predict_proba(build_matchup_rows(
        player_stats, prof_idx, [(b, a) for a, b in pairs],
        as_of_date, as_of_year, medians))[:, 1]

    asym = float(np.mean(np.abs(fwd + rev - 1.0)))
    sym = (fwd + (1.0 - rev)) / 2.0

    n = len(names)
    P = np.full((n, n), 0.5, dtype=float)
    for (a, b), p in zip(pairs, sym):
        P[idx[a], idx[b]] = p
        P[idx[b], idx[a]] = 1.0 - p
    return P, names, asym


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_tournament(P, n_players, n_sims, rng):
    """
    Vectorised bracket simulation. `current` holds one column per remaining
    slot and one row per simulation; each round halves the width. Draw order
    encodes bracket structure, so adjacent columns are the pairings.
    """
    if n_players & (n_players - 1) != 0:
        raise ValueError(f"Draw size must be a power of two, got {n_players}.")

    current = np.tile(np.arange(n_players), (n_sims, 1))
    while current.shape[1] > 1:
        a, b = current[:, 0::2], current[:, 1::2]
        current = np.where(rng.random(a.shape) < P[a, b], a, b)
    return current[:, 0]


def championship_probabilities(draw, P, n_sims, rng):
    champions = simulate_tournament(P, len(draw), n_sims, rng)
    counts = Counter(champions.tolist())
    return {draw[i]["name"]: c / n_sims for i, c in counts.most_common()}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def markdown_report(cv, importances, champ_probs, asym, n_rows, n_wimb_rows,
                    n_matches, n_sims, seed, date_range):
    out = ["<!-- generated by wimbledon_rf.py; do not edit by hand -->"]
    out.append(f"\n_{n_matches:,} matches ({date_range}) produced {n_rows:,} training rows, "
               f"of which {n_wimb_rows:,} are Wimbledon. Seed {seed}._\n")

    if cv is not None and len(cv):
        m, s = cv.mean(numeric_only=True), cv.std(numeric_only=True)
        out.append("### Cross-validated performance (Wimbledon evaluation slice)\n")
        out.append(f"Trained on all surfaces, scored on {int(cv['n_eval'].sum()):,} "
                   f"held-out Wimbledon rows across {len(cv)} folds.\n")
        out.append("| Model | Brier | Log-loss | Accuracy |")
        out.append("|---|---|---|---|")
        out.append(f"| Random forest | {m['brier']:.4f} ± {s['brier']:.4f} | "
                   f"{m['logloss']:.4f} | {m['acc']:.1%} |")
        out.append(f"| Rank-only logistic (floor) | {m['base_brier']:.4f} ± "
                   f"{s['base_brier']:.4f} | {m['base_logloss']:.4f} | {m['base_acc']:.1%} |")
        out.append(f"\nThe forest improves on the rank-only floor by "
                   f"**{m['base_logloss'] - m['logloss']:.4f} log-loss**.\n")

    out.append("### Feature importance (p1/p2 pairs combined)\n")
    out.append("| Feature | Importance |")
    out.append("|---|---|")
    for base, imp in importances.items():
        out.append(f"| `{base}` | {imp:.3f} |")

    out.append("\n### Model coherence\n")
    out.append(f"Mean antisymmetry violation across all draw pairs, "
               f"|P(a beats b) + P(b beats a) − 1|: **{asym:.4f}**. The simulation "
               f"uses the symmetrised average of both orderings.\n")

    out.append("### 2026 championship probabilities\n")
    out.append(f"{n_sims:,} simulations from the actual draw, pre-tournament.\n")
    out.append("| Player | Win probability |")
    out.append("|---|---|")
    for player, p in champ_probs.items():
        if p >= 0.01:
            out.append(f"| {player} | {p:.1%} |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sims", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(results_dir, exist_ok=True)

    matches = load_matches(matches_path)
    profiles = pd.read_csv(profiles_path)
    prof_idx = profiles.set_index("name")

    player_stats = Player_Stats()
    train_df = pd.DataFrame(build_training_rows(matches, player_stats))
    n_wimb_rows = int(train_df["is_wimb"].sum())
    date_range = f"{matches['date'].min():%Y-%m} to {matches['date'].max():%Y-%m}"
    print(f"{len(matches):,} matches -> {len(train_df):,} training rows "
          f"({n_wimb_rows:,} Wimbledon)")

    model, feature_medians = train_model(train_df)

    cv = None
    if not args.skip_cv:
        print("\nCross-validating (train: all surfaces, evaluate: Wimbledon only)...")
        cv = cv_evaluate(train_df, n_splits=args.folds)
        m, s = cv.mean(numeric_only=True), cv.std(numeric_only=True)
        print(f"  forest     brier {m['brier']:.4f} +/- {s['brier']:.4f}  "
              f"logloss {m['logloss']:.4f}  acc {m['acc']:.3f}")
        print(f"  rank-only  brier {m['base_brier']:.4f} +/- {s['base_brier']:.4f}  "
              f"logloss {m['base_logloss']:.4f}  acc {m['base_acc']:.3f}")
        print(f"  evaluated on {int(cv['n_eval'].sum()):,} Wimbledon rows")

    with open(draw_path) as f:
        draw = json.load(f)

    as_of_date = matches["date"].max()
    as_of_year = int(as_of_date.year)
    P, names, asym = precompute_win_probs(
        draw, model, player_stats, prof_idx, as_of_date, as_of_year, feature_medians)
    print(f"\nMean antisymmetry violation across draw pairs: {asym:.4f}")

    rng = np.random.default_rng(args.seed)
    champ_probs = championship_probabilities(draw, P, args.n_sims, rng)

    print(f"\n--- Wimbledon 2026 Championship Probabilities ({args.n_sims:,} sims) ---")
    for player, p in champ_probs.items():
        if p >= 0.01:
            print(f"  {player:<35} {p:.1%}")

    report = markdown_report(cv, aggregate_importances(model), champ_probs, asym,
                             len(train_df), n_wimb_rows, len(matches),
                             args.n_sims, args.seed, date_range)
    report_path = os.path.join(results_dir, "model_report.md")
    with open(report_path, "w") as f:
        f.write(report + "\n")

    pre_path = os.path.join(results_dir, "pretournament.json")
    with open(pre_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "n_sims": args.n_sims,
            "as_of_date": str(as_of_date.date()),
            "antisymmetry": asym,
            "names": names,
            "championship_probs": champ_probs,
            "pairwise": P.tolist(),
        }, f, indent=2)

    print(f"\nWrote {report_path}")
    print(f"Wrote {pre_path}")


if __name__ == "__main__":
    main()