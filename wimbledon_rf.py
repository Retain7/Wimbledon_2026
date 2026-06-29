
"""
wimbledon_rf.py
---------------
Trains a Random Forest on ATP grass match history and simulates Wimbledon 2026.

Training approach:
  - Every historical grass match becomes a training row.
  - All per-match features are computed from each player's history UP TO BUT
    NOT INCLUDING that match (no data leakage).
  - Player profiles (player_profiles.csv) are used only at simulation time.

Updated features vs prior version:
  - wimb_formula_diff      Replaced grass ELO with the classic Wimbledon seeding formula proxy
  - grass_serve_quality    Fraction of serve points won strictly on grass (career rolling avg)
  - Dropped raw p1/p2 rank to prevent collinearity with rank_diff
  - Removed recent_win_rate to eliminate clay season contamination

Prerequisites:
    1. Run fetch_data.py first — populates ./data/
    2. wimbledon_2026_draw.json must be in ./data/

Usage:
    python wimbledon_rf.py
"""

import os
import json
import random
import warnings
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
matches_path = os.path.join(data_dir, "atp_matches_all.csv")
profiles_path = os.path.join(data_dir, "player_profiles.csv")
draw_path = os.path.join(data_dir, "wimbledon_2026_draw.json")

FEATURES = [
    # Ranking
    "rank_diff",
    "p1_peak_rank",
    "p2_peak_rank",
    # Grass / Wimbledon Formula
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
    # Grass-Isolated Serve Metrics
    "p1_grass_serve_quality",
    "p2_grass_serve_quality",
    "p1_ace_rate",
    "p2_ace_rate",
    "p1_first_serve_pct",
    "p2_first_serve_pct",
    "p1_second_serve_won_pct",
    "p2_second_serve_won_pct",
]

RF_PARAMS = dict(
    n_estimators=1000,
    max_depth=15,
    min_samples_leaf=25,
    max_features="sqrt",
    criterion="log_loss",
    random_state=42,
    n_jobs=-1,
)


def log_rank_diff(r1, r2):
    """Log-ratio rank gap: log(r1) - log(r2)."""
    return np.log(r1) - np.log(r2)


def normalise(name):
    return str(name).strip().lower()


class Player_Stats:
    """Centralizes rolling player statistics and feature generation."""

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
        if not (pd.notna(svpt) and svpt > 0 and pd.notna(ace) and pd.notna(fst_in) and pd.notna(fst_won) and pd.notna(snd_won)):
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

    def build_feature_vector(self, p1_name, p2_name, p1_rank, p2_rank, current_date, year=None, medians=None):
        row = {
            "rank_diff": log_rank_diff(p1_rank, p2_rank),
            "wimb_formula_diff": self.wimbledon_formula_score(p1_name, current_date, p1_rank) - self.wimbledon_formula_score(p2_name, current_date, p2_rank),
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
        }

        if medians is not None:
            for feature, value in row.items():
                if pd.isna(value):
                    row[feature] = medians.get(feature, value)
        return row


def load_matches(matches_path):
    matches = pd.read_csv(matches_path, low_memory=False)
    matches["date"] = pd.to_datetime(matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    matches = matches.sort_values("date").reset_index(drop=True)

    for col in ["winner_rank", "loser_rank", "w_ace", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
                "l_ace", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")

    matches["_w"] = matches["winner_name"].apply(normalise)
    matches["_l"] = matches["loser_name"].apply(normalise)
    matches["_is_grass"] = matches["surface"] == "Grass"
    matches["_is_wimb"] = matches["tourney_name"].str.contains("Wimbledon", na=False)
    matches["_is_gs"] = matches["tourney_level"] == "G"
    matches["_year"] = matches["date"].dt.year
    return matches


def build_training_rows(matches, player_stats):
    training_rows = []
    for midx, match in matches.iterrows():
        w = match["_w"]
        l = match["_l"]
        dt = match["date"]
        yr = match["_year"]
        is_grass = match["_is_grass"]
        is_wimb = match["_is_wimb"]
        is_gs = match["_is_gs"]
        is_gf = is_gs and match.get("round") == "F"
        wr = match["winner_rank"]
        lr = match["loser_rank"]

        if is_grass and pd.notna(wr) and pd.notna(lr) and pd.notna(dt):
            winner_row = player_stats.build_feature_vector(w, l, wr, lr, dt, year=yr)
            loser_row = player_stats.build_feature_vector(l, w, lr, wr, dt, year=yr)
            winner_row.update({"match_id": midx, "label": 1})
            loser_row.update({"match_id": midx, "label": 0})
            training_rows.append(winner_row)
            training_rows.append(loser_row)

        player_stats.update_after_match(match, w, l, dt, yr, is_grass, is_wimb, is_gf, wr, lr)
        player_stats.accumulate_serve(
            w,
            match.get("w_ace", np.nan),
            match.get("w_svpt", np.nan),
            match.get("w_1stIn", np.nan),
            match.get("w_1stWon", np.nan),
            match.get("w_2ndWon", np.nan),
            is_grass,
        )
        player_stats.accumulate_serve(
            l,
            match.get("l_ace", np.nan),
            match.get("l_svpt", np.nan),
            match.get("l_1stIn", np.nan),
            match.get("l_1stWon", np.nan),
            match.get("l_2ndWon", np.nan),
            is_grass,
        )
    return training_rows


def train_model(train_df):
    missing = set(FEATURES) - set(train_df.columns)
    assert not missing, f"FEATURES declared but never populated: {sorted(missing)}"

    feature_medians = {}
    for col in FEATURES:
        med_val = train_df[col].median()
        feature_medians[col] = med_val
        train_df[col] = train_df[col].fillna(med_val)

    X = train_df[FEATURES]
    y = train_df["label"]
    groups = train_df["match_id"]

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    agg = defaultdict(float)
    for feat, imp in importances.items():
        base = feat[3:] if feat.startswith(("p1_", "p2_")) else feat
        agg[base] += imp
    agg = pd.Series(agg).sort_values(ascending=False)
    print("\nFeature importances (p1/p2 combined, per concept):")
    for base, imp in agg.items():
        print(f"  {base:<25} {imp:.3f}")

    return model, feature_medians


def cv_brier(model, X, y, groups, feature_cols):
    gkf = GroupKFold(n_splits=5)
    briers, accs = [], []
    Xc = X[feature_cols]
    for tr, te in gkf.split(Xc, y, groups):
        fm = RandomForestClassifier(**RF_PARAMS)
        fm.fit(Xc.iloc[tr], y.iloc[tr])
        proba = fm.predict_proba(Xc.iloc[te])[:, 1]
        briers.append(brier_score_loss(y.iloc[te], proba))
        accs.append(accuracy_score(y.iloc[te], (proba >= 0.5).astype(int)))
    return np.mean(briers), np.std(briers), np.mean(accs)


def build_matchup(player_stats, prof_idx, p1_name, p2_name, as_of_date, as_of_year, feature_medians):
    p1 = prof_idx.loc[p1_name]
    p2 = prof_idx.loc[p2_name]
    n1 = normalise(p1["sackmann_name"])
    n2 = normalise(p2["sackmann_name"])

    row = player_stats.build_feature_vector(
        n1,
        n2,
        p1["rank"],
        p2["rank"],
        as_of_date,
        year=as_of_year,
        medians=feature_medians,
    )
    return pd.DataFrame([row])[FEATURES]


def precompute_win_probs(draw, model, player_stats, prof_idx, as_of_date, as_of_year, feature_medians):
    names = [p["name"] for p in draw]
    if len(set(names)) != len(names):
        raise ValueError("Draw contains duplicate player names; cache keys would collide.")

    pairs = [tuple(sorted((a, b))) for a, b in combinations(names, 2)]
    batch = pd.concat(
        [build_matchup(player_stats, prof_idx, lo, hi, as_of_date, as_of_year, feature_medians) for lo, hi in pairs],
        ignore_index=True,
    )
    probs = model.predict_proba(batch)[:, 1]
    return {pair: float(p) for pair, p in zip(pairs, probs)}


def simulate_tournament(draw, win_prob):
    round_players = list(draw)
    while len(round_players) > 1:
        next_round = []
        for i in range(0, len(round_players), 2):
            p1, p2 = round_players[i], round_players[i + 1]
            n1, n2 = p1["name"], p2["name"]
            if n1 < n2:
                prob_p1 = win_prob[(n1, n2)]
            else:
                prob_p1 = 1.0 - win_prob[(n2, n1)]
            winner = p1 if random.random() < prob_p1 else p2
            next_round.append(winner)
        round_players = next_round
    return round_players[0]


def main():
    matches = load_matches(matches_path)
    profiles = pd.read_csv(profiles_path)
    prof_idx = profiles.set_index("name")

    player_stats = Player_Stats()
    training_rows = build_training_rows(matches, player_stats)
    train_df = pd.DataFrame(training_rows)

    model, feature_medians = train_model(train_df)

    X = train_df[FEATURES]
    y = train_df["label"]
    groups = train_df["match_id"]
    b_full, s_full, a_full = cv_brier(model, X, y, groups, FEATURES)
    print(f"  brier: {b_full:.4f} +/- {s_full:.4f}  acc={a_full:.3f}")

    with open(draw_path, "r") as f:
        draw = json.load(f)

    as_of_date = matches["date"].max()
    as_of_year = int(as_of_date.year)
    win_prob = precompute_win_probs(draw, model, player_stats, prof_idx, as_of_date, as_of_year, feature_medians)

    n_sims = 50000
    results = Counter()
    for _ in range(n_sims):
        champion = simulate_tournament(draw, win_prob)
        results[champion["name"]] += 1

    print("\n--- Wimbledon 2026 Championship Probabilities ---")
    for player, wins in results.most_common():
        probability = wins / n_sims * 100
        if probability >= 1:
            print(f"  {player:<35} {probability:.1f}%")


if __name__ == "__main__":
    main()

