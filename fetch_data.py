"""
fetch_data.py
-------------
Downloads ATP match data from TennisMyLife and computes per-player feature
values needed by wimbledon_rf.py.

Source: https://stats.tennismylife.org/data/YYYY.csv
        Same CSV format as Jeff Sackmann's tennis_atp repo, updated daily.

Run this ONCE before training:
    python fetch_data.py

Outputs (written to ./data/):
    atp_matches_all.csv   — concatenated match history (2015–2026)
    player_profiles.csv   — one row per draw player with all model features:
        rank                  current ATP ranking (from draw JSON)
        recent_grass_success  grass win rate, 2025–2026 only
        peak_wimb_success     best single-season Wimbledon win rate ever
        two_month_success     win rate in the ~60 days before Wimbledon 2026
        peak_ranking          lowest (best) ranking number ever seen
        gs_champion           1 if player has ever won a Grand Slam, else 0
        ytd_win_rate          win rate in 2026 up to Wimbledon
        serve_quality         rolling career 1stSvPct * 1stWonPct + 2ndSvPct * 2ndWonPct
                              (weighted serve points won — best proxy without speed data)
        peak_wimb_success     already listed above; also used in rf training loop
"""

import os
import json
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TML_BASE         = "https://stats.tennismylife.org/data"
YEARS            = list(range(2015, 2027))
OUT_DIR          = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WIMBLEDON_START  = datetime(2026, 6, 30)
TWO_MONTH_CUTOFF = WIMBLEDON_START - timedelta(days=60)
YTD_CUTOFF       = datetime(2026, 1, 1)

os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Step 1 — Download CSVs
# ---------------------------------------------------------------------------
def fetch_year(year: int) -> pd.DataFrame | None:
    cache = os.path.join(OUT_DIR, f"atp_matches_{year}.csv")
    if os.path.exists(cache):
        print(f"  [cache] {year}")
        return pd.read_csv(cache, low_memory=False)

    url = f"{TML_BASE}/{year}.csv"
    print(f"  [fetch] {year} ... ", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with open(cache, "wb") as f:
            f.write(data)
        print("ok")
        return pd.read_csv(cache, low_memory=False)
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code} — skipping")
        return None
    except Exception as e:
        print(f"ERROR: {e} — skipping")
        return None


print("=== Downloading match data ===")
frames = [df for year in YEARS if (df := fetch_year(year)) is not None]

if not frames:
    raise RuntimeError(
        "No match data downloaded. Check your internet connection and that "
        "https://stats.tennismylife.org/data/2024.csv is reachable."
    )

matches = pd.concat(frames, ignore_index=True)
matches["date"] = pd.to_datetime(
    matches["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
)

# Coerce numeric columns that may have been read as strings
for col in ["winner_rank", "loser_rank",
            "w_ace", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms",
            "l_ace", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms"]:
    if col in matches.columns:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")

combined_path = os.path.join(OUT_DIR, "atp_matches_all.csv")
matches.to_csv(combined_path, index=False)
print(f"\nSaved {len(matches):,} total rows → {combined_path}\n")

# ---------------------------------------------------------------------------
# Step 2 — Load draw
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
draw_path  = os.path.join(script_dir, "data", "wimbledon_2026_draw.json")
with open(draw_path, "r") as f:
    draw = json.load(f)

player_names = {p["name"]: p["rank"] for p in draw}

def draw_name_to_sackmann(name: str) -> str:
    parts = name.split(", ", 1)
    if len(parts) == 2:
        last, first = parts
        return f"{first.title()} {last.title()}"
    return name.title()

def normalise(name: str) -> str:
    return str(name).strip().lower()

matches["_winner"] = matches["winner_name"].apply(normalise)
matches["_loser"]  = matches["loser_name"].apply(normalise)
matches["_is_gs"]  = matches["tourney_level"] == "G"
matches["_is_wimb"] = matches["tourney_name"].str.contains("Wimbledon", na=False)

# ---------------------------------------------------------------------------
# Step 3 — Pre-compute feature sub-datasets
# ---------------------------------------------------------------------------
grass        = matches[matches["surface"] == "Grass"]
wimb         = matches[matches["_is_wimb"]]
recent       = matches[matches["date"] >= TWO_MONTH_CUTOFF]
ytd          = matches[matches["date"] >= YTD_CUTOFF]
grass_recent = grass[grass["date"] >= datetime(2025, 1, 1)]

# Grand Slam champions: winner of a GS final
gs_finals    = matches[matches["_is_gs"] & (matches["round"] == "F")]
gs_champs    = set(gs_finals["_winner"].unique())

def win_rate(df: pd.DataFrame, player_norm: str) -> float:
    wins  = (df["_winner"] == player_norm).sum()
    total = wins + (df["_loser"] == player_norm).sum()
    return wins / total if total > 0 else float("nan")

def best_wimbledon_win_rate(player_norm: str) -> float:
    best = float("nan")
    for _, grp in wimb.groupby(wimb["date"].dt.year):
        rate = win_rate(grp, player_norm)
        if not pd.isna(rate) and (pd.isna(best) or rate > best):
            best = rate
    return best

def peak_ranking_fn(player_norm: str) -> float:
    w = matches.loc[matches["_winner"] == player_norm, "winner_rank"].dropna()
    l = matches.loc[matches["_loser"]  == player_norm, "loser_rank"].dropna()
    combined = pd.concat([w, l])
    return float(combined.min()) if len(combined) > 0 else float("nan")

def serve_quality(player_norm: str) -> float:
    """
    Serve quality = weighted serve points won rate.
    Formula: (1stIn/svpt)*( 1stWon/1stIn) + (1 - 1stIn/svpt)*(2ndWon/(svpt-1stIn))
           = 1stWon/svpt + 2ndWon/svpt
           = (1stWon + 2ndWon) / svpt
    This is simply the fraction of all serve points won — the cleanest
    single-number proxy for serve quality without speed data.
    Averaged across all matches where serve stats are available.
    """
    won = matches[matches["_winner"] == player_norm]
    lost = matches[matches["_loser"] == player_norm]

    total_svpt  = won["w_svpt"].sum()  + lost["l_svpt"].sum()
    total_won   = (won["w_1stWon"].sum() + won["w_2ndWon"].sum() +
                   lost["l_1stWon"].sum() + lost["l_2ndWon"].sum())

    if total_svpt > 0:
        return float(total_won / total_svpt)
    return float("nan")

# ---------------------------------------------------------------------------
# Step 4 — Compute player profiles
# ---------------------------------------------------------------------------
print("=== Computing player profiles ===")
rows = []
for draw_name, current_rank in player_names.items():
    sack_name = draw_name_to_sackmann(draw_name)
    norm      = normalise(sack_name)

    row = {
        "name":                 draw_name,
        "sackmann_name":        sack_name,
        "rank":                 current_rank,
        "recent_grass_success": win_rate(grass_recent, norm),
        "peak_wimb_success":    best_wimbledon_win_rate(norm),
        "two_month_success":    win_rate(recent, norm),
        "peak_ranking":         peak_ranking_fn(norm),
        "gs_champion":          1 if norm in gs_champs else 0,
        "ytd_win_rate":         win_rate(ytd, norm),
        "serve_quality":        serve_quality(norm),
    }

    has_nan = any(pd.isna(v) for v in [
        row["recent_grass_success"], row["peak_wimb_success"],
        row["two_month_success"],    row["peak_ranking"],
        row["ytd_win_rate"],         row["serve_quality"],
    ])
    tag = "(some NaN — will be median-imputed)" if has_nan else (
        f"grass={row['recent_grass_success']:.2f}  "
        f"wimb={row['peak_wimb_success']:.2f}  "
        f"2mo={row['two_month_success']:.2f}  "
        f"gs={'Y' if row['gs_champion'] else 'N'}  "
        f"ytd={row['ytd_win_rate']:.2f}  "
        f"srv={row['serve_quality']:.2f}"
    )
    print(f"  {sack_name:<30}  {tag}")
    rows.append(row)

profiles = pd.DataFrame(rows)

impute_cols = [
    "recent_grass_success", "peak_wimb_success", "two_month_success",
    "peak_ranking", "ytd_win_rate", "serve_quality",
]
for col in impute_cols:
    median   = profiles[col].median()
    n_filled = profiles[col].isna().sum()
    if n_filled:
        print(f"  Filling {n_filled} NaN in '{col}' with median {median:.3f}")
    profiles[col] = profiles[col].fillna(median)

profiles_path = os.path.join(OUT_DIR, "player_profiles.csv")
profiles.to_csv(profiles_path, index=False)
print(f"\nSaved {len(profiles)} player profiles → {profiles_path}")
print("All done. You can now run wimbledon_rf.py")