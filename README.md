# Wimbledon 2026 — Random Forest Win Probability Model

Predicts the winner of the 2026 Wimbledon gentlemen's singles draw with a random forest trained on ATP match data (2015–2026), then simulates the tournament 50,000 times to get a per-player championship probability.

**Result:** 5-fold cross-validated Brier score of **0.2022 ± 0.0026** (accuracy 67.8%) on 3,311 grass-court matches, using a leakage-free feature pipeline and match-grouped CV.

## Data

- ATP match-level data 2015–2026 (via [tennisabstract.com](https://tennisabstract.com)), 32,237 matches total.
- Each match contributes two rows (winner-perspective, loser-perspective), giving 6,622 training rows.
- Features for a given match are built from `Player_Stats` state before `update_after_match` is called on that match, thereby ensuring no lookahead leakage from the match being predicted.

## Model choice: why random forest

Three properties of tennis-match data rule out a plain logistic regression and make a neural network impractical, while a random forest handles both:

- **Non-linearity.** Several features have diminishing or plateauing effects rather than linear ones — e.g. the gap between a bad serve and a good serve matters more than the gap between a good serve and a great one. `ace_rate`, `grass_serve_quality`, and `peak_rank` all show this pattern.
- **Multicollinearity.** The win-rate features (`grass_win_rate`, `ytd_win_rate`, `top10_win_rate`, `peak_wimb_rate`) and the serve features (`grass_serve_quality`, `ace_rate`, `second_serve_won_pct`) are correlated by construction (r = 0.5–0.7 between related pairs; `rank_diff` and `wimb_formula_diff` are r = -0.83). A regression's coefficients would be unstable under this; a tree ensemble is unaffected.
- **Non-independence of observations.** A player's form and fatigue carry from match to match within a tournament, so the assumption behind standard regression inference doesn't hold cleanly here either.

A neural network would in principle handle the non-linearity too, but grass is the shortest ATP season by a wide margin (~10% of tour matches), so the effective sample size is too small to train one without overfitting, and it would sacrifice the interpretability that matters for sanity-checking a sports model.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 1000 | More trees reduce the variance of the ensemble's averaged class-probability estimate. |
| `max_depth` | 15 | 19 features with paired (p1/p2) structure need enough depth to capture interaction effects between them. |
| `min_samples_leaf` | 25 | Regularizes against `max_depth=15` by requiring  historical support behind every split — important given the training set is only ~3,300 matches. |
| `criterion` | `log_loss` | Penalizes confident wrong predictions more than Gini does, which matters because the output used downstream is the probability itself, not just the argmax class. |

## Features

| Feature | Description |
|---|---|
| `rank_diff` | Log-difference of ATP rank (1→50 is not equivalent to 50→100, so raw rank difference would be misleading). |
| `peak_rank` | Player's best-ever ATP ranking. |
| `grass_win_rate` | Career win rate restricted to grass-court matches. |
| `peak_wimb_rate` | Best single-year win rate the player has posted at Wimbledon itself. |
| `ytd_win_rate` | Win rate in the current season, as a form signal. |
| `top10_win_rate` | Win rate specifically against top-10 opponents — a proxy for upset/deep-run potential. |
| `ace_rate`, `first_serve_pct`, `second_serve_won_pct` | Grass-restricted serve statistics. |
| `grass_serve_quality` | % of service points won, grass-restricted. Currently doesn't account for serve speed. |
| `wimb_formula_diff` | Proxy for Wimbledon's former seeding formula (ATP rank blended with a grass-specific rating), reconstructed from rolling grass results to test whether it would still carry predictive value if reinstated. |

## Results

**5-fold GroupKFold cross-validation** (folds split on `match_id`, so a match's winner-row and loser-row never land in different folds):

- Brier score: **0.2022 ± 0.0026**
- Accuracy: **67.8%**

**Feature importance** (p1/p2 pairs combined):

| Feature | Importance |
|---|---|
| `wimb_formula_diff` | 0.207 |
| `rank_diff` | 0.140 |
| `peak_rank` | 0.107 |
| `ytd_win_rate` | 0.082 |
| `ace_rate` | 0.074 |
| `grass_serve_quality` | 0.073 |
| `peak_wimb_rate` | 0.072 |
| `grass_win_rate` | 0.070 |
| `top10_win_rate` | 0.067 |
| `second_serve_won_pct` | 0.054 |
| `first_serve_pct` | 0.054 |

The reconstructed Wimbledon seeding formula and log rank-difference are the two strongest signals by a clear margin — a reasonable result, since both are themselves aggregations of a player's grass-specific track record.

**2026 championship probabilities** (50,000 tournament simulations from the actual draw, pre-tournament):

| Player | Win probability |
|---|---|
| Sinner | 30.1% |
| Zverev | 9.0% |
| Auger-Aliassime | 8.1% |
| Djokovic | 7.0% |
| Shelton | 6.6% |
| Fritz | 4.4% |
| De Minaur | 3.8% |
| *(remaining field ≥1%)* | — |

## Limitations

- **Grass-only training set is small.** Restricting to `is_grass` matches leaves 3,311 matches out of 32,237 logged (10.4%) — grass has the shortest ATP season of any surface. This is the main reason imputation (feature medians) is doing real work for players with thin grass history, and it's why `min_samples_leaf=25` is set as high as it is relative to the depth.
- **No time-decay on historical rates.** `grass_win_rate`, `top10_win_rate`, etc. are unweighted career rates — a win from 2015 counts identically to one from last month, despite tennis form and injury cycles moving faster than that.
- **Static, not conditional, simulation.** `simulate_tournament` uses the same pre-tournament `win_prob` for every round of the bracket, including the final. It doesn't re-condition on actual round results (an early upset is new information the model currently never sees).
- **Uncalibrated probabilities.** Random forests are known to compress probabilities toward 0.5 relative to true frequencies, particularly with a smoothing parameter like `min_samples_leaf=25`. No calibration layer (Platt/isotonic) or reliability check has been applied yet, so these probabilities are better trusted as a *ranking* of players than as literal frequencies.
- **Missing shot-level data.** Granular data — serve speed, rally shot placement, forehand/backhand pace — isn't available yet at the resolution needed for this project, but is being generated for future versions.
- **Random-fold rather than time-respecting validation.** GroupKFold prevents leakage within a match but still shuffles matches across years into folds, so the CV score can be modestly optimistic relative to a true walk-forward (train on past years only, test on the next) evaluation.

## Going forward

- Add probability calibration (`CalibratedClassifierCV`, isotonic) and report a reliability diagram before/after.
- Add a walk-forward validation report (train ≤ year Y, test year Y+1) alongside the GroupKFold number.
- Add a logistic regression baseline and a gradient-boosted tree (XGBoost/LightGBM) comparison to substantiate the random forest choice with evidence rather than assertion.
- Re-run inference round-by-round during the actual tournament instead of relying on a single pre-tournament snapshot.
- Compare the model's implied probabilities against prediction-market (Kalshi) or de-vigged sportsbook odds to check whether the model has real edge, measured on Brier score rather than P&L alone.

## Acknowledgments

Data generated by [tennisabstract.com](https://tennisabstract.com).
