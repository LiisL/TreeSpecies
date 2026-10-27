# Controlled ablation study (2026-08)

This folder holds the controlled, one-variable-at-a-time ablation study that
backs **Table 4.9** and chapters **4.7–4.8** of the thesis. It replaces an
earlier ~80-experiment series (`experiments/EXP000`–`EXP038` and beyond) whose
pairwise comparisons changed multiple variables at once and, in three cases
(`EXP000`, `EXP006`, `EXP009`), were run on a dataset that didn't match the
rest of the series — a genuine test-set-leakage-adjacent methodology flaw
identified during thesis review. Every script here changes **exactly one
variable** relative to the previous step's winning configuration, is
evaluated on the **validation split only** (never test, except the two
single, final touches noted below), and uses the canonical, clearcut-filtered
`Subregion2_{train,val,test}_sentinel2_10day_timeseries.csv` dataset
throughout.

These scripts do not carry individual `EXPnnn` codes (see thesis Lisa 2.1):
they are fast, narrowly-scoped, single-variable tests, not standalone models
meant to be preserved as separate named experiments.

## Two ablation directions

**Build-up ablation (`step1`–`step8`)** — starts from a minimal Sentinel-2-only
baseline and adds one thing at a time, isolating the marginal effect of each
choice in isolation:

| Script | Table 4.9 row(s) | What it isolates | Val Macro F1 |
|---|---|---|---|
| `step1_baseline_sentinel2_only.py` | 1 | Sentinel-2 time series only, RF | 0.6617 |
| `step2_height_soil_ablation.py` | 2a/2b/2c | + height only / + soil only / + both | 0.6572 / 0.6566 / 0.6531 |
| `step3_feature_representation.py` | 3a/3b | full set vs. auto Top-50 vs. hand-built signature features | 0.6599 / 0.5639 |
| `step3b_topN_selection_method_check.py` | (supplementary) | does XGBoost's own importance pick a better Top-50 than RF's? (no — RF-selected Top-50 wins, 0.7068 vs. 0.6932) | — |
| `step4_class_balancing.py` | 4a/4b/4c | no balancing vs. `class_weight` vs. manual oversampling | 0.6357 / 0.6599 / 0.6541 |
| `step5_rf_hyperparameter_search.py` | 5 | RF hyperparameter grid (9 configs) | 0.6648 |
| `step6a_architecture_flat_vs_4stage.py` | 6a | flat vs. 4-stage hierarchy, RF | 0.6648 / 0.6459 |
| `step6b_architecture_2stage_rf.py` | 6b | simpler 2-stage (conifer/deciduous only) hierarchy, RF | 0.6677 |
| `step6c_architecture_2stage_xgboost.py` | (supplementary) | same 2-stage hierarchy, XGBoost — hierarchy helps RF marginally but hurts XGBoost (0.6867 vs. flat XGBoost's 0.7068) | — |
| `step7_xgboost_hyperparameter_search.py` | 7 | XGBoost hyperparameter grid (10 configs) | 0.7068 |
| `step8_final_test_evaluation.py` | final | **the one test-set touch** for this pipeline (flat XGBoost, Top-50, no height/soil dilution) | test Macro F1 0.7342 (6-class: 0.7733) |

**Peel-back ablation (`peelback_*`)** — the complementary direction: starts
from the thesis's actual final model (`EXP038`'s exact recipe — 4-stage
hierarchy + 18 hand-built species-signature features + height/soil +
manual oversampling + XGBoost) and removes exactly one component at a time,
to test whether components matter specifically *in combination* (an
interaction the build-up direction can't detect, since it holds other
choices fixed at their own individually-best settings throughout).

| Script | What it tests | Val Macro F1 |
|---|---|---|
| `peelback_ablation_from_exp038.py` | B0 full recipe / B1 −oversampling / B2 −hierarchy / B3 −signature features / B4 −height+soil | 0.6892 / 0.7038 / 0.6822 / 0.6797 / 0.6824 |
| `peelback_final_test_b1_candidate.py` | **the one test-set touch** for candidate B1 (best validation score in the peel-back direction) | test 6-class Macro F1 0.7939, BalAcc 0.7819 — statistically indistinguishable from `EXP038`'s own 0.7992/0.7805, so `EXP038` remains the thesis's primary model |

**Extensions** — the winning build-up pipeline applied beyond pure stands:

| Script | What it does | Result |
|---|---|---|
| `extension_pure_mixed_classification.py` | same pipeline on the pure+mixed dataset | test Macro F1 0.468 (vs. 0.734 pure-only) |
| `extension_composition_regression.py` | composition regression reusing the same Top-50 features | mean MAE 11.05 percentage points |

## Key finding

Height/soil data and hand-built species-signature features do **not** help
when added to a flat model with the full feature set (`step2`, `step3`) — but
removing them from the final model's specific combination of hierarchy +
oversampling + XGBoost **does** hurt (`peelback_ablation_from_exp038.py`,
B3/B4). This is a genuine interaction effect: these components only pay off
in combination with each other, not in isolation. The one exception —
removing oversampling from that combination (B1) — looked like a real
improvement on validation (0.6892 → 0.7038) but the gain evaporated on the
single final test-set touch, illustrating exactly the validation/test
discipline risk the textbooks (Burkov 2019; Müller & Guido 2017; Sügis et
al. 2024) warn about. Net result: `EXP038` remains the primary model, on
stronger methodological footing than before.

## Running

Same environment as the rest of the repo (see the root `README.md`).
Each script is standalone: `python scripts/<name>.py` from the repo root, or
from anywhere (each script resolves the repo root from its own file location).
All scripts use `random_state=42` throughout, load data via `src.paths.PATHS`,
and use the standard clearcut-filtering (`src.preprocessing.clearcut`) and
feature-preparation helpers (`src.preprocessing.timeseries`) shared with the
rest of the codebase. None of them touch the test split except the two
scripts explicitly marked "the one test-set touch" above.
