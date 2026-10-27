"""Age-stratified breakdown for EXP038 (hierarchical XGBoost, promoted to
primary model - see EXPERIMENTS.md), mirroring the same analysis already run
for EXP021/EXP029/EXP032 (see EXPERIMENTS.md's age-effect follow-ups). Pure
post-hoc analysis of EXP038's own saved test predictions - does not retrain
anything.

Why this is needed now: the thesis's primary model changed from EXP032 (RF)
to EXP038 (XGBoost, same 4-stage hierarchy and feature set, better on every
metric), but EXP038's age-stratified numbers were never computed.
"""

import os
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import evaluate_model
from src.paths import PATHS

# =========================================================
# FILES
# =========================================================

PRED_FILE = os.path.join(PATHS.outputs_tables, "EXP038_test_predictions.csv")
SPLIT_FILE = os.path.join(PATHS.processed, "final_split_clean.gpkg")

OUT_DIR = os.path.join(PATHS.outputs_analysis, "EXP038_age_split_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

GPKG_OUT = os.path.join(OUT_DIR, "EXP038_age_split_test_predictions.gpkg")
SUMMARY_OUT = os.path.join(PATHS.outputs_tables, "EXP038_age_split_summary.csv")
PER_SPECIES_OUT = os.path.join(PATHS.outputs_tables, "EXP038_age_split_per_species_f1.csv")

SUBREGION = "Subregion2"
AGE_THRESHOLD = 20

# =========================================================
# 1. LOAD PREDICTIONS AND STAND INFO (same join logic as EXP032's script)
# =========================================================

preds = pd.read_csv(PRED_FILE)
print("Test predictions:", len(preds))

stands = gpd.read_file(SPLIT_FILE)

test_stands = stands[
    (stands["subregion"] == SUBREGION) & (stands["dataset_split"] == "test")
]

age_varies = test_stands.groupby("id_left")["vanus"].nunique()
assert (age_varies <= 1).all(), "vanus is not constant within a stand - check the data"

test_stands = test_stands.drop_duplicates(subset="id_left")[["id_left", "vanus", "geometry"]]

# =========================================================
# 2. JOIN PREDICTIONS + AGE + GEOMETRY
# =========================================================

merged = test_stands.merge(preds, left_on="id_left", right_on="id", how="inner")

print("Predictions after joining with stand info:", len(merged))
assert len(merged) == len(preds), (
    "Some test predictions could not find a matching stand in "
    "final_split_clean.gpkg - check the subregion/id filter"
)

merged["correct"] = merged["y_true"] == merged["y_pred"]
merged["age_group"] = merged["vanus"].apply(
    lambda v: f"<{AGE_THRESHOLD}y" if v < AGE_THRESHOLD else f">={AGE_THRESHOLD}y"
)

print("\nStands by age group:")
print(merged["age_group"].value_counts())

# =========================================================
# 3. OVERALL + STAGE-1 (GROUP) CLASSIFICATION REPORTS BY AGE GROUP
# =========================================================

summary_rows = []
for group_name in [f"<{AGE_THRESHOLD}y", f">={AGE_THRESHOLD}y"]:
    subset = merged[merged["age_group"] == group_name]

    metrics = evaluate_model(
        f"EXP038 TEST - age {group_name} (n={len(subset)}) - final species",
        subset["y_true"],
        subset["y_pred"],
    )
    metrics["age_group"] = group_name
    metrics["n"] = len(subset)

    stage1_metrics = evaluate_model(
        f"EXP038 TEST - age {group_name} (n={len(subset)}) - stage 1 (group)",
        subset["stage1_group_true"],
        subset["stage1_group_pred"],
    )
    metrics["stage1_group_accuracy"] = stage1_metrics["accuracy"]
    metrics["stage1_group_macro_f1"] = stage1_metrics["macro_f1"]

    hb_subset = subset[subset["y_true"] == "HB"]
    if len(hb_subset) > 0:
        hb_recall = (hb_subset["y_pred"] == "HB").mean()
        hb_pred_as_hb = subset[subset["y_pred"] == "HB"]
        hb_precision = (hb_pred_as_hb["y_true"] == "HB").mean() if len(hb_pred_as_hb) > 0 else float("nan")
        metrics["hb_support"] = len(hb_subset)
        metrics["hb_recall"] = hb_recall
        metrics["hb_precision"] = hb_precision
        print(f"HB ({group_name}): support={len(hb_subset)}, recall={hb_recall:.3f}, precision={hb_precision:.3f}")

    summary_rows.append(metrics)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
print("\nAge-split summary saved:", SUMMARY_OUT)
print(summary_df.to_string(index=False))

# =========================================================
# 4. PER-SPECIES F1 BY AGE GROUP (Table-4.4-style breakdown)
# =========================================================

from sklearn.metrics import f1_score

species_list = sorted(merged["y_true"].unique())
per_species_rows = []
for group_name in [f"<{AGE_THRESHOLD}y", f">={AGE_THRESHOLD}y"]:
    subset = merged[merged["age_group"] == group_name]
    f1_per_class = f1_score(subset["y_true"], subset["y_pred"], labels=species_list, average=None, zero_division=0)
    support_per_class = subset["y_true"].value_counts().reindex(species_list, fill_value=0)
    for sp, f1, n in zip(species_list, f1_per_class, support_per_class):
        per_species_rows.append({"age_group": group_name, "species": sp, "f1": f1, "support": n})

per_species_df = pd.DataFrame(per_species_rows)
per_species_pivot = per_species_df.pivot(index="species", columns="age_group", values="f1")
per_species_df.to_csv(PER_SPECIES_OUT, index=False, encoding="utf-8-sig")
print("\nPer-species F1 by age group:")
print(per_species_pivot.to_string())
print("\nSaved:", PER_SPECIES_OUT)

# =========================================================
# 5. SAVE FOR QGIS
# =========================================================

merged.to_file(GPKG_OUT, driver="GPKG")
print("\nAge-group GPKG saved:", GPKG_OUT)
