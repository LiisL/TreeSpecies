"""Supervisor-requested follow-up analysis for EXP021 (best multiclass model):
does classification accuracy differ between young (<20y) and older (>=20y)
stands? Reads the per-stand test predictions saved by exp021.py and joins
them against `vanus` (age) and `geometry` from the canonical label file, so
results can be inspected numerically (classification_report per age group)
and visually (a GPKG to open in QGIS).

Does not retrain anything - pure post-hoc analysis of exp021.py's own
test-set predictions.
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

PRED_FILE = os.path.join(PATHS.outputs_tables, "EXP021_test_predictions.csv")
SPLIT_FILE = os.path.join(PATHS.processed, "final_split_clean.gpkg")

OUT_DIR = os.path.join(PATHS.outputs_analysis, "EXP021_age_split_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

GPKG_OUT = os.path.join(OUT_DIR, "EXP021_age_split_test_predictions.gpkg")

# EXP021 (like all Family A experiments) trains/tests only on Subregion2 -
# final_split_clean.gpkg covers all 8 subregions nationwide, so this must be
# filtered before joining or vanus/geometry could be pulled from unrelated
# stands.
SUBREGION = "Subregion2"
AGE_THRESHOLD = 20

# =========================================================
# 1. LOAD PREDICTIONS AND STAND INFO
# =========================================================

preds = pd.read_csv(PRED_FILE)
print("Test predictions:", len(preds))

stands = gpd.read_file(SPLIT_FILE)

test_stands = stands[
    (stands["subregion"] == SUBREGION) & (stands["dataset_split"] == "test")
]

# Each stand (id_left) can appear multiple times (one row per canopy
# layer/species), but age (vanus) and geometry are constant within a stand -
# verify this before deduplicating, rather than assuming it.
age_varies = test_stands.groupby("id_left")["vanus"].nunique()
assert (age_varies <= 1).all(), "vanus is not constant within a stand - check the data"

test_stands = test_stands.drop_duplicates(subset="id_left")[
    ["id_left", "vanus", "geometry"]
]

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
# 3. CLASSIFICATION REPORTS BY AGE GROUP
# =========================================================

for group_name in [f"<{AGE_THRESHOLD}y", f">={AGE_THRESHOLD}y"]:
    subset = merged[merged["age_group"] == group_name]
    evaluate_model(
        f"EXP021 TEST - age {group_name} (n={len(subset)})",
        subset["y_true"],
        subset["y_pred"],
    )

# =========================================================
# 4. SAVE FOR QGIS
# =========================================================

merged.to_file(GPKG_OUT, driver="GPKG")
print("\nAge-group GPKG saved:", GPKG_OUT)
print("Columns:", merged.columns.tolist())
