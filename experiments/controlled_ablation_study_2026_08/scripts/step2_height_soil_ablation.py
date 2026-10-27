# -*- coding: utf-8 -*-
"""Clean 4-way ablation isolating height vs soil vs both, addressing
reviewer point 9: same dataset, same spatial split, same feature
representation approach (full auto set, no Top-50/signature reduction -
matching Steps 1/2 exactly), same RF hyperparameters, same clearcut
filtering. Only the aux-feature inclusion changes across the 4 rows:
  (a) Sentinel-2 only                  = Step 1 (already run, reused)
  (b) Sentinel-2 + height only          (NEW)
  (c) Sentinel-2 + soil only            (NEW)
  (d) Sentinel-2 + height + soil        = Step 2 (already run, reused)
"""
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.preprocessing.timeseries import (
    long_to_wide, merge_aux, encode_categoricals_consistently,
    align_columns, drop_all_nan_train_columns,
)
from src.preprocessing.clearcut import remove_clearcut_ids
from src.evaluation.metrics import evaluate_model
from src.paths import PATHS

DATA_DIR = PATHS.intermediate
TRAIN_FILE = os.path.join(DATA_DIR, "Subregion2_train_sentinel2_10day_timeseries.csv")
VAL_FILE = os.path.join(DATA_DIR, "Subregion2_val_sentinel2_10day_timeseries.csv")
TEST_FILE = os.path.join(DATA_DIR, "Subregion2_test_sentinel2_10day_timeseries.csv")
OVERLAP_FILE = os.path.join(PATHS.reference, "clearcut_overlap", "stands_overlapping_clearcuts_over20pct.csv")
AUX_FILE = os.path.join(DATA_DIR, "stand_aux_features.csv")

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

TS_FEATURE_COLS = [
    "B2_mean", "B3_mean", "B4_mean", "B5_mean", "B6_mean", "B7_mean",
    "B8_mean", "B8A_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDRE_mean", "NDMI_mean", "NBR_mean"
]
AUX_NUMERIC_COLS = ["elev_mean", "elev_median", "elev_std", "elev_min", "elev_max"]
AUX_CATEGORICAL_COLS = ["soil_main"]

RF_KWARGS = dict(n_estimators=400, max_depth=None, min_samples_leaf=2,
                  class_weight="balanced", random_state=42, n_jobs=-1)

# =========================================================
# LOAD (long form, shared across all 4 variants)
# =========================================================
train_long = pd.read_csv(TRAIN_FILE)
val_long = pd.read_csv(VAL_FILE)
test_long = pd.read_csv(TEST_FILE)
aux = pd.read_csv(AUX_FILE)

train_long, val_long, test_long = remove_clearcut_ids(
    train_long, val_long, test_long, OVERLAP_FILE, ID_COL, TARGET_COL
)

train_wide_base = long_to_wide(train_long, "train", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
val_wide_base = long_to_wide(val_long, "val", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
test_wide_base = long_to_wide(test_long, "test", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)


def run_variant(label, use_height, use_soil):
    train_wide = train_wide_base.copy()
    val_wide = val_wide_base.copy()
    test_wide = test_wide_base.copy()

    numeric_cols = AUX_NUMERIC_COLS if use_height else []
    categorical_cols = AUX_CATEGORICAL_COLS if use_soil else []

    if numeric_cols or categorical_cols:
        train_wide = merge_aux(train_wide, aux, "train", ID_COL, numeric_cols, categorical_cols)
        val_wide = merge_aux(val_wide, aux, "val", ID_COL, numeric_cols, categorical_cols)
        test_wide = merge_aux(test_wide, aux, "test", ID_COL, numeric_cols, categorical_cols)

    if categorical_cols:
        train_wide, val_wide, test_wide = encode_categoricals_consistently(
            train_wide, val_wide, test_wide, categorical_cols
        )

    train_wide, val_wide, test_wide, feature_cols = align_columns(
        train_wide, val_wide, test_wide, ID_COL, TARGET_COL
    )

    X_train = train_wide[feature_cols].copy()
    y_train = train_wide[TARGET_COL].copy()
    X_val = val_wide[feature_cols].copy()
    y_val = val_wide[TARGET_COL].copy()

    X_train, X_val, _, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_val.copy())
    print(f"\n{label}: feature count = {X_train.shape[1]}")

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    rf = RandomForestClassifier(**RF_KWARGS)
    rf.fit(X_train_imp, y_train)
    pred = rf.predict(X_val_imp)
    return evaluate_model(f"{label} - VALIDATION", y_val, pred)


results = {}
results["(a) Sentinel-2 only"] = run_variant("(a) S2 only", use_height=False, use_soil=False)
results["(b) Sentinel-2 + height"] = run_variant("(b) S2 + height", use_height=True, use_soil=False)
results["(c) Sentinel-2 + soil"] = run_variant("(c) S2 + soil", use_height=False, use_soil=True)
results["(d) Sentinel-2 + height + soil"] = run_variant("(d) S2 + height + soil", use_height=True, use_soil=True)

print("\n\n" + "=" * 70)
print("HEIGHT vs SOIL ABLATION SUMMARY (validation, RF, canonical clearcut-filtered data)")
print("=" * 70)
for label, m in results.items():
    print(f"{label:35s} Acc={m['accuracy']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  MacroF1={m['macro_f1']:.4f}  Kappa={m['kappa']:.4f}")
