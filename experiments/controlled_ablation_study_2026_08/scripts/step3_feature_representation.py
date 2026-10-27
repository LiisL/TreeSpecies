# -*- coding: utf-8 -*-
"""Step 3 of the controlled ablation: feature REPRESENTATION comparison,
holding everything else fixed (clearcut-filtered canonical Subregion2 data,
same RF config, class_weight=balanced, no oversampling - that's Step 4).

Three variants, all built on top of Step 2's full Sentinel-2 + height/soil
feature set:
  (a) Full set               -> reuses Step 2 result (0.6531 Macro F1)
  (b) Auto Top-50 (by RF importance from the full-set model)
  (c) Hand-built species-signature features (EXP021 style) + height/soil
      (NOT the full S2 set - just the 18 hand-crafted signature features
      plus height/soil, i.e. what EXP021 actually used as its feature set)
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
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


def safe_mean(df, cols):
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return np.nan
    return df[existing].mean(axis=1)


def safe_diff(df, col_late, col_early):
    if col_late in df.columns and col_early in df.columns:
        return df[col_late] - df[col_early]
    return np.nan


def add_species_signature_features(df):
    df = df.copy()
    if "elev_max" in df.columns and "elev_std" in df.columns:
        df["HB_height_index"] = df["elev_max"] + df["elev_std"]
        df["HB_height_product"] = df["elev_max"] * df["elev_std"]
    else:
        df["HB_height_index"] = np.nan
        df["HB_height_product"] = np.nan
    if all(c in df.columns for c in ["elev_std", "elev_mean"]):
        df["HB_structure_ratio"] = df["elev_std"] / (df["elev_mean"] + 1)
    else:
        df["HB_structure_ratio"] = np.nan
    df["HB_spring_index"] = safe_mean(df, ["NDVI_mean_2025_05_01", "NDMI_mean_2025_05_01", "NBR_mean_2025_05_01"])
    df["HB_autumn_ndre"] = safe_mean(df, ["NDRE_mean_2025_09_28", "NDRE_mean_2025_10_08", "NDRE_mean_2025_10_18"])
    df["KS_moisture_index"] = safe_mean(df, ["NDMI_mean_2025_09_28", "NDRE_mean_2025_09_28", "B11_mean_2025_09_28"])
    df["KU_winter_index"] = safe_mean(df, ["B11_mean_2025_02_20", "B12_mean_2025_02_20", "NDVI_mean_2025_02_20"])
    df["KU_autumn_index"] = safe_mean(df, ["NDRE_mean_2025_09_28", "NDVI_mean_2025_10_18", "NDMI_mean_2025_09_28"])
    df["LM_autumn_index"] = safe_mean(df, ["NDRE_mean_2025_09_28", "NDVI_mean_2025_09_28", "B8_mean_2025_09_28", "B7_mean_2025_09_28"])
    df["LV_july_index"] = safe_mean(df, ["B6_mean_2025_07_20", "B7_mean_2025_07_20", "B8_mean_2025_07_20", "B8A_mean_2025_07_20"])
    df["LV_july_swir_index"] = safe_mean(df, ["B11_mean_2025_07_20", "B12_mean_2025_07_20"])
    df["MA_spring_nir"] = safe_mean(df, ["B8_mean_2025_05_01", "B8A_mean_2025_05_01", "B6_mean_2025_05_01", "NDVI_mean_2025_05_01"])
    df["NDVI_change_spring"] = safe_diff(df, "NDVI_mean_2025_05_21", "NDVI_mean_2025_04_11")
    df["NDVI_change_summer"] = safe_diff(df, "NDVI_mean_2025_07_20", "NDVI_mean_2025_05_21")
    df["NDVI_change_autumn"] = safe_diff(df, "NDVI_mean_2025_10_18", "NDVI_mean_2025_09_08")
    df["NDRE_change_autumn"] = safe_diff(df, "NDRE_mean_2025_10_18", "NDRE_mean_2025_09_08")
    df["NDMI_change_autumn"] = safe_diff(df, "NDMI_mean_2025_10_18", "NDMI_mean_2025_09_08")
    df["NBR_change_autumn"] = safe_diff(df, "NBR_mean_2025_10_18", "NBR_mean_2025_09_08")
    signature_cols = [
        "HB_height_index", "HB_height_product", "HB_structure_ratio", "HB_spring_index", "HB_autumn_ndre",
        "KS_moisture_index", "KU_winter_index", "KU_autumn_index", "LM_autumn_index",
        "LV_july_index", "LV_july_swir_index", "MA_spring_nir",
        "NDVI_change_spring", "NDVI_change_summer", "NDVI_change_autumn",
        "NDRE_change_autumn", "NDMI_change_autumn", "NBR_change_autumn",
    ]
    return df, signature_cols


# =========================================================
# LOAD + PREP (identical to Step 2)
# =========================================================
train_long = pd.read_csv(TRAIN_FILE)
val_long = pd.read_csv(VAL_FILE)
test_long = pd.read_csv(TEST_FILE)
aux = pd.read_csv(AUX_FILE)

train_long, val_long, test_long = remove_clearcut_ids(
    train_long, val_long, test_long, OVERLAP_FILE, ID_COL, TARGET_COL
)

train_wide = long_to_wide(train_long, "train", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
val_wide = long_to_wide(val_long, "val", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
test_wide = long_to_wide(test_long, "test", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)

train_wide = merge_aux(train_wide, aux, "train", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
val_wide = merge_aux(val_wide, aux, "val", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
test_wide = merge_aux(test_wide, aux, "test", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)

# keep an un-signature copy for the "full set" + "Top-50" variants
train_wide_full = train_wide.copy()
val_wide_full = val_wide.copy()
test_wide_full = test_wide.copy()

train_wide_full, val_wide_full, test_wide_full = encode_categoricals_consistently(
    train_wide_full, val_wide_full, test_wide_full, AUX_CATEGORICAL_COLS
)
train_wide_full, val_wide_full, test_wide_full, feature_cols_full = align_columns(
    train_wide_full, val_wide_full, test_wide_full, ID_COL, TARGET_COL
)

X_train_full = train_wide_full[feature_cols_full].copy()
y_train = train_wide_full[TARGET_COL].copy()
X_val_full = val_wide_full[feature_cols_full].copy()
y_val = val_wide_full[TARGET_COL].copy()
X_test_full = test_wide_full[feature_cols_full].copy()
y_test = test_wide_full[TARGET_COL].copy()

X_train_full, X_val_full, X_test_full, feature_cols_full_used = drop_all_nan_train_columns(
    X_train_full, X_val_full, X_test_full
)

results = {}

# =========================================================
# (a) FULL SET - reuse Step 2 (do not re-report, just recompute for a clean
#     summary table; identical inputs/model to step2_add_height_soil.py)
# =========================================================
imputer_a = SimpleImputer(strategy="median")
X_train_a = imputer_a.fit_transform(X_train_full)
X_val_a = imputer_a.transform(X_val_full)

rf_a = RandomForestClassifier(**RF_KWARGS)
rf_a.fit(X_train_a, y_train)
pred_a = rf_a.predict(X_val_a)
results["(a) Full S2 + height/soil"] = evaluate_model("STEP3a Full set - VALIDATION", y_val, pred_a)

# =========================================================
# (b) AUTO TOP-50 (by importance from the full-set model just trained)
# =========================================================
importance_df = pd.DataFrame({
    "feature": X_train_full.columns.tolist(),
    "importance": rf_a.feature_importances_
}).sort_values("importance", ascending=False)
top50 = importance_df.head(50)["feature"].tolist()

X_train_b = X_train_full[top50].copy()
X_val_b = X_val_full[top50].copy()

imputer_b = SimpleImputer(strategy="median")
X_train_b_imp = imputer_b.fit_transform(X_train_b)
X_val_b_imp = imputer_b.transform(X_val_b)

rf_b = RandomForestClassifier(**RF_KWARGS)
rf_b.fit(X_train_b_imp, y_train)
pred_b = rf_b.predict(X_val_b_imp)
results["(b) Auto Top-50 features"] = evaluate_model("STEP3b Top-50 - VALIDATION", y_val, pred_b)

# =========================================================
# (c) HAND-BUILT SPECIES-SIGNATURE FEATURES (+ height/soil numeric, no full S2)
# =========================================================
train_sig, sig_cols = add_species_signature_features(train_wide)
val_sig, _ = add_species_signature_features(val_wide)
test_sig, _ = add_species_signature_features(test_wide)

train_sig, val_sig, test_sig = encode_categoricals_consistently(
    train_sig, val_sig, test_sig, AUX_CATEGORICAL_COLS
)

soil_dummy_cols = [c for c in train_sig.columns if c.startswith("soil_main_")]
signature_feature_cols = sig_cols + AUX_NUMERIC_COLS + soil_dummy_cols
signature_feature_cols = [c for c in signature_feature_cols if c in train_sig.columns]

X_train_c = train_sig[signature_feature_cols].copy()
X_val_c = val_sig[signature_feature_cols].copy()
X_test_c = test_sig[signature_feature_cols].copy()

X_train_c, X_val_c, X_test_c, sig_cols_used = drop_all_nan_train_columns(X_train_c, X_val_c, X_test_c)
print("\nSpecies-signature feature count (incl. height/soil):", X_train_c.shape[1])

imputer_c = SimpleImputer(strategy="median")
X_train_c_imp = imputer_c.fit_transform(X_train_c)
X_val_c_imp = imputer_c.transform(X_val_c)

rf_c = RandomForestClassifier(**RF_KWARGS)
rf_c.fit(X_train_c_imp, y_train)
pred_c = rf_c.predict(X_val_c_imp)
results["(c) Hand-built species-signature + height/soil"] = evaluate_model("STEP3c Species-signature - VALIDATION", y_val, pred_c)

# =========================================================
# SUMMARY
# =========================================================
print("\n\n" + "=" * 70)
print("STEP 3 SUMMARY (feature representation, validation, canonical data)")
print("=" * 70)
for label, m in results.items():
    print(f"{label:45s} Acc={m['accuracy']:.3f}  BalAcc={m['balanced_accuracy']:.3f}  MacroF1={m['macro_f1']:.3f}  Kappa={m['kappa']:.3f}")

joblib.dump(
    {"rf_full": rf_a, "rf_top50": rf_b, "rf_signature": rf_c,
     "feature_cols_full": feature_cols_full_used, "feature_cols_top50": top50,
     "feature_cols_signature": sig_cols_used, "results": results},
    os.path.join(PATHS.models_trained, "STEP3_feature_representation.joblib")
)
print("\nSaved: STEP3_feature_representation.joblib")
