# -*- coding: utf-8 -*-
"""Step 4 of the controlled ablation: CLASS BALANCING comparison, holding
everything else fixed (clearcut-filtered canonical Subregion2 data, Top-50
auto-selected features from Step 3 - the winning feature representation,
same RF hyperparameters otherwise).

Three variants:
  (a) No balancing          - plain RF, no class_weight, no oversampling
  (b) class_weight=balanced - reuses Step 3b setup (already the baseline
                              used in Steps 1-3)
  (c) Manual oversampling   - RF without class_weight, TRAIN oversampled
                              per EXP021's SAMPLING_STRATEGY
"""
import os
import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.preprocessing.timeseries import (
    long_to_wide, merge_aux, encode_categoricals_consistently,
    align_columns, drop_all_nan_train_columns,
)
from src.preprocessing.clearcut import remove_clearcut_ids
from src.models.sampling import manual_oversample
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

SAMPLING_STRATEGY = {"HB": 1200, "LM": 1200, "KU": 1600, "LV": 2000, "MUU": 500}

# =========================================================
# LOAD + PREP (identical to Steps 1-3)
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

train_wide, val_wide, test_wide = encode_categoricals_consistently(
    train_wide, val_wide, test_wide, AUX_CATEGORICAL_COLS
)
train_wide, val_wide, test_wide, feature_cols = align_columns(
    train_wide, val_wide, test_wide, ID_COL, TARGET_COL
)

X_train = train_wide[feature_cols].copy()
y_train = train_wide[TARGET_COL].copy()
X_val = val_wide[feature_cols].copy()
y_val = val_wide[TARGET_COL].copy()
X_test = test_wide[feature_cols].copy()
y_test = test_wide[TARGET_COL].copy()

X_train, X_val, X_test, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_test)

# reproduce Step 3's Top-50 selection: importances from a class_weight=balanced
# full-set RF (same procedure as Step 3a/3b)
imputer0 = SimpleImputer(strategy="median")
X_train_imp0 = imputer0.fit_transform(X_train)
rf0 = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                              class_weight="balanced", random_state=42, n_jobs=-1)
rf0.fit(X_train_imp0, y_train)
importance_df = pd.DataFrame({
    "feature": X_train.columns.tolist(),
    "importance": rf0.feature_importances_
}).sort_values("importance", ascending=False)
top50 = importance_df.head(50)["feature"].tolist()

X_train_50 = X_train[top50].copy()
X_val_50 = X_val[top50].copy()

results = {}

# =========================================================
# (a) NO BALANCING - plain RF
# =========================================================
imputer_a = SimpleImputer(strategy="median")
X_train_a = imputer_a.fit_transform(X_train_50)
X_val_a = imputer_a.transform(X_val_50)

rf_a = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
rf_a.fit(X_train_a, y_train)
pred_a = rf_a.predict(X_val_a)
results["(a) No balancing"] = evaluate_model("STEP4a No balancing - VALIDATION", y_val, pred_a)

# =========================================================
# (b) class_weight=balanced (= Step 3b, recomputed here on same split objects)
# =========================================================
imputer_b = SimpleImputer(strategy="median")
X_train_b = imputer_b.fit_transform(X_train_50)
X_val_b = imputer_b.transform(X_val_50)

rf_b = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                               class_weight="balanced", random_state=42, n_jobs=-1)
rf_b.fit(X_train_b, y_train)
pred_b = rf_b.predict(X_val_b)
results["(b) class_weight=balanced"] = evaluate_model("STEP4b class_weight=balanced - VALIDATION", y_val, pred_b)

# =========================================================
# (c) MANUAL OVERSAMPLING (no class_weight)
# =========================================================
X_train_os, y_train_os = manual_oversample(
    X_train_50, y_train, SAMPLING_STRATEGY, TARGET_COL, random_state=42
)
print("\nClass counts after oversampling:")
print(y_train_os.value_counts())

imputer_c = SimpleImputer(strategy="median")
X_train_c = imputer_c.fit_transform(X_train_os)
X_val_c = imputer_c.transform(X_val_50)

rf_c = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
rf_c.fit(X_train_c, y_train_os)
pred_c = rf_c.predict(X_val_c)
results["(c) Manual oversampling"] = evaluate_model("STEP4c Manual oversampling - VALIDATION", y_val, pred_c)

# =========================================================
# SUMMARY
# =========================================================
print("\n\n" + "=" * 70)
print("STEP 4 SUMMARY (class balancing, Top-50 features, validation)")
print("=" * 70)
for label, m in results.items():
    print(f"{label:30s} Acc={m['accuracy']:.3f}  BalAcc={m['balanced_accuracy']:.3f}  MacroF1={m['macro_f1']:.3f}  Kappa={m['kappa']:.3f}")

joblib.dump(
    {"rf_no_balance": rf_a, "rf_class_weight": rf_b, "rf_oversample": rf_c,
     "feature_cols_top50": top50, "results": results},
    os.path.join(PATHS.models_trained, "STEP4_class_balancing.joblib")
)
print("\nSaved: STEP4_class_balancing.joblib")
