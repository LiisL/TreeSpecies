# -*- coding: utf-8 -*-
"""Follow-up to Step 3/7: does XGBoost's OWN feature importance pick a
better Top-50 than the RF importance used throughout Steps 3-8? Same full
feature set (S2 + elevation + soil one-hot), same clearcut-filtered
canonical data, Step 7's winning XGBoost hyperparameters used both to
derive importances and to fit the final Top-50 model.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

BEST_XGB_KWARGS = dict(n_estimators=500, max_depth=5, learning_rate=0.08,
                        min_child_weight=5, gamma=0.1, random_state=42, n_jobs=-1,
                        eval_metric="mlogloss")

# =========================================================
# LOAD + PREP (identical to Steps 1-7)
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

# =========================================================
# (0) recap: RF-based Top-50 (used throughout Steps 3-8)
# =========================================================
imputer0 = SimpleImputer(strategy="median")
X_train_imp0 = imputer0.fit_transform(X_train)
rf0 = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                              class_weight="balanced", random_state=42, n_jobs=-1)
rf0.fit(X_train_imp0, y_train)
importance_rf = pd.DataFrame({
    "feature": X_train.columns.tolist(), "importance": rf0.feature_importances_
}).sort_values("importance", ascending=False)
top50_rf = importance_rf.head(50)["feature"].tolist()

# =========================================================
# (1) NEW: XGBoost-based Top-50 (importances from a full-feature XGBoost,
#     Step 7's winning hyperparameters, trained on TRAIN only)
# =========================================================
le0 = LabelEncoder()
y_train_enc0 = le0.fit_transform(y_train)
sw0 = compute_sample_weight("balanced", y_train_enc0)

xgb_full = XGBClassifier(**BEST_XGB_KWARGS)
xgb_full.fit(X_train_imp0, y_train_enc0, sample_weight=sw0)
importance_xgb = pd.DataFrame({
    "feature": X_train.columns.tolist(), "importance": xgb_full.feature_importances_
}).sort_values("importance", ascending=False)
top50_xgb = importance_xgb.head(50)["feature"].tolist()

overlap = set(top50_rf) & set(top50_xgb)
print(f"\nOverlap between RF-Top50 and XGB-Top50: {len(overlap)} / 50 features shared")

elev_in_xgb50 = [f for f in top50_xgb if f.startswith("elev")]
soil_in_xgb50 = [f for f in top50_xgb if f.startswith("soil")]
print("Elevation features in XGB-Top50:", elev_in_xgb50)
print("Soil one-hot features in XGB-Top50:", soil_in_xgb50)

# =========================================================
# Evaluate flat XGBoost (Step 7 config) on each Top-50 variant, VALIDATION
# =========================================================
def eval_top50(top50, label):
    X_tr = X_train[top50].copy()
    X_v = X_val[top50].copy()
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_v_imp = imputer.transform(X_v)
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_train)
    sw = compute_sample_weight("balanced", y_tr_enc)
    model = XGBClassifier(**BEST_XGB_KWARGS)
    model.fit(X_tr_imp, y_tr_enc, sample_weight=sw)
    pred = le.inverse_transform(model.predict(X_v_imp))
    return evaluate_model(f"{label} - VALIDATION", y_val, pred)

metrics_rf50 = eval_top50(top50_rf, "RF-based Top-50 + flat XGBoost")
metrics_xgb50 = eval_top50(top50_xgb, "XGBoost-based Top-50 + flat XGBoost")

print("\n\n" + "=" * 70)
print("SUMMARY: feature-selection method x final XGBoost model (validation)")
print("=" * 70)
print(f"RF-based Top-50   -> XGBoost   MacroF1={metrics_rf50['macro_f1']:.4f}  (Step 7 result)")
print(f"XGBoost-based Top-50 -> XGBoost   MacroF1={metrics_xgb50['macro_f1']:.4f}  (this run)")

joblib.dump(
    {"top50_rf": top50_rf, "top50_xgb": top50_xgb, "importance_rf": importance_rf,
     "importance_xgb": importance_xgb, "metrics_rf50": metrics_rf50, "metrics_xgb50": metrics_xgb50},
    os.path.join(PATHS.models_trained, "STEP3b_xgb_top50_check.joblib")
)
print("\nSaved: STEP3b_xgb_top50_check.joblib")
