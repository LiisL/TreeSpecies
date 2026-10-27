# -*- coding: utf-8 -*-
"""Step 6c: a SIMPLER 2-stage hierarchy variant, requested as a follow-up to
Step 6's flat-vs-4-stage comparison. Same Top-50 features / RF hyperparams
(Step 5 winner) / class_weight=balanced / no oversampling as Steps 5-6.

  Stage 1: CONIFER vs DECIDUOUS only (no separate MUU group - "ainult
           okaspuud, lehtpuud")
  Stage 2: within CONIFER -> KU vs MA
           within DECIDUOUS -> HB vs KS vs LM vs LV all at once ("teises
           faasis koik") - no further LV-first split like EXP032/Step6b

Compare against Step 6's flat (0.665) and 4-stage (0.646) RF results.
"""
import os
import sys
from pathlib import Path
import numpy as np
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

RF_KWARGS = dict(n_estimators=700, max_depth=None, min_samples_leaf=2,
                  min_samples_split=2, max_features="sqrt",
                  class_weight="balanced", random_state=42, n_jobs=-1)

# MUU is folded into whichever group it is spectrally closest to at
# training time (there is no separate MUU branch) - it is assigned to
# DECIDUOUS here only for the purpose of giving it a training bucket;
# in practice MUU is a single stray test row and is not the point of
# this experiment.
GROUP_MAP = {
    "KU": "CONIFER", "MA": "CONIFER",
    "HB": "DECIDUOUS", "KS": "DECIDUOUS", "LM": "DECIDUOUS", "LV": "DECIDUOUS",
    "MUU": "DECIDUOUS",
}
CONIFER_SPECIES = {"KU", "MA"}
DECIDUOUS_SPECIES = {"HB", "KS", "LM", "LV", "MUU"}

# =========================================================
# LOAD + PREP (identical to Steps 1-6)
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

# same Top-50 selection as Steps 3-6
imputer0 = SimpleImputer(strategy="median")
X_train_imp0 = imputer0.fit_transform(X_train)
rf0 = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                              class_weight="balanced", random_state=42, n_jobs=-1)
rf0.fit(X_train_imp0, y_train)
importance_df = pd.DataFrame({
    "feature": X_train.columns.tolist(), "importance": rf0.feature_importances_
}).sort_values("importance", ascending=False)
top50 = importance_df.head(50)["feature"].tolist()

X_train_50 = X_train[top50].copy()
X_val_50 = X_val[top50].copy()

# =========================================================
# Stage 1: CONIFER vs DECIDUOUS
# =========================================================
y_group_train = y_train.map(GROUP_MAP)
assert y_group_train.isna().sum() == 0

imputer1 = SimpleImputer(strategy="median")
X_train_imp1 = imputer1.fit_transform(X_train_50)
stage1 = RandomForestClassifier(**RF_KWARGS)
stage1.fit(X_train_imp1, y_group_train)

# =========================================================
# Stage 2a: KU vs MA (true conifer rows)
# =========================================================
conifer_mask_train = y_train.isin(CONIFER_SPECIES)
imputer2a = SimpleImputer(strategy="median")
X_conifer_imp = imputer2a.fit_transform(X_train_50[conifer_mask_train])
stage2a = RandomForestClassifier(**RF_KWARGS)
stage2a.fit(X_conifer_imp, y_train[conifer_mask_train])

# =========================================================
# Stage 2b: HB vs KS vs LM vs LV (+ MUU), all at once (true deciduous rows)
# =========================================================
deciduous_mask_train = y_train.isin(DECIDUOUS_SPECIES)
imputer2b = SimpleImputer(strategy="median")
X_deciduous_imp = imputer2b.fit_transform(X_train_50[deciduous_mask_train])
stage2b = RandomForestClassifier(**RF_KWARGS)
stage2b.fit(X_deciduous_imp, y_train[deciduous_mask_train])


def predict_hierarchical(X):
    X_imp1 = imputer1.transform(X)
    group_pred = stage1.predict(X_imp1)
    final_pred = np.empty(len(X), dtype=object)

    conifer_rows = group_pred == "CONIFER"
    if conifer_rows.any():
        X_c = imputer2a.transform(X.loc[conifer_rows])
        final_pred[conifer_rows] = stage2a.predict(X_c)

    deciduous_rows = group_pred == "DECIDUOUS"
    if deciduous_rows.any():
        X_d = imputer2b.transform(X.loc[deciduous_rows])
        final_pred[deciduous_rows] = stage2b.predict(X_d)

    return final_pred, group_pred


val_final_pred, val_group_pred = predict_hierarchical(X_val_50)
metrics_hier2 = evaluate_model("STEP6c 2-stage hierarchical (conifer/deciduous only) - VALIDATION", y_val, val_final_pred)
evaluate_model("STEP6c Stage-1 group only - VALIDATION", y_val.map(GROUP_MAP), val_group_pred)

print("\n\n" + "=" * 70)
print("STEP 6c SUMMARY - comparison of all architecture variants (RF, Top-50, validation)")
print("=" * 70)
print("(a) Flat single-stage      MacroF1=0.665  (Step 6)")
print("(b) Hierarchical 4-stage   MacroF1=0.646  (Step 6, LV-first split)")
print(f"(c) Hierarchical 2-stage   MacroF1={metrics_hier2['macro_f1']:.3f}  (this run, conifer/deciduous only)")

joblib.dump(
    {"stage1": stage1, "stage2a": stage2a, "stage2b": stage2b,
     "imputer1": imputer1, "imputer2a": imputer2a, "imputer2b": imputer2b,
     "feature_cols_top50": top50, "metrics": metrics_hier2},
    os.path.join(PATHS.models_trained, "STEP6c_2stage_hierarchy.joblib")
)
print("\nSaved: STEP6c_2stage_hierarchy.joblib")
