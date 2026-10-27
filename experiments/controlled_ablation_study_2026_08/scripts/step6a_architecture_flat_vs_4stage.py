# -*- coding: utf-8 -*-
"""Step 6 of the controlled ablation: ARCHITECTURE comparison (single-stage
flat RF vs. 4-stage hierarchical RF cascade), holding everything else fixed:
same clearcut-filtered canonical data, same Top-50 auto-selected feature set
(computed once, reused identically in every stage - not re-selected per
stage), same RF hyperparameters (Step 5 winner: n_estimators=700,
min_samples_leaf=2, max_features=sqrt, class_weight=balanced), no manual
oversampling (Step 4 established class_weight=balanced alone is best).

Hierarchy (structure borrowed from EXP032, but re-fit here on the current
Top-50 feature set / no-oversampling setup instead of EXP032's own
species-signature+oversampling combo, to isolate architecture as the only
variable):
  Stage 1    : {CONIFER, DECIDUOUS, MUU}
  Stage 2a   : KU vs MA                     (true-conifer rows only)
  Stage 2b-i : LV vs REST(HB+KS+LM)          (true-deciduous rows only)
  Stage 2b-ii: HB vs KS vs LM                (true rest-of-deciduous rows only)
Evaluation routes on PREDICTED stage-1/2b-i output, never ground truth.
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

GROUP_MAP = {
    "KU": "CONIFER", "MA": "CONIFER",
    "HB": "DECIDUOUS", "KS": "DECIDUOUS", "LM": "DECIDUOUS", "LV": "DECIDUOUS",
    "MUU": "MUU"
}
CONIFER_SPECIES = {"KU", "MA"}
DECIDUOUS_SPECIES = {"HB", "KS", "LM", "LV"}
REST_OF_DECIDUOUS_SPECIES = {"HB", "KS", "LM"}
GROUP_COL = "target_group"
LV_VS_REST_COL = "target_lv_vs_rest"

# =========================================================
# LOAD + PREP (identical to Steps 1-5)
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

# same Top-50 selection procedure as Steps 3-5 (importances from a
# class_weight=balanced full-set RF), fixed and reused identically for
# both the flat model and every stage of the hierarchy below
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
# (a) FLAT (single-stage) - reuses Step 5 winning config
# =========================================================
imputer_flat = SimpleImputer(strategy="median")
X_train_flat_imp = imputer_flat.fit_transform(X_train_50)
X_val_flat_imp = imputer_flat.transform(X_val_50)

rf_flat = RandomForestClassifier(**RF_KWARGS)
rf_flat.fit(X_train_flat_imp, y_train)
pred_flat = rf_flat.predict(X_val_flat_imp)
metrics_flat = evaluate_model("STEP6a Flat single-stage - VALIDATION", y_val, pred_flat)

# =========================================================
# (b) HIERARCHICAL (4-stage cascade), same Top-50 features/hyperparams,
#     class_weight=balanced at every stage, NO oversampling
# =========================================================
y_group_train = y_train.map(GROUP_MAP)

imputer1 = SimpleImputer(strategy="median")
X_train_50_imp1 = imputer1.fit_transform(X_train_50)
stage1 = RandomForestClassifier(**RF_KWARGS)
stage1.fit(X_train_50_imp1, y_group_train)

conifer_mask_train = y_train.isin(CONIFER_SPECIES)
imputer2a = SimpleImputer(strategy="median")
X_conifer_imp = imputer2a.fit_transform(X_train_50[conifer_mask_train])
stage2a = RandomForestClassifier(**RF_KWARGS)
stage2a.fit(X_conifer_imp, y_train[conifer_mask_train])

deciduous_mask_train = y_train.isin(DECIDUOUS_SPECIES)
y_lv_vs_rest_train = y_train[deciduous_mask_train].apply(lambda s: "LV" if s == "LV" else "REST")
imputer2bi = SimpleImputer(strategy="median")
X_deciduous_imp = imputer2bi.fit_transform(X_train_50[deciduous_mask_train])
stage2bi = RandomForestClassifier(**RF_KWARGS)
stage2bi.fit(X_deciduous_imp, y_lv_vs_rest_train)

rest_mask_train = y_train.isin(REST_OF_DECIDUOUS_SPECIES)
imputer2bii = SimpleImputer(strategy="median")
X_rest_imp = imputer2bii.fit_transform(X_train_50[rest_mask_train])
stage2bii = RandomForestClassifier(**RF_KWARGS)
stage2bii.fit(X_rest_imp, y_train[rest_mask_train])


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
        X_d = imputer2bi.transform(X.loc[deciduous_rows])
        lv_pred = stage2bi.predict(X_d)
        is_lv = lv_pred == "LV"
        deciduous_idx = np.where(deciduous_rows)[0]
        lv_idx = deciduous_idx[is_lv]
        rest_idx = deciduous_idx[~is_lv]
        final_pred[lv_idx] = "LV"
        if len(rest_idx) > 0:
            X_r = imputer2bii.transform(X.iloc[rest_idx])
            final_pred[rest_idx] = stage2bii.predict(X_r)

    muu_rows = group_pred == "MUU"
    final_pred[muu_rows] = "MUU"
    return final_pred, group_pred


val_final_pred, val_group_pred = predict_hierarchical(X_val_50)
metrics_hier = evaluate_model("STEP6b Hierarchical (4-stage) - VALIDATION", y_val, val_final_pred)
evaluate_model("STEP6b Stage-1 group only - VALIDATION", y_val.map(GROUP_MAP), val_group_pred)

# =========================================================
# SUMMARY
# =========================================================
print("\n\n" + "=" * 70)
print("STEP 6 SUMMARY (architecture, Top-50 features, validation)")
print("=" * 70)
print(f"(a) Flat single-stage    Acc={metrics_flat['accuracy']:.3f}  BalAcc={metrics_flat['balanced_accuracy']:.3f}  MacroF1={metrics_flat['macro_f1']:.3f}  Kappa={metrics_flat['kappa']:.3f}")
print(f"(b) Hierarchical 4-stage Acc={metrics_hier['accuracy']:.3f}  BalAcc={metrics_hier['balanced_accuracy']:.3f}  MacroF1={metrics_hier['macro_f1']:.3f}  Kappa={metrics_hier['kappa']:.3f}")

joblib.dump(
    {"rf_flat": rf_flat, "stage1": stage1, "stage2a": stage2a, "stage2bi": stage2bi, "stage2bii": stage2bii,
     "imputer1": imputer1, "imputer2a": imputer2a, "imputer2bi": imputer2bi, "imputer2bii": imputer2bii,
     "feature_cols_top50": top50, "metrics_flat": metrics_flat, "metrics_hier": metrics_hier},
    os.path.join(PATHS.models_trained, "STEP6_architecture.joblib")
)
print("\nSaved: STEP6_architecture.joblib")
