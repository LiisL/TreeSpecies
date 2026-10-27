# -*- coding: utf-8 -*-
"""Step 7 of the controlled ablation: XGBoost hyperparameter search, on the
current flat single-stage pipeline (Step 6's winning architecture), same
Top-50 auto-selected features, class balancing via sample_weight="balanced"
(the XGBoost equivalent of RF's class_weight=balanced - Step 4 established
balanced weighting beats no-balancing/oversampling), no manual oversampling.

NOTE: does NOT reuse EXP041_search_results.csv - same reasoning as Step 5:
that grid was run inside EXP032's hierarchical architecture with species-
signature features and manual oversampling (a different context). Reruns
the SAME 10-config grid (taken verbatim from EXP041) directly on the
current flat Top-50 pipeline.
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

# Same 10-config grid as EXP041, verbatim (param_i=1 is EXP038's original default)
PARAM_GRID = [
    dict(param_i=1, n_estimators=400, max_depth=6, learning_rate=0.1),
    dict(param_i=2, n_estimators=800, max_depth=4, learning_rate=0.05),
    dict(param_i=3, n_estimators=200, max_depth=9, learning_rate=0.1),
    dict(param_i=4, n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
    dict(param_i=5, n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, min_child_weight=3),
    dict(param_i=6, n_estimators=1000, max_depth=4, learning_rate=0.03),
    dict(param_i=7, n_estimators=300, max_depth=6, learning_rate=0.15, subsample=0.8, colsample_bytree=1.0),
    dict(param_i=8, n_estimators=500, max_depth=5, learning_rate=0.08, min_child_weight=5, gamma=0.1),
    dict(param_i=9, n_estimators=400, max_depth=10, learning_rate=0.05, subsample=0.7, colsample_bytree=0.7),
    dict(param_i=10, n_estimators=700, max_depth=6, learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0),
]

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

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_50)
X_val_imp = imputer.transform(X_val_50)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)
sample_weight = compute_sample_weight("balanced", y_train_enc)

rows = []
best = None
for cfg in PARAM_GRID:
    kwargs = {k: v for k, v in cfg.items() if k != "param_i"}
    model = XGBClassifier(**kwargs, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    model.fit(X_train_imp, y_train_enc, sample_weight=sample_weight)
    pred_enc = model.predict(X_val_imp)
    pred = le.inverse_transform(pred_enc)
    m = evaluate_model(f"STEP7 param_i={cfg['param_i']} - VALIDATION", y_val, pred)
    row = {**cfg, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
           "val_balanced_accuracy": m["balanced_accuracy"], "val_kappa": m["kappa"]}
    rows.append(row)
    if best is None or m["macro_f1"] > best[1]["macro_f1"]:
        best = (cfg, m, model)

results_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
print("\n\n" + "=" * 70)
print("STEP 7 SUMMARY (XGBoost hyperparameter search, Top-50 features, validation)")
print("=" * 70)
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(PATHS.outputs_tables, "STEP7_xgb_search_results.csv"), index=False, encoding="utf-8-sig")

joblib.dump(
    {"best_config": best[0], "best_metrics": best[1], "best_model": best[2], "label_encoder": le,
     "feature_cols_top50": top50, "all_results": rows},
    os.path.join(PATHS.models_trained, "STEP7_xgb_best.joblib")
)
print("\nBest config:", best[0])
print("Best Macro F1:", best[1]["macro_f1"])
print("Saved: STEP7_xgb_best.joblib, STEP7_xgb_search_results.csv")
