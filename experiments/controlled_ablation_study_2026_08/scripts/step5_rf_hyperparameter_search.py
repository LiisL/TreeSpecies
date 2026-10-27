# -*- coding: utf-8 -*-
"""Step 5 of the controlled ablation: RF hyperparameter search, holding the
feature representation fixed at Step 3/4's winner (auto Top-50 + class_weight
balanced family), clearcut-filtered canonical Subregion2 data.

NOTE: does NOT reuse EXP042_search_results.csv - that grid search was run
inside EXP032's hierarchical architecture on species-signature features
(a different feature/architecture context), so its winning hyperparameters
are not guaranteed to transfer to the current flat Top-50 pipeline. Reusing
it here would silently reintroduce a confound. Instead this reruns the SAME
9-config grid (taken verbatim from EXP019/EXP042) directly on the current
flat pipeline, so the search space is unchanged but the context is correct.
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

# Same 9-config grid as EXP019/EXP042, verbatim (param_i kept for traceability)
GRID = [
    dict(param_i=1, n_estimators=400, max_depth=None, min_samples_leaf=2, min_samples_split=2, max_features="sqrt", class_weight="balanced"),
    dict(param_i=2, n_estimators=400, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features="sqrt", class_weight="balanced"),
    dict(param_i=3, n_estimators=700, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features="sqrt", class_weight="balanced_subsample"),
    dict(param_i=4, n_estimators=700, max_depth=None, min_samples_leaf=2, min_samples_split=2, max_features="sqrt", class_weight="balanced"),
    dict(param_i=5, n_estimators=1000, max_depth=None, min_samples_leaf=2, min_samples_split=5, max_features="sqrt", class_weight="balanced_subsample"),
    dict(param_i=6, n_estimators=700, max_depth=40, min_samples_leaf=1, min_samples_split=2, max_features="log2", class_weight="balanced"),
    dict(param_i=7, n_estimators=1000, max_depth=60, min_samples_leaf=1, min_samples_split=5, max_features=0.3, class_weight="balanced_subsample"),
    dict(param_i=8, n_estimators=700, max_depth=30, min_samples_leaf=2, min_samples_split=5, max_features=0.5, class_weight="balanced"),
    dict(param_i=9, n_estimators=1000, max_depth=None, min_samples_leaf=4, min_samples_split=10, max_features="sqrt", class_weight="balanced_subsample"),
]

# =========================================================
# LOAD + PREP (identical to Steps 1-4)
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
    "feature": X_train.columns.tolist(),
    "importance": rf0.feature_importances_
}).sort_values("importance", ascending=False)
top50 = importance_df.head(50)["feature"].tolist()

X_train_50 = X_train[top50].copy()
X_val_50 = X_val[top50].copy()

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train_50)
X_val_imp = imputer.transform(X_val_50)

rows = []
best = None
for cfg in GRID:
    kwargs = {k: v for k, v in cfg.items() if k != "param_i"}
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **kwargs)
    rf.fit(X_train_imp, y_train)
    pred = rf.predict(X_val_imp)
    m = evaluate_model(f"STEP5 param_i={cfg['param_i']} - VALIDATION", y_val, pred)
    row = {**cfg, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"],
           "val_balanced_accuracy": m["balanced_accuracy"], "val_kappa": m["kappa"]}
    rows.append(row)
    if best is None or m["macro_f1"] > best[1]["macro_f1"]:
        best = (cfg, m, rf)

results_df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
print("\n\n" + "=" * 70)
print("STEP 5 SUMMARY (RF hyperparameter search, Top-50 features, validation)")
print("=" * 70)
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(PATHS.outputs_tables, "STEP5_rf_search_results.csv"), index=False, encoding="utf-8-sig")

joblib.dump(
    {"best_config": best[0], "best_metrics": best[1], "best_model": best[2],
     "feature_cols_top50": top50, "all_results": rows},
    os.path.join(PATHS.models_trained, "STEP5_rf_best.joblib")
)
print("\nBest config:", best[0])
print("Best Macro F1:", best[1]["macro_f1"])
print("Saved: STEP5_rf_best.joblib, STEP5_rf_search_results.csv")
