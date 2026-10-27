# -*- coding: utf-8 -*-
"""FINAL single test-set evaluation for the pure-stand classification model
that won the controlled ablation (Steps 1-7): single-stage XGBoost, Top-50
auto-selected Sentinel-2 features (no height/soil), sample_weight="balanced",
Step 7's winning hyperparameters. Refits on TRAIN+VAL (standard convention
used throughout this project - EXP000/021/032/038 all do this), evaluates
ONCE on TEST. This is the ONLY test-set touch for this model - it maps to
RQ1 (does a workable pure-stand species model exist) and RQ3 (RF vs XGBoost)
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
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.preprocessing.timeseries import (
    long_to_wide, merge_aux, encode_categoricals_consistently,
    align_columns, drop_all_nan_train_columns,
)
from src.preprocessing.clearcut import remove_clearcut_ids
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix_pretty
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

MODEL_OUT = os.path.join(PATHS.models_trained, "FINAL_pure_xgb_flat_top50.joblib")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "FINAL_pure_xgb_flat_top50_test_predictions.csv")
CM_OUT = os.path.join(PATHS.outputs_figures, "FINAL_pure_xgb_flat_top50_confusion_matrix_test.png")

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

# Top-50 selection: as in Steps 3-7, from a class_weight=balanced full-set
# RF trained on TRAIN only. IMPORTANT: this selection step happened BEFORE
# ever looking at TEST, and is now treated as a FIXED, already-decided
# choice (that's what Steps 1-7 were for) - it is not re-derived on
# TRAIN+VAL here, to avoid quietly changing the feature set at the final
# hurdle.
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
X_test_50 = X_test[top50].copy()

# =========================================================
# Sanity check: confirm this matches Step 7's VALIDATION number first
# (train-only fit -> val), before touching test
# =========================================================
imputer_val = SimpleImputer(strategy="median")
X_train_imp_val = imputer_val.fit_transform(X_train_50)
X_val_imp_val = imputer_val.transform(X_val_50)

le_val = LabelEncoder()
y_train_enc_val = le_val.fit_transform(y_train)
sw_val = compute_sample_weight("balanced", y_train_enc_val)

model_val_check = XGBClassifier(**BEST_XGB_KWARGS)
model_val_check.fit(X_train_imp_val, y_train_enc_val, sample_weight=sw_val)
pred_val_check = le_val.inverse_transform(model_val_check.predict(X_val_imp_val))
print("\n--- Sanity check: reproducing Step 7 VALIDATION result ---")
val_check_metrics = evaluate_model("Sanity check - VALIDATION (train-only fit)", y_val, pred_val_check)

# =========================================================
# FINAL MODEL: refit on TRAIN+VAL, evaluate ONCE on TEST
# =========================================================
X_trainval_50 = pd.concat([X_train_50, X_val_50], axis=0).reset_index(drop=True)
y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

imputer_final = SimpleImputer(strategy="median")
X_trainval_imp = imputer_final.fit_transform(X_trainval_50)
X_test_imp = imputer_final.transform(X_test_50)

le = LabelEncoder()
y_trainval_enc = le.fit_transform(y_trainval)
sample_weight = compute_sample_weight("balanced", y_trainval_enc)

print("\n--- FINAL model: TRAIN+VAL -> TEST (the one and only test-set touch) ---")
model_final = XGBClassifier(**BEST_XGB_KWARGS)
model_final.fit(X_trainval_imp, y_trainval_enc, sample_weight=sample_weight)
test_pred = le.inverse_transform(model_final.predict(X_test_imp))

test_metrics = evaluate_model("FINAL pure-stand XGBoost (flat, Top-50) - TEST", y_test, test_pred)
test_metrics["hb_precision"] = precision_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_recall"] = recall_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_f1"] = f1_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
print(f"\nHB precision={test_metrics['hb_precision']:.4f}, HB recall={test_metrics['hb_recall']:.4f}, HB F1={test_metrics['hb_f1']:.4f}")

plot_confusion_matrix_pretty(
    y_test, test_pred, "FINAL pure-stand XGBoost (flat, Top-50) - test confusion matrix",
    CM_OUT, normalize=False
)

test_predictions_df = pd.DataFrame({
    ID_COL: test_wide[ID_COL].values, "y_true": y_test.values, "y_pred": test_pred
})
test_predictions_df.to_csv(TEST_PRED_OUT, index=False, encoding="utf-8-sig")

joblib.dump(
    {"model": model_final, "imputer": imputer_final, "label_encoder": le,
     "feature_cols_top50": top50, "val_check_metrics": val_check_metrics, "test_metrics": test_metrics,
     "xgb_kwargs": BEST_XGB_KWARGS, "removed_clearcut_overlap_file": OVERLAP_FILE,
     "experiment": "FINAL_pure_xgb_flat_top50"},
    MODEL_OUT
)

print("\n\n" + "=" * 70)
print("FINAL COMPARISON: old EXP038 (hierarchical, confounded) vs new (clean ablation)")
print("=" * 70)
print("EXP038 (old primary)   TEST: Accuracy=0.868  MacroF1=0.684  Kappa=0.818  HB_F1=0.614")
print(f"NEW winner (flat XGB)  TEST: Accuracy={test_metrics['accuracy']:.3f}  MacroF1={test_metrics['macro_f1']:.3f}  Kappa={test_metrics['kappa']:.3f}  HB_F1={test_metrics['hb_f1']:.3f}")
print("\nSaved:", MODEL_OUT)
print("Test predictions:", TEST_PRED_OUT)
print("Confusion matrix:", CM_OUT)
