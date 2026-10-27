# -*- coding: utf-8 -*-
"""Extension step 1 (RQ4/RQ5): repeat the winning pure-stand pipeline
(Steps 1-8: flat XGBoost, Top-50 auto-selected features - NO species-
signature meta-features, NO oversampling, sample_weight=balanced) on the
EXTENDED dataset that includes BOTH pure (>=80% purity) AND mixed stands
(EXP047/048's composition_base_2026_subregion2.gpkg export, 10,309 parcels).

Target is still "target" = the DOMINANT species per parcel (same 7 classes).
This answers: does the pipeline that won on pure stands still work when
mixed stands are included in training/evaluation?

Top-50 is RE-SELECTED fresh on this dataset (different rows, different aux
column names: height_*/soil_majority instead of elev_*/soil_main) - the
selection PROCEDURE is identical to Steps 3-7 (RF importance, class_weight=
balanced, TRAIN only), but the actual 50 columns may differ.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.preprocessing.timeseries import (
    aggregate_daily_to_windows, long_to_wide, merge_aux,
    encode_categoricals_consistently, align_columns, drop_all_nan_train_columns,
)
from src.preprocessing.clearcut import remove_clearcut_ids
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix_pretty
from src.paths import PATHS

DAILY_DIR = PATHS.intermediate / "EXP047_composition_daily"
TRAIN_FILE = DAILY_DIR / "composition_train_sentinel2_daily_timeseries.csv"
VAL_FILE = DAILY_DIR / "composition_val_sentinel2_daily_timeseries.csv"
TEST_FILE = DAILY_DIR / "composition_test_sentinel2_daily_timeseries.csv"
PARCEL_FILE = PATHS.raw / "composition_base_2026_subregion2.gpkg"
OVERLAP_FILE = os.path.join(PATHS.reference, "clearcut_overlap", "stands_overlapping_clearcuts_over20pct.csv")

MODEL_OUT = os.path.join(PATHS.models_trained, "EXT1_pure_mixed_flat_xgb.joblib")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "EXT1_pure_mixed_test_predictions.csv")
CM_OUT = os.path.join(PATHS.outputs_figures, "EXT1_pure_mixed_confusion_matrix_test.png")

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

WINDOW_DATES = [
    "2025-01-31", "2025-02-10", "2025-02-20", "2025-03-02", "2025-03-12",
    "2025-04-01", "2025-04-11", "2025-04-21", "2025-05-01", "2025-05-11",
    "2025-05-21", "2025-05-31", "2025-06-10", "2025-06-20", "2025-06-30",
    "2025-07-10", "2025-07-20", "2025-07-30", "2025-08-09", "2025-08-19",
    "2025-08-29", "2025-09-08", "2025-09-18", "2025-09-28", "2025-10-08",
    "2025-10-18",
]
WINDOW_DAYS = 10

TS_FEATURE_COLS = [
    "B2_mean", "B3_mean", "B4_mean", "B5_mean", "B6_mean", "B7_mean",
    "B8_mean", "B8A_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDRE_mean", "NDMI_mean", "NBR_mean"
]
AUX_NUMERIC_COLS = ["height_mean", "height_median", "height_std", "height_min", "height_max"]
AUX_CATEGORICAL_COLS = ["soil_majority"]

BEST_XGB_KWARGS = dict(n_estimators=500, max_depth=5, learning_rate=0.08,
                        min_child_weight=5, gamma=0.1, random_state=42, n_jobs=-1,
                        eval_metric="mlogloss")


def load_aux_from_parcel_file():
    gdf = gpd.read_file(PARCEL_FILE, ignore_geometry=True)
    aux = gdf[[ID_COL, "elev_mean", "elev_median", "elev_std", "elev_min", "elev_max", "soil_main"]].copy()
    aux = aux.rename(columns={
        "elev_mean": "height_mean", "elev_median": "height_median", "elev_std": "height_std",
        "elev_min": "height_min", "elev_max": "height_max", "soil_main": "soil_majority",
    })
    return aux


train_long = pd.read_csv(TRAIN_FILE)
val_long = pd.read_csv(VAL_FILE)
test_long = pd.read_csv(TEST_FILE)
aux = load_aux_from_parcel_file()

train_long, val_long, test_long = remove_clearcut_ids(train_long, val_long, test_long, OVERLAP_FILE, ID_COL, TARGET_COL)

train_windowed = aggregate_daily_to_windows(train_long, ID_COL, DATE_COL, TARGET_COL, TS_FEATURE_COLS, WINDOW_DATES, WINDOW_DAYS)
val_windowed = aggregate_daily_to_windows(val_long, ID_COL, DATE_COL, TARGET_COL, TS_FEATURE_COLS, WINDOW_DATES, WINDOW_DAYS)
test_windowed = aggregate_daily_to_windows(test_long, ID_COL, DATE_COL, TARGET_COL, TS_FEATURE_COLS, WINDOW_DATES, WINDOW_DAYS)

train_wide = long_to_wide(train_windowed, "train", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
val_wide = long_to_wide(val_windowed, "val", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
test_wide = long_to_wide(test_windowed, "test", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)

train_wide = merge_aux(train_wide, aux, "train", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
val_wide = merge_aux(val_wide, aux, "val", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
test_wide = merge_aux(test_wide, aux, "test", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)

train_wide, val_wide, test_wide = encode_categoricals_consistently(train_wide, val_wide, test_wide, AUX_CATEGORICAL_COLS)
train_wide, val_wide, test_wide, feature_cols = align_columns(train_wide, val_wide, test_wide, ID_COL, TARGET_COL)

X_train = train_wide[feature_cols].copy()
y_train = train_wide[TARGET_COL].copy()
X_val = val_wide[feature_cols].copy()
y_val = val_wide[TARGET_COL].copy()
X_test = test_wide[feature_cols].copy()
y_test = test_wide[TARGET_COL].copy()

X_train, X_val, X_test, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_test)
print("Full feature count:", X_train.shape[1])
print("Train/Val/Test parcels:", len(X_train), len(X_val), len(X_test))
print("\nTrain species distribution:\n", y_train.value_counts())

# =========================================================
# Top-50 selection (fresh, same procedure as Steps 3-7)
# =========================================================
imputer0 = SimpleImputer(strategy="median")
X_train_imp0 = imputer0.fit_transform(X_train)
rf0 = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                              class_weight="balanced", random_state=42, n_jobs=-1)
rf0.fit(X_train_imp0, y_train)
importance_df = pd.DataFrame({
    "feature": X_train.columns.tolist(), "importance": rf0.feature_importances_
}).sort_values("importance", ascending=False)
top50 = importance_df.head(50)["feature"].tolist()
print("\nHeight features in new Top-50:", [f for f in top50 if f.startswith("height")])
print("Soil features in new Top-50:", [f for f in top50 if f.startswith("soil")])

X_train_50 = X_train[top50].copy()
X_val_50 = X_val[top50].copy()
X_test_50 = X_test[top50].copy()

# =========================================================
# Sanity check: train-only fit -> VALIDATION
# =========================================================
imputer_v = SimpleImputer(strategy="median")
X_train_imp_v = imputer_v.fit_transform(X_train_50)
X_val_imp_v = imputer_v.transform(X_val_50)
le_v = LabelEncoder()
y_train_enc_v = le_v.fit_transform(y_train)
sw_v = compute_sample_weight("balanced", y_train_enc_v)
model_v = XGBClassifier(**BEST_XGB_KWARGS)
model_v.fit(X_train_imp_v, y_train_enc_v, sample_weight=sw_v)
pred_v = le_v.inverse_transform(model_v.predict(X_val_imp_v))
val_metrics = evaluate_model("EXT1 pure+mixed flat XGBoost - VALIDATION", y_val, pred_v)

# =========================================================
# FINAL: TRAIN+VAL -> TEST (single touch)
# =========================================================
X_trainval_50 = pd.concat([X_train_50, X_val_50], axis=0).reset_index(drop=True)
y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

imputer_f = SimpleImputer(strategy="median")
X_trainval_imp = imputer_f.fit_transform(X_trainval_50)
X_test_imp = imputer_f.transform(X_test_50)

le = LabelEncoder()
y_trainval_enc = le.fit_transform(y_trainval)
sw = compute_sample_weight("balanced", y_trainval_enc)

model_final = XGBClassifier(**BEST_XGB_KWARGS)
model_final.fit(X_trainval_imp, y_trainval_enc, sample_weight=sw)
test_pred = le.inverse_transform(model_final.predict(X_test_imp))

test_metrics = evaluate_model("EXT1 pure+mixed flat XGBoost - TEST", y_test, test_pred)
test_metrics["hb_precision"] = precision_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_recall"] = recall_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_f1"] = f1_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)

plot_confusion_matrix_pretty(y_test, test_pred, "EXT1 pure+mixed flat XGBoost - test confusion matrix", CM_OUT, normalize=False)

test_predictions_df = pd.DataFrame({ID_COL: test_wide[ID_COL].values, "y_true": y_test.values, "y_pred": test_pred})
test_predictions_df.to_csv(TEST_PRED_OUT, index=False, encoding="utf-8-sig")

joblib.dump(
    {"model": model_final, "imputer": imputer_f, "label_encoder": le, "feature_cols_top50": top50,
     "val_metrics": val_metrics, "test_metrics": test_metrics,
     "removed_clearcut_overlap_file": OVERLAP_FILE, "experiment": "EXT1_pure_mixed_flat_xgb"},
    MODEL_OUT
)

print("\n\n" + "=" * 70)
print("SUMMARY: pure-only (Table, samm 8) vs pure+mixed (this run)")
print("=" * 70)
print(f"Pure-only  VALIDATION MacroF1=0.7068 | TEST MacroF1=0.7342 (n_test=1510)")
print(f"Pure+mixed VALIDATION MacroF1={val_metrics['macro_f1']:.4f} | TEST MacroF1={test_metrics['macro_f1']:.4f} (n_test={len(y_test)})")
print("\nSaved:", MODEL_OUT)
