# -*- coding: utf-8 -*-
"""Extension step 2 (RQ2): species composition regression - predicting each
species' % share of the stand (not just the dominant species), on the same
pure+mixed dataset as Extension step 1, using the SAME winning pipeline
choices established by the classification ablation (Steps 1-8):
  - Same Top-50 feature set selected in Extension step 1 (EXT1's Top-50,
    reused as-is - same input features, only the algorithm/task changes)
  - XGBoost (won classification at every step) via MultiOutputRegressor
    (XGBoost has no native multi-output regressor), Step 7's winning
    hyperparameters carried over unchanged for consistency
  - No oversampling (not meaningful for a continuous regression target)
  - TRAIN-only fit -> VALIDATION (informational), then TRAIN+VAL -> TEST
    (single touch)

Compare against EXP076 (old, confounded: species-signature features +
different hyperparams) TEST results, if available.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

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

MODEL_OUT = os.path.join(PATHS.models_trained, "EXT2_composition_regression_xgb.joblib")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "EXT2_composition_test_predictions.csv")
REGRESSION_METRICS_OUT = os.path.join(PATHS.outputs_tables, "EXT2_regression_metrics_per_species.csv")
CM_DERIVED_OUT = os.path.join(PATHS.outputs_figures, "EXT2_derived_dominant_species_confusion_matrix.png")

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

SPECIES_7 = ["HB", "KS", "KU", "LM", "LV", "MA", "MUU"]
SHARE_COLS = [f"share_{s}" for s in SPECIES_7]

BEST_XGB_KWARGS = dict(n_estimators=500, max_depth=5, learning_rate=0.08,
                        min_child_weight=5, gamma=0.1, random_state=42, n_jobs=-1)


def load_aux_from_parcel_file():
    gdf = gpd.read_file(PARCEL_FILE, ignore_geometry=True)
    aux = gdf[[ID_COL, "elev_mean", "elev_median", "elev_std", "elev_min", "elev_max", "soil_main"]].copy()
    aux = aux.rename(columns={
        "elev_mean": "height_mean", "elev_median": "height_median", "elev_std": "height_std",
        "elev_min": "height_min", "elev_max": "height_max", "soil_main": "soil_majority",
    })
    return aux


def build_composition_targets():
    gdf = gpd.read_file(PARCEL_FILE, ignore_geometry=True)
    shares = pd.DataFrame(0.0, index=gdf.index, columns=SHARE_COLS)
    for i in range(1, 6):
        liik_col, osa_col = f"liik_{i}", f"osakaal_{i}"
        if liik_col not in gdf.columns:
            continue
        for s in SPECIES_7:
            mask = gdf[liik_col] == s
            shares.loc[mask, f"share_{s}"] += gdf.loc[mask, osa_col].fillna(0)
    row_sums = shares.sum(axis=1).replace(0, np.nan)
    shares = shares.div(row_sums, axis=0) * 100.0
    shares[ID_COL] = gdf[ID_COL].values
    n_no_composition = shares[SHARE_COLS].isna().all(axis=1).sum()
    print(f"Parcels with no usable composition breakdown after pivot: {n_no_composition} / {len(shares)}")
    return shares


def renormalize_predictions(pred):
    pred = np.clip(pred, 0, None)
    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return pred / row_sums * 100.0


def report_regression_metrics(name, y_true, y_pred):
    rows = []
    for i, s in enumerate(SPECIES_7):
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = mean_squared_error(y_true[:, i], y_pred[:, i]) ** 0.5
        rows.append({"species": s, "r2": r2, "mae": mae, "rmse": rmse})
    df = pd.DataFrame(rows)
    overall_mae = mean_absolute_error(y_true, y_pred)
    print(f"\n================ EXT2 composition regression - {name} ================")
    print(df.to_string(index=False))
    print(f"\nOverall (all species, macro) MAE: {overall_mae:.3f} percentage points")
    return df, overall_mae


# =========================================================
# LOAD + PREP (identical to Extension step 1 / EXP076)
# =========================================================
train_long = pd.read_csv(TRAIN_FILE)
val_long = pd.read_csv(VAL_FILE)
test_long = pd.read_csv(TEST_FILE)
aux = load_aux_from_parcel_file()
composition_targets = build_composition_targets()

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

train_wide = train_wide.merge(composition_targets, on=ID_COL, how="left")
val_wide = val_wide.merge(composition_targets, on=ID_COL, how="left")
test_wide = test_wide.merge(composition_targets, on=ID_COL, how="left")

train_wide = train_wide.dropna(subset=SHARE_COLS, how="all")
val_wide = val_wide.dropna(subset=SHARE_COLS, how="all")
test_wide = test_wide.dropna(subset=SHARE_COLS, how="all")

# reuse Extension step 1's Top-50 feature set (same input features feed
# both the dominant-species classifier and this regressor)
ext1 = joblib.load(os.path.join(PATHS.models_trained, "EXT1_pure_mixed_flat_xgb.joblib"))
top50 = ext1["feature_cols_top50"]
top50 = [c for c in top50 if c in train_wide.columns]
print(f"\nReusing Extension step 1's Top-50 ({len(top50)} available of 50)")

X_train = train_wide[top50].copy()
Y_train = train_wide[SHARE_COLS].fillna(0).copy()
dom_train = train_wide[TARGET_COL].copy()

X_val = val_wide[top50].copy()
Y_val = val_wide[SHARE_COLS].fillna(0).copy()
dom_val = val_wide[TARGET_COL].copy()

X_test = test_wide[top50].copy()
Y_test = test_wide[SHARE_COLS].fillna(0).copy()
dom_test = test_wide[TARGET_COL].copy()

X_train, X_val, X_test, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_test)
print("Train rows with usable composition target:", len(X_train))
print("Mean species shares (train):\n", Y_train.mean().round(1))

# =========================================================
# TRAIN-only -> VALIDATION (informational)
# =========================================================
print("\n" + "=" * 60)
print("TRAIN-only XGBoost regressor -> VALIDATION")
print("=" * 60)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
model_trainonly = MultiOutputRegressor(XGBRegressor(**BEST_XGB_KWARGS), n_jobs=1)
model_trainonly.fit(X_train_imp, Y_train)

X_val_imp = imputer.transform(X_val)
val_pred = renormalize_predictions(model_trainonly.predict(X_val_imp))
val_metrics_df, val_overall_mae = report_regression_metrics("VALIDATION", Y_val.values, val_pred)

# =========================================================
# FINAL: TRAIN+VAL -> TEST (single touch)
# =========================================================
print("\n" + "=" * 60)
print("TRAIN+VAL XGBoost regressor -> TEST")
print("=" * 60)

X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
Y_trainval = pd.concat([Y_train, Y_val], axis=0).reset_index(drop=True)

valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()
X_trainval = X_trainval[valid_cols_trainval].copy()
X_test_final = X_test[valid_cols_trainval].copy()

imputer_final = SimpleImputer(strategy="median")
X_trainval_imp = imputer_final.fit_transform(X_trainval)
model_final = MultiOutputRegressor(XGBRegressor(**BEST_XGB_KWARGS), n_jobs=1)
model_final.fit(X_trainval_imp, Y_trainval)

X_test_imp = imputer_final.transform(X_test_final)
test_pred = renormalize_predictions(model_final.predict(X_test_imp))

test_metrics_df, test_overall_mae = report_regression_metrics("TEST", Y_test.values, test_pred)
test_metrics_df.to_csv(REGRESSION_METRICS_OUT, index=False, encoding="utf-8-sig")

# derived dominant-species classification check (a secondary diagnostic,
# not the primary regression metric)
pred_dominant = np.array(SPECIES_7)[test_pred.argmax(axis=1)]
evaluate_model("EXT2 derived dominant species (argmax of predicted composition) - TEST", dom_test, pred_dominant)
plot_confusion_matrix_pretty(
    dom_test, pred_dominant, "EXT2 derived dominant-species confusion matrix (test)",
    CM_DERIVED_OUT, normalize=True
)

test_predictions_df = pd.DataFrame({ID_COL: test_wide[ID_COL].values, "dominant_true": dom_test.values, "dominant_pred_from_composition": pred_dominant})
for i, s in enumerate(SPECIES_7):
    test_predictions_df[f"true_{s}"] = Y_test[f"share_{s}"].values
    test_predictions_df[f"pred_{s}"] = test_pred[:, i]
test_predictions_df.to_csv(TEST_PRED_OUT, index=False, encoding="utf-8-sig")

joblib.dump({
    "model": model_final, "imputer": imputer_final, "feature_cols": valid_cols_trainval,
    "species_order": SPECIES_7, "id_col": ID_COL, "target_col": TARGET_COL,
    "val_overall_mae": val_overall_mae, "test_overall_mae": test_overall_mae,
    "regression_metrics_val": val_metrics_df.to_dict(orient="records"),
    "regression_metrics_test": test_metrics_df.to_dict(orient="records"),
    "removed_clearcut_overlap_file": OVERLAP_FILE, "experiment": "EXT2_composition_regression_xgb",
}, MODEL_OUT)

print("\n\nSaved:", MODEL_OUT)
print("Test predictions:", TEST_PRED_OUT)
print("Regression metrics:", REGRESSION_METRICS_OUT)
