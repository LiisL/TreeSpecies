"""EXP050: multi-output regression predicting full stand SPECIES
COMPOSITION (each species' % share of the stand), not just the dominant
species - the Klehr et al. 2025-style approach already cited in the
thesis, confirmed via explicit user decision as the intended meaning of
"classify composition" (see
[[project-treespecies-exp047-composition-classification]]).

Uses the SAME data pipeline as EXP048/049 (EXP047's daily Statistical-API
download over composition_base_2026_subregion2.gpkg's 10,309 pure+mixed
parcels, aggregated to the standard 26 10-day windows via
src.preprocessing.timeseries.aggregate_daily_to_windows, same height/soil
AUX source, same species-signature meta-features) - only the TARGET and
MODEL differ: instead of a 4-stage classification hierarchy, a single
RandomForestRegressor with one output per species (sklearn's
RandomForestRegressor supports multi-output natively, no wrapper needed).

Target construction: composition_base_2026_subregion2.gpkg carries
composition as up to 5 VARIABLE slots (liik_1..5/osakaal_1..5, already
normalized to the project's 7-class HB/KS/KU/LM/LV/MA/MUU scheme by
scripts/build_composition_base_dataset.py). This script pivots those
slots into a FIXED-width target matrix (one column per species, summing
any repeated-species slots, though repeats are not expected), then
renormalizes each row to sum to exactly 100 (raw osakaalude_summa isn't
always exactly 100 - rounding in the source data).

Evaluation, two angles:
  1. Direct regression metrics per species (R2, MAE, RMSE) plus a
     macro-averaged summary - the actual composition-prediction task.
  2. A derived classification check: the species with the highest
     PREDICTED share is treated as a predicted dominant species and
     compared against the true peapuuliik via the same evaluate_model()
     used by every classification experiment - this ties the regression
     result back to the project's primary metric (Macro F1) for a rough,
     if indirect, comparison against EXP032/038/044/045/048/049.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import geopandas as gpd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.timeseries import (
    aggregate_daily_to_windows,
    long_to_wide,
    merge_aux,
    encode_categoricals_consistently,
    align_columns,
    drop_all_nan_train_columns,
)
from src.preprocessing.clearcut import remove_clearcut_ids
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix_pretty
from src.paths import PATHS

# =========================================================
# FILES
# =========================================================

DAILY_DIR = PATHS.intermediate / "EXP047_composition_daily"
TRAIN_FILE = DAILY_DIR / "composition_train_sentinel2_daily_timeseries.csv"
VAL_FILE = DAILY_DIR / "composition_val_sentinel2_daily_timeseries.csv"
TEST_FILE = DAILY_DIR / "composition_test_sentinel2_daily_timeseries.csv"

PARCEL_FILE = PATHS.raw / "composition_base_2026_subregion2.gpkg"  # source of AUX + composition targets

OVERLAP_FILE = os.path.join(
    PATHS.reference, "clearcut_overlap", "stands_overlapping_clearcuts_over20pct.csv"
)

MODEL_OUT = os.path.join(PATHS.models_trained, "EXP050_rf_composition_regression.joblib")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "EXP050_test_predictions_composition.csv")
REGRESSION_METRICS_OUT = os.path.join(PATHS.outputs_tables, "EXP050_regression_metrics_per_species.csv")
CM_DERIVED_OUT = os.path.join(PATHS.outputs_figures, "EXP050_derived_dominant_species_confusion_matrix.png")

# =========================================================
# CONFIG (data pipeline identical to EXP048/049)
# =========================================================

ID_COL = "id"
TARGET_COL = "target"  # dominant species, used only for the derived-classification check + AUX/clearcut compatibility
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

RF_REGRESSOR_KWARGS = dict(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    random_state=42, n_jobs=-1
)

# =========================================================
# HELPER FUNCTIONS (species-signature features, byte-identical to EXP032/038/043/048/049)
# =========================================================


def safe_mean(df, cols):
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return np.nan
    return df[existing].mean(axis=1)


def safe_diff(df, col_late, col_early):
    if col_late in df.columns and col_early in df.columns:
        return df[col_late] - df[col_early]
    return np.nan


def add_species_signature_features(df, name):
    df = df.copy()
    print(f"\nCreating species-specific meta-features: {name}")

    if "height_max" in df.columns and "height_std" in df.columns:
        df["HB_height_index"] = df["height_max"] + df["height_std"]
        df["HB_height_product"] = df["height_max"] * df["height_std"]
    else:
        df["HB_height_index"] = np.nan
        df["HB_height_product"] = np.nan

    if all(c in df.columns for c in ["height_std", "height_mean"]):
        df["HB_structure_ratio"] = df["height_std"] / (df["height_mean"] + 1)
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

    print("Meta-features added: 18")
    return df


def load_aux_from_parcel_file():
    gdf = gpd.read_file(PARCEL_FILE, ignore_geometry=True)
    aux = gdf[[ID_COL, "elev_mean", "elev_median", "elev_std", "elev_min", "elev_max", "soil_main"]].copy()
    aux = aux.rename(columns={
        "elev_mean": "height_mean", "elev_median": "height_median", "elev_std": "height_std",
        "elev_min": "height_min", "elev_max": "height_max", "soil_main": "soil_majority",
    })
    return aux


def build_composition_targets():
    """Pivots composition_base_2026_subregion2.gpkg's variable liik_1..5/
    osakaal_1..5 slots into a fixed-width, per-species share matrix
    (one column per SPECIES_7 entry), renormalized per row to sum to
    exactly 100 - the source osakaalude_summa isn't always exactly 100
    (rounding in the underlying Metsaregister export).
    """
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


def main():
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

    train_wide = add_species_signature_features(train_wide, "train")
    val_wide = add_species_signature_features(val_wide, "val")
    test_wide = add_species_signature_features(test_wide, "test")

    train_wide, val_wide, test_wide = encode_categoricals_consistently(train_wide, val_wide, test_wide, AUX_CATEGORICAL_COLS)
    train_wide, val_wide, test_wide, feature_cols = align_columns(train_wide, val_wide, test_wide, ID_COL, TARGET_COL)

    # attach composition targets (join by id, drop rows with no usable composition)
    train_wide = train_wide.merge(composition_targets, on=ID_COL, how="left")
    val_wide = val_wide.merge(composition_targets, on=ID_COL, how="left")
    test_wide = test_wide.merge(composition_targets, on=ID_COL, how="left")

    train_wide = train_wide.dropna(subset=SHARE_COLS, how="all")
    val_wide = val_wide.dropna(subset=SHARE_COLS, how="all")
    test_wide = test_wide.dropna(subset=SHARE_COLS, how="all")

    X_train = train_wide[feature_cols].copy()
    Y_train = train_wide[SHARE_COLS].fillna(0).copy()
    dom_train = train_wide[TARGET_COL].copy()

    X_val = val_wide[feature_cols].copy()
    Y_val = val_wide[SHARE_COLS].fillna(0).copy()
    dom_val = val_wide[TARGET_COL].copy()

    X_test = test_wide[feature_cols].copy()
    Y_test = test_wide[SHARE_COLS].fillna(0).copy()
    dom_test = test_wide[TARGET_COL].copy()

    X_train, X_val, X_test, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_test)

    print("\nTrain rows with usable composition target:", len(X_train))
    print("Mean species shares (train):")
    print(Y_train.mean().round(1))

    # =========================================================
    # TRAIN-only -> VALIDATION (informational, matches project convention)
    # =========================================================
    print("\n" + "=" * 60)
    print("TRAIN-only regressor -> VALIDATION")
    print("=" * 60)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    model_trainonly = RandomForestRegressor(**RF_REGRESSOR_KWARGS)
    model_trainonly.fit(X_train_imp, Y_train)

    X_val_imp = imputer.transform(X_val)
    val_pred = renormalize_predictions(model_trainonly.predict(X_val_imp))
    report_regression_metrics("VALIDATION", Y_val.values, val_pred)

    # =========================================================
    # FINAL: TRAIN+VAL -> TEST
    # =========================================================
    print("\n" + "=" * 60)
    print("TRAIN+VAL regressor -> TEST")
    print("=" * 60)

    X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    Y_trainval = pd.concat([Y_train, Y_val], axis=0).reset_index(drop=True)

    valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()
    X_trainval = X_trainval[valid_cols_trainval].copy()
    X_test_final = X_test[valid_cols_trainval].copy()

    imputer_final = SimpleImputer(strategy="median")
    X_trainval_imp = imputer_final.fit_transform(X_trainval)
    model_final = RandomForestRegressor(**RF_REGRESSOR_KWARGS)
    model_final.fit(X_trainval_imp, Y_trainval)

    X_test_imp = imputer_final.transform(X_test_final)
    test_pred = renormalize_predictions(model_final.predict(X_test_imp))

    metrics_df = report_regression_metrics("TEST", Y_test.values, test_pred)
    metrics_df.to_csv(REGRESSION_METRICS_OUT, index=False, encoding="utf-8-sig")

    # derived dominant-species classification check
    pred_dominant = np.array(SPECIES_7)[test_pred.argmax(axis=1)]
    evaluate_model("EXP050 derived dominant species (argmax of predicted composition) - TEST", dom_test, pred_dominant)
    plot_confusion_matrix_pretty(
        dom_test, pred_dominant, "EXP050 derived dominant-species confusion matrix (test)",
        CM_DERIVED_OUT, normalize=True
    )

    test_predictions_df = pd.DataFrame({ID_COL: test_wide[ID_COL].values, "dominant_true": dom_test.values, "dominant_pred_from_composition": pred_dominant})
    for i, s in enumerate(SPECIES_7):
        test_predictions_df[f"true_{s}"] = Y_test[f"share_{s}"].values
        test_predictions_df[f"pred_{s}"] = test_pred[:, i]
    test_predictions_df.to_csv(TEST_PRED_OUT, index=False, encoding="utf-8-sig")
    print("\nTest predictions saved:", TEST_PRED_OUT)

    joblib.dump({
        "model": model_final, "imputer": imputer_final,
        "feature_cols": valid_cols_trainval,
        "species_order": SPECIES_7,
        "id_col": ID_COL, "target_col": TARGET_COL,
        "regression_metrics_test": metrics_df.to_dict(orient="records"),
        "removed_clearcut_overlap_file": OVERLAP_FILE,
        "experiment": "EXP050_composition_regression",
    }, MODEL_OUT)

    print("\nDone.")
    print("Model:", MODEL_OUT)


def renormalize_predictions(pred):
    """Clips negative predictions (shares can't be negative) then rescales
    each row to sum to exactly 100 - independent per-species regression
    outputs won't naturally sum to 100 on their own."""
    pred = np.clip(pred, 0, None)
    row_sums = pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid div-by-zero; an all-zero row stays all-zero
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
    print(f"\n================ EXP050 composition regression - {name} ================")
    print(df.to_string(index=False))
    print(f"\nOverall (all species, macro) MAE: {overall_mae:.3f} percentage points")
    return df


if __name__ == "__main__":
    main()
