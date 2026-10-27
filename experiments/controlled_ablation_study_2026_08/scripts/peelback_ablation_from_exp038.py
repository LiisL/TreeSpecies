# -*- coding: utf-8 -*-
"""'Peel-back' ablation: starts from EXP038's EXACT recipe (4-stage
hierarchy + 18 hand-built species-signature features + height/soil aux +
manual oversampling + sample_weight=balanced + XGBoost, n_estimators=400,
max_depth=6, learning_rate=0.1 - all identical to experiments/EXP038_xgboost_hierarchical/exp038.py)
and removes exactly ONE component at a time, evaluated on VALIDATION only
(train-only fit -> val, same convention as every other experiment in this
project; TEST is never touched here).

This directly tests whether hierarchy / species-signature features /
oversampling matter specifically IN COMBINATION with each other, which the
earlier "build-up" ablation (Steps 1-8) could not test (it held the other
choices fixed at their own individually-best settings throughout).

Variants:
  B0  EXP038 full recipe (reproduction/baseline for this ablation)
  B1  - oversampling      (sample_weight=balanced only, no manual_oversample)
  B2  - hierarchy         (flat XGBoost, single model, no cascade)
  B3  - signature features (auto Top-50 instead of the 18 hand-built ones)
  B4  - height/soil aux   (Sentinel-2 only; signature features degrade
                           gracefully - height-dependent ones become NaN
                           and get dropped by drop_all_nan_train_columns)
"""
import os
import sys
from pathlib import Path
import numpy as np
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
from src.models.sampling import manual_oversample
from src.evaluation.metrics import evaluate_model
from src.paths import PATHS

DATA_DIR = PATHS.intermediate
TRAIN_FILE = os.path.join(DATA_DIR, "Subregion2_train_sentinel2_10day_timeseries.csv")
VAL_FILE = os.path.join(DATA_DIR, "Subregion2_val_sentinel2_10day_timeseries.csv")
TEST_FILE = os.path.join(DATA_DIR, "Subregion2_test_sentinel2_10day_timeseries.csv")
AUX_FILE = str(Path(PATHS.raw).resolve().parents[1] / "analysis" / "EXP027_subregion2_daily_stats" / "tables" / "subregion2_parcel_height_soil.csv")
OVERLAP_FILE = os.path.join(PATHS.reference, "clearcut_overlap", "stands_overlapping_clearcuts_over20pct.csv")

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

TS_FEATURE_COLS = [
    "B2_mean", "B3_mean", "B4_mean", "B5_mean", "B6_mean", "B7_mean",
    "B8_mean", "B8A_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDRE_mean", "NDMI_mean", "NBR_mean"
]
AUX_NUMERIC_COLS = ["height_mean", "height_median", "height_std", "height_min", "height_max"]
AUX_CATEGORICAL_COLS = ["soil_majority"]

SAMPLING_STRATEGY = {"HB": 1200, "LM": 1200, "KU": 1600, "LV": 2000, "MUU": 500}
SAMPLING_STRATEGY_STAGE1 = {"MUU": 500}

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

XGB_KWARGS = dict(n_estimators=400, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)


def safe_mean(df, cols):
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return np.nan
    return df[existing].mean(axis=1)


def safe_diff(df, col_late, col_early):
    if col_late in df.columns and col_early in df.columns:
        return df[col_late] - df[col_early]
    return np.nan


def add_species_signature_features(df):
    df = df.copy()
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
    return df


def fit_xgb_stage(X, y, xgb_kwargs, sampling_strategy, target_col_name, use_oversampling):
    if use_oversampling:
        X_os, y_os = manual_oversample(X, y, sampling_strategy, target_col_name, random_state=42)
    else:
        X_os, y_os = X, y
    imputer = SimpleImputer(strategy="median")
    X_os_imp = imputer.fit_transform(X_os)
    le = LabelEncoder()
    y_os_enc = le.fit_transform(y_os)
    sample_weight = compute_sample_weight("balanced", y_os_enc)
    model = XGBClassifier(**xgb_kwargs)
    model.fit(X_os_imp, y_os_enc, sample_weight=sample_weight)
    return model, imputer, le


def train_hierarchical(X_train, y_train, use_oversampling):
    y_group = y_train.map(GROUP_MAP)
    stage1, imputer1, le1 = fit_xgb_stage(X_train, y_group, XGB_KWARGS, SAMPLING_STRATEGY_STAGE1, GROUP_COL, use_oversampling)

    conifer_mask = y_train.isin(CONIFER_SPECIES)
    stage2a, imputer2a, le2a = fit_xgb_stage(X_train[conifer_mask], y_train[conifer_mask], XGB_KWARGS, SAMPLING_STRATEGY, TARGET_COL, use_oversampling)

    deciduous_mask = y_train.isin(DECIDUOUS_SPECIES)
    y_lv_vs_rest = y_train[deciduous_mask].apply(lambda s: "LV" if s == "LV" else "REST")
    sampling_strategy_lv = {"LV": SAMPLING_STRATEGY.get("LV", 2000)}
    stage2bi, imputer2bi, le2bi = fit_xgb_stage(X_train[deciduous_mask], y_lv_vs_rest, XGB_KWARGS, sampling_strategy_lv, LV_VS_REST_COL, use_oversampling)

    rest_mask = y_train.isin(REST_OF_DECIDUOUS_SPECIES)
    stage2bii, imputer2bii, le2bii = fit_xgb_stage(X_train[rest_mask], y_train[rest_mask], XGB_KWARGS, SAMPLING_STRATEGY, TARGET_COL, use_oversampling)

    return {
        "stage1": stage1, "imputer1": imputer1, "le1": le1,
        "stage2a": stage2a, "imputer2a": imputer2a, "le2a": le2a,
        "stage2bi": stage2bi, "imputer2bi": imputer2bi, "le2bi": le2bi,
        "stage2bii": stage2bii, "imputer2bii": imputer2bii, "le2bii": le2bii,
    }


def predict_hierarchical(X, m):
    X_imp1 = m["imputer1"].transform(X)
    group_pred = m["le1"].inverse_transform(m["stage1"].predict(X_imp1))
    final_pred = np.empty(len(X), dtype=object)

    conifer_rows = group_pred == "CONIFER"
    if conifer_rows.any():
        X_c = m["imputer2a"].transform(X.loc[conifer_rows])
        final_pred[conifer_rows] = m["le2a"].inverse_transform(m["stage2a"].predict(X_c))

    deciduous_rows = group_pred == "DECIDUOUS"
    if deciduous_rows.any():
        X_d = m["imputer2bi"].transform(X.loc[deciduous_rows])
        lv_pred = m["le2bi"].inverse_transform(m["stage2bi"].predict(X_d))
        is_lv = lv_pred == "LV"
        deciduous_idx = np.where(deciduous_rows)[0]
        lv_idx = deciduous_idx[is_lv]
        rest_idx = deciduous_idx[~is_lv]
        final_pred[lv_idx] = "LV"
        if len(rest_idx) > 0:
            X_r = m["imputer2bii"].transform(X.iloc[rest_idx])
            final_pred[rest_idx] = m["le2bii"].inverse_transform(m["stage2bii"].predict(X_r))

    muu_rows = group_pred == "MUU"
    final_pred[muu_rows] = "MUU"
    return final_pred


def build_features(use_aux, use_signature):
    train_long = pd.read_csv(TRAIN_FILE)
    val_long = pd.read_csv(VAL_FILE)
    test_long = pd.read_csv(TEST_FILE)

    train_long, val_long, test_long = remove_clearcut_ids(train_long, val_long, test_long, OVERLAP_FILE, ID_COL, TARGET_COL)

    train_wide = long_to_wide(train_long, "train", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
    val_wide = long_to_wide(val_long, "val", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
    test_wide = long_to_wide(test_long, "test", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)

    if use_aux:
        aux = pd.read_csv(AUX_FILE)
        train_wide = merge_aux(train_wide, aux, "train", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
        val_wide = merge_aux(val_wide, aux, "val", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
        test_wide = merge_aux(test_wide, aux, "test", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)

    if use_signature:
        train_wide = add_species_signature_features(train_wide)
        val_wide = add_species_signature_features(val_wide)
        test_wide = add_species_signature_features(test_wide)

    if use_aux:
        train_wide, val_wide, test_wide = encode_categoricals_consistently(train_wide, val_wide, test_wide, AUX_CATEGORICAL_COLS)

    train_wide, val_wide, test_wide, feature_cols = align_columns(train_wide, val_wide, test_wide, ID_COL, TARGET_COL)

    X_train = train_wide[feature_cols].copy()
    y_train = train_wide[TARGET_COL].copy()
    X_val = val_wide[feature_cols].copy()
    y_val = val_wide[TARGET_COL].copy()

    X_train, X_val, _, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_val.copy())
    return X_train, y_train, X_val, y_val, feature_cols_used


def top50_from_rf(X_train, y_train):
    imputer0 = SimpleImputer(strategy="median")
    X_train_imp0 = imputer0.fit_transform(X_train)
    rf0 = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    rf0.fit(X_train_imp0, y_train)
    importance_df = pd.DataFrame({"feature": X_train.columns.tolist(), "importance": rf0.feature_importances_}).sort_values("importance", ascending=False)
    return importance_df.head(50)["feature"].tolist()


results = {}

# =========================================================
# B0: EXP038 full recipe (hierarchy + signature + aux + oversampling)
# =========================================================
print("\n" + "=" * 70 + "\nB0: EXP038 full recipe\n" + "=" * 70)
X_train, y_train, X_val, y_val, _ = build_features(use_aux=True, use_signature=True)
m = train_hierarchical(X_train, y_train, use_oversampling=True)
pred = predict_hierarchical(X_val, m)
results["B0_full_recipe"] = evaluate_model("B0 EXP038 full recipe - VALIDATION", y_val, pred)

# =========================================================
# B1: - oversampling (sample_weight only)
# =========================================================
print("\n" + "=" * 70 + "\nB1: - oversampling\n" + "=" * 70)
m = train_hierarchical(X_train, y_train, use_oversampling=False)
pred = predict_hierarchical(X_val, m)
results["B1_no_oversampling"] = evaluate_model("B1 - oversampling - VALIDATION", y_val, pred)

# =========================================================
# B2: - hierarchy (flat XGBoost)
# =========================================================
print("\n" + "=" * 70 + "\nB2: - hierarchy (flat)\n" + "=" * 70)
model_flat, imputer_flat, le_flat = fit_xgb_stage(X_train, y_train, XGB_KWARGS, SAMPLING_STRATEGY, TARGET_COL, use_oversampling=True)
X_val_imp_flat = imputer_flat.transform(X_val)
pred_flat = le_flat.inverse_transform(model_flat.predict(X_val_imp_flat))
results["B2_no_hierarchy_flat"] = evaluate_model("B2 - hierarchy (flat) - VALIDATION", y_val, pred_flat)

# =========================================================
# B3: - signature features (auto Top-50 instead), aux still included
# =========================================================
print("\n" + "=" * 70 + "\nB3: - signature features (Top-50 instead)\n" + "=" * 70)
X_train_nosig, y_train_nosig, X_val_nosig, y_val_nosig, _ = build_features(use_aux=True, use_signature=False)
top50 = top50_from_rf(X_train_nosig, y_train_nosig)
X_train_top50 = X_train_nosig[top50].copy()
X_val_top50 = X_val_nosig[top50].copy()
m = train_hierarchical(X_train_top50, y_train_nosig, use_oversampling=True)
pred = predict_hierarchical(X_val_top50, m)
results["B3_top50_instead_of_signature"] = evaluate_model("B3 - signature features (Top-50) - VALIDATION", y_val_nosig, pred)

# =========================================================
# B4: - height/soil aux (Sentinel-2 only, signature features degrade gracefully)
# =========================================================
print("\n" + "=" * 70 + "\nB4: - height/soil aux\n" + "=" * 70)
X_train_noaux, y_train_noaux, X_val_noaux, y_val_noaux, _ = build_features(use_aux=False, use_signature=True)
m = train_hierarchical(X_train_noaux, y_train_noaux, use_oversampling=True)
pred = predict_hierarchical(X_val_noaux, m)
results["B4_no_aux"] = evaluate_model("B4 - height/soil aux - VALIDATION", y_val_noaux, pred)

# =========================================================
# SUMMARY
# =========================================================
print("\n\n" + "=" * 70)
print("PEEL-BACK ABLATION SUMMARY (validation, starting from EXP038's recipe)")
print("=" * 70)
for label, m_ in results.items():
    print(f"{label:35s} Acc={m_['accuracy']:.4f}  BalAcc={m_['balanced_accuracy']:.4f}  MacroF1={m_['macro_f1']:.4f}  Kappa={m_['kappa']:.4f}")

joblib.dump(results, os.path.join(PATHS.models_trained, "EXP038_peelback_ablation_results.joblib"))
print("\nSaved: EXP038_peelback_ablation_results.joblib")
