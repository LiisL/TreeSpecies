# -*- coding: utf-8 -*-
"""FINAL single test-set evaluation for candidate B1: EXP038's exact recipe
(4-stage hierarchy + 18 hand-built species-signature features + height/soil
aux + XGBoost n_estimators=400/max_depth=6/learning_rate=0.1) MINUS manual
oversampling (sample_weight=balanced only). This was the winning variant of
the peel-back ablation (validation Macro F1 0.7038 vs EXP038's own 0.6892).

Refits on TRAIN+VAL (standard project convention), evaluates ONCE on TEST -
the single test-set touch for this candidate.
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
from sklearn.metrics import precision_score, recall_score, f1_score
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

MODEL_OUT = os.path.join(PATHS.models_trained, "B1_exp038_no_oversampling_FINAL.joblib")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "B1_FINAL_test_predictions.csv")
CM_OUT = os.path.join(PATHS.outputs_figures, "B1_FINAL_confusion_matrix_test.png")


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


def fit_xgb_stage_no_oversampling(X, y, xgb_kwargs):
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    sample_weight = compute_sample_weight("balanced", y_enc)
    model = XGBClassifier(**xgb_kwargs)
    model.fit(X_imp, y_enc, sample_weight=sample_weight)
    return model, imputer, le


def train_hierarchical_no_oversampling(X_train, y_train):
    y_group = y_train.map(GROUP_MAP)
    stage1, imputer1, le1 = fit_xgb_stage_no_oversampling(X_train, y_group, XGB_KWARGS)

    conifer_mask = y_train.isin(CONIFER_SPECIES)
    stage2a, imputer2a, le2a = fit_xgb_stage_no_oversampling(X_train[conifer_mask], y_train[conifer_mask], XGB_KWARGS)

    deciduous_mask = y_train.isin(DECIDUOUS_SPECIES)
    y_lv_vs_rest = y_train[deciduous_mask].apply(lambda s: "LV" if s == "LV" else "REST")
    stage2bi, imputer2bi, le2bi = fit_xgb_stage_no_oversampling(X_train[deciduous_mask], y_lv_vs_rest, XGB_KWARGS)

    rest_mask = y_train.isin(REST_OF_DECIDUOUS_SPECIES)
    stage2bii, imputer2bii, le2bii = fit_xgb_stage_no_oversampling(X_train[rest_mask], y_train[rest_mask], XGB_KWARGS)

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
    return final_pred, group_pred


# =========================================================
# LOAD + PREP (identical to EXP038 / peel-back B0-B4)
# =========================================================
train_long = pd.read_csv(TRAIN_FILE)
val_long = pd.read_csv(VAL_FILE)
test_long = pd.read_csv(TEST_FILE)
aux = pd.read_csv(AUX_FILE)

train_long, val_long, test_long = remove_clearcut_ids(train_long, val_long, test_long, OVERLAP_FILE, ID_COL, TARGET_COL)

train_wide = long_to_wide(train_long, "train", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
val_wide = long_to_wide(val_long, "val", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
test_wide = long_to_wide(test_long, "test", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)

train_wide = merge_aux(train_wide, aux, "train", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
val_wide = merge_aux(val_wide, aux, "val", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
test_wide = merge_aux(test_wide, aux, "test", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)

train_wide = add_species_signature_features(train_wide)
val_wide = add_species_signature_features(val_wide)
test_wide = add_species_signature_features(test_wide)

train_wide, val_wide, test_wide = encode_categoricals_consistently(train_wide, val_wide, test_wide, AUX_CATEGORICAL_COLS)
train_wide, val_wide, test_wide, feature_cols = align_columns(train_wide, val_wide, test_wide, ID_COL, TARGET_COL)

X_train = train_wide[feature_cols].copy()
y_train = train_wide[TARGET_COL].copy()
X_val = val_wide[feature_cols].copy()
y_val = val_wide[TARGET_COL].copy()
X_test = test_wide[feature_cols].copy()
y_test = test_wide[TARGET_COL].copy()

X_train, X_val, X_test, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_test)

# =========================================================
# Sanity check: reproduce B1's validation number (train-only -> val)
# =========================================================
print("\n--- Sanity check: reproducing peel-back B1 VALIDATION result ---")
m_check = train_hierarchical_no_oversampling(X_train, y_train)
val_pred_check, _ = predict_hierarchical(X_val, m_check)
val_check_metrics = evaluate_model("Sanity check B1 - VALIDATION", y_val, val_pred_check)

# =========================================================
# FINAL: TRAIN+VAL -> TEST (single touch)
# =========================================================
print("\n--- FINAL model: TRAIN+VAL -> TEST (the one and only test-set touch) ---")
X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()
X_trainval = X_trainval[valid_cols_trainval].copy()
X_test_final = X_test[valid_cols_trainval].copy()

m_final = train_hierarchical_no_oversampling(X_trainval, y_trainval)
test_pred, test_group_pred = predict_hierarchical(X_test_final, m_final)

test_metrics = evaluate_model("FINAL B1 (EXP038 - oversampling) - TEST", y_test, test_pred)
test_metrics["hb_precision"] = precision_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_recall"] = recall_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_f1"] = f1_score(y_test, test_pred, labels=["HB"], average="macro", zero_division=0)
print(f"\nHB precision={test_metrics['hb_precision']:.4f}, HB recall={test_metrics['hb_recall']:.4f}, HB F1={test_metrics['hb_f1']:.4f}")

# 6-class (excl. MUU singleton) comparison, matching the earlier corrected methodology
six = ["HB", "KS", "KU", "LM", "LV", "MA"]
mask6 = y_test.isin(six)
from sklearn.metrics import f1_score as f1s, balanced_accuracy_score as bas
macro_f1_6 = f1s(y_test[mask6], pd.Series(test_pred)[mask6.values], average="macro", labels=six)
balacc_6 = bas(y_test[mask6], pd.Series(test_pred)[mask6.values])
print(f"\n6-class (excl. MUU) Macro F1: {macro_f1_6:.4f}, BalAcc: {balacc_6:.4f}")

plot_confusion_matrix_pretty(y_test, test_pred, "FINAL B1 (EXP038 - oversampling) - test confusion matrix", CM_OUT, normalize=False)

test_predictions_df = pd.DataFrame({ID_COL: test_wide[ID_COL].values, "y_true": y_test.values, "y_pred": test_pred})
test_predictions_df.to_csv(TEST_PRED_OUT, index=False, encoding="utf-8-sig")

joblib.dump({
    "stage1": m_final["stage1"], "imputer1": m_final["imputer1"], "le1": m_final["le1"],
    "stage2a": m_final["stage2a"], "imputer2a": m_final["imputer2a"], "le2a": m_final["le2a"],
    "stage2bi": m_final["stage2bi"], "imputer2bi": m_final["imputer2bi"], "le2bi": m_final["le2bi"],
    "stage2bii": m_final["stage2bii"], "imputer2bii": m_final["imputer2bii"], "le2bii": m_final["le2bii"],
    "feature_cols": valid_cols_trainval,
    "val_check_metrics": val_check_metrics, "test_metrics": test_metrics,
    "macro_f1_6class": macro_f1_6, "balacc_6class": balacc_6,
    "removed_clearcut_overlap_file": OVERLAP_FILE,
    "experiment": "B1_exp038_no_oversampling_FINAL",
}, MODEL_OUT)

print("\n\n" + "=" * 70)
print("FINAL COMPARISON (test set, 6-class excl. MUU singleton)")
print("=" * 70)
print("EXP038 (old primary)     6-class MacroF1=0.7992  BalAcc=0.7805")
print(f"B1 (- oversampling)      6-class MacroF1={macro_f1_6:.4f}  BalAcc={balacc_6:.4f}")
print("\nSaved:", MODEL_OUT)
