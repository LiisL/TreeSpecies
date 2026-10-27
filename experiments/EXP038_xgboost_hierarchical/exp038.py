"""EXP038: EXP032's exact 4-stage hierarchy (current primary), with ONE
change - XGBoost (gradient boosting) instead of Random Forest at every
stage. Isolates the CLASSIFIER FAMILY as the only new variable: same
features, same species-signature meta-features, same oversampling
strategy, same predicted-routing discipline. First real test of a
non-RF model family in this project (the thesis's own ch. 2.6 already
name-checks XGBoost as an alternative worth comparing against).

Implementation notes, since XGBoost's sklearn API differs from RF's in
ways that needed handling, not just a class-name swap:
  - XGBoost's installed version (3.3.0) does NOT accept string class
    labels directly (unlike sklearn's RandomForestClassifier) - confirmed
    by a failing smoke test ("Invalid classes inferred... expected [0 1
    2], got ['HB' 'KS' 'MA']"). Each stage needs its own LabelEncoder,
    fit alongside the model, to encode y before training and decode
    predictions back to species strings before they're used for cascade
    routing (which compares against string literals like "CONIFER").
  - XGBoost has no `class_weight="balanced"` parameter - replicated the
    same balancing INTENT via `sklearn.utils.class_weight.compute_sample_weight("balanced", y)`
    passed as `sample_weight` to `.fit()`, applied AFTER the same
    `manual_oversample` step RF used - so both mechanisms RF combined
    (oversampling + balanced weighting) are still both present, just the
    weighting implementation differs to match XGBoost's API.
  - XGBoost has NATIVE missing-value handling (learns a default split
    direction for NaN during training) - confirmed working in a smoke
    test with injected NaNs. This experiment deliberately does NOT use
    that yet - it keeps the SAME SimpleImputer median-imputation as RF,
    to isolate the classifier as the only variable. A native-NaN variant
    (dropping the imputer) is a natural, cheap follow-up if this looks
    promising, not conflated with this first clean comparison.
  - Hyperparameters (`n_estimators=400, max_depth=6, learning_rate=0.1`)
    are reasonable defaults, not a tuned search - XGBoost conventionally
    uses much shallower trees than RF's unlimited depth (boosting many
    weak learners vs. RF's many independent deep ones), so max_depth=6
    (not RF's None) reflects that difference in how the two algorithms
    are meant to be used, not an attempt to handicap either one.

Compare against: EXP029 (Accuracy 0.823, Macro F1 0.620), EXP032 (current
primary, Accuracy 0.8219, Macro F1 0.6246, HB F1 0.5161).
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.timeseries import (
    long_to_wide,
    merge_aux,
    encode_categoricals_consistently,
    align_columns,
    drop_all_nan_train_columns,
)
from src.preprocessing.clearcut import remove_clearcut_ids
from src.models.sampling import manual_oversample
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix_pretty
from src.paths import PATHS

# =========================================================
# FILES (identical to EXP029/EXP031)
# =========================================================

DATA_DIR = PATHS.intermediate

TRAIN_FILE = os.path.join(DATA_DIR, "Subregion2_train_sentinel2_10day_timeseries.csv")
VAL_FILE   = os.path.join(DATA_DIR, "Subregion2_val_sentinel2_10day_timeseries.csv")
TEST_FILE  = os.path.join(DATA_DIR, "Subregion2_test_sentinel2_10day_timeseries.csv")

AUX_FILE = str(
    Path(PATHS.raw).resolve().parents[1]
    / "experiments" / "EXP027_subregion2_daily_stats" / "tables" / "subregion2_parcel_height_soil.csv"
)

OVERLAP_FILE = os.path.join(
    PATHS.reference,
    "clearcut_overlap",
    "stands_overlapping_clearcuts_over20pct.csv"
)

MODEL_OUT = os.path.join(PATHS.models_trained, "EXP038_xgb_hierarchical.joblib")
FI_OUT_STAGE1 = os.path.join(PATHS.outputs_tables, "EXP038_feature_importance_stage1_group.csv")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "EXP038_test_predictions.csv")
CM_OUT = os.path.join(PATHS.outputs_figures, "EXP038_confusion_matrix_test_normalized.png")
CM_STAGE1_OUT = os.path.join(PATHS.outputs_figures, "EXP038_stage1_group_confusion_matrix.png")
CM_STAGE2BI_OUT = os.path.join(PATHS.outputs_figures, "EXP038_stage2bi_lv_vs_rest_confusion_matrix.png")

# =========================================================
# CONFIG (identical to EXP031 except the extra deciduous cascade)
# =========================================================

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

TS_FEATURE_COLS = [
    "B2_mean", "B3_mean", "B4_mean", "B5_mean", "B6_mean", "B7_mean",
    "B8_mean", "B8A_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "NDRE_mean", "NDMI_mean", "NBR_mean"
]

AUX_NUMERIC_COLS = [
    "height_mean", "height_median", "height_std", "height_min", "height_max"
]
AUX_CATEGORICAL_COLS = ["soil_majority"]

# original EXP021/EXP029 species-level oversampling target, reused as-is
# throughout (isolates the hierarchical ARCHITECTURE as the only new
# variable, not the oversampling policy)
SAMPLING_STRATEGY = {
    "HB": 1200, "LM": 1200, "KU": 1600, "LV": 2000, "MUU": 500
}

GROUP_MAP = {
    "KU": "CONIFER", "MA": "CONIFER",
    "HB": "DECIDUOUS", "KS": "DECIDUOUS", "LM": "DECIDUOUS", "LV": "DECIDUOUS",
    "MUU": "MUU"
}
CONIFER_SPECIES = {"KU", "MA"}
DECIDUOUS_SPECIES = {"HB", "KS", "LM", "LV"}
REST_OF_DECIDUOUS_SPECIES = {"HB", "KS", "LM"}  # deciduous minus LV

SAMPLING_STRATEGY_STAGE1 = {"MUU": 500}
GROUP_COL = "target_group"
LV_VS_REST_COL = "target_lv_vs_rest"

XGB_KWARGS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.1,
    random_state=42, n_jobs=-1
)

# =========================================================
# HELPER FUNCTIONS (species-signature features, identical to EXP029/031)
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

    df["HB_spring_index"] = safe_mean(
        df, ["NDVI_mean_2025_05_01", "NDMI_mean_2025_05_01", "NBR_mean_2025_05_01"]
    )
    df["HB_autumn_ndre"] = safe_mean(
        df, ["NDRE_mean_2025_09_28", "NDRE_mean_2025_10_08", "NDRE_mean_2025_10_18"]
    )
    df["KS_moisture_index"] = safe_mean(
        df, ["NDMI_mean_2025_09_28", "NDRE_mean_2025_09_28", "B11_mean_2025_09_28"]
    )
    df["KU_winter_index"] = safe_mean(
        df, ["B11_mean_2025_02_20", "B12_mean_2025_02_20", "NDVI_mean_2025_02_20"]
    )
    df["KU_autumn_index"] = safe_mean(
        df, ["NDRE_mean_2025_09_28", "NDVI_mean_2025_10_18", "NDMI_mean_2025_09_28"]
    )
    df["LM_autumn_index"] = safe_mean(
        df, ["NDRE_mean_2025_09_28", "NDVI_mean_2025_09_28", "B8_mean_2025_09_28", "B7_mean_2025_09_28"]
    )
    df["LV_july_index"] = safe_mean(
        df, ["B6_mean_2025_07_20", "B7_mean_2025_07_20", "B8_mean_2025_07_20", "B8A_mean_2025_07_20"]
    )
    df["LV_july_swir_index"] = safe_mean(df, ["B11_mean_2025_07_20", "B12_mean_2025_07_20"])
    df["MA_spring_nir"] = safe_mean(
        df, ["B8_mean_2025_05_01", "B8A_mean_2025_05_01", "B6_mean_2025_05_01", "NDVI_mean_2025_05_01"]
    )

    df["NDVI_change_spring"] = safe_diff(df, "NDVI_mean_2025_05_21", "NDVI_mean_2025_04_11")
    df["NDVI_change_summer"] = safe_diff(df, "NDVI_mean_2025_07_20", "NDVI_mean_2025_05_21")
    df["NDVI_change_autumn"] = safe_diff(df, "NDVI_mean_2025_10_18", "NDVI_mean_2025_09_08")
    df["NDRE_change_autumn"] = safe_diff(df, "NDRE_mean_2025_10_18", "NDRE_mean_2025_09_08")
    df["NDMI_change_autumn"] = safe_diff(df, "NDMI_mean_2025_10_18", "NDMI_mean_2025_09_08")
    df["NBR_change_autumn"] = safe_diff(df, "NBR_mean_2025_10_18", "NBR_mean_2025_09_08")

    print("Meta-features added: 18")
    return df


def fit_xgb_stage(X_os_imp, y_os_str, xgb_kwargs):
    """XGBoost's installed version needs numeric-encoded labels (unlike
    RF) and has no `class_weight` param (unlike RF) - a LabelEncoder plus
    `compute_sample_weight("balanced", ...)` replicate the same two
    things RF's `class_weight="balanced"` + manual_oversample did
    together, just via XGBoost's own API shape.
    """
    le = LabelEncoder()
    y_os_enc = le.fit_transform(y_os_str)
    sample_weight = compute_sample_weight("balanced", y_os_enc)
    model = XGBClassifier(**xgb_kwargs)
    model.fit(X_os_imp, y_os_enc, sample_weight=sample_weight)
    return model, le


def train_hierarchical(X_train, y_train, sampling_strategy_stage1, sampling_strategy_species, xgb_kwargs):
    """Trains the 4-model hierarchy on TRAIN(+VAL) data."""
    y_group = y_train.map(GROUP_MAP)
    assert y_group.isna().sum() == 0, "Unmapped species found in GROUP_MAP"

    # --- Stage 1: group classifier ---
    X_group_os, y_group_os = manual_oversample(
        X_train, y_group, sampling_strategy_stage1, GROUP_COL, random_state=42
    )
    imputer1 = SimpleImputer(strategy="median")
    X_group_os_imp = imputer1.fit_transform(X_group_os)
    stage1, le1 = fit_xgb_stage(X_group_os_imp, y_group_os, xgb_kwargs)

    # --- Stage 2a: conifer species (KU vs MA), true-conifer rows only ---
    conifer_mask = y_train.isin(CONIFER_SPECIES)
    X_conifer_os, y_conifer_os = manual_oversample(
        X_train[conifer_mask], y_train[conifer_mask], sampling_strategy_species, TARGET_COL, random_state=42
    )
    imputer2a = SimpleImputer(strategy="median")
    X_conifer_os_imp = imputer2a.fit_transform(X_conifer_os)
    stage2a, le2a = fit_xgb_stage(X_conifer_os_imp, y_conifer_os, xgb_kwargs)

    # --- Stage 2b-i: LV vs REST(HB/KS/LM), true-deciduous rows only ---
    deciduous_mask = y_train.isin(DECIDUOUS_SPECIES)
    y_lv_vs_rest = y_train[deciduous_mask].apply(lambda s: "LV" if s == "LV" else "REST")
    sampling_strategy_lv = {"LV": sampling_strategy_species.get("LV", 2000)}
    X_lv_os, y_lv_os = manual_oversample(
        X_train[deciduous_mask], y_lv_vs_rest, sampling_strategy_lv, LV_VS_REST_COL, random_state=42
    )
    imputer2bi = SimpleImputer(strategy="median")
    X_lv_os_imp = imputer2bi.fit_transform(X_lv_os)
    stage2bi, le2bi = fit_xgb_stage(X_lv_os_imp, y_lv_os, xgb_kwargs)

    # --- Stage 2b-ii: HB vs KS vs LM, true rest-of-deciduous rows only ---
    rest_mask = y_train.isin(REST_OF_DECIDUOUS_SPECIES)
    X_rest_os, y_rest_os = manual_oversample(
        X_train[rest_mask], y_train[rest_mask], sampling_strategy_species, TARGET_COL, random_state=42
    )
    imputer2bii = SimpleImputer(strategy="median")
    X_rest_os_imp = imputer2bii.fit_transform(X_rest_os)
    stage2bii, le2bii = fit_xgb_stage(X_rest_os_imp, y_rest_os, xgb_kwargs)

    return {
        "stage1": stage1, "imputer1": imputer1, "le1": le1,
        "stage2a": stage2a, "imputer2a": imputer2a, "le2a": le2a,
        "stage2bi": stage2bi, "imputer2bi": imputer2bi, "le2bi": le2bi,
        "stage2bii": stage2bii, "imputer2bii": imputer2bii, "le2bii": le2bii,
    }


def predict_hierarchical(X, m):
    """Realistic end-to-end inference: every stage routes using the
    PREVIOUS stage's PREDICTED output, never ground truth. Returns
    (final_species_pred, stage1_group_pred, stage2bi_lv_vs_rest_pred)
    - the latter is np.nan for rows never routed through stage 2b-i.
    Each stage's raw integer prediction is decoded back to its original
    species string via that stage's own LabelEncoder before being used
    for routing (which compares against string literals like "CONIFER").
    """
    X_imp1 = m["imputer1"].transform(X)
    group_pred = m["le1"].inverse_transform(m["stage1"].predict(X_imp1))

    final_pred = np.empty(len(X), dtype=object)
    lv_vs_rest_pred = np.full(len(X), np.nan, dtype=object)

    conifer_rows = group_pred == "CONIFER"
    if conifer_rows.any():
        X_conifer_imp = m["imputer2a"].transform(X.loc[conifer_rows])
        final_pred[conifer_rows] = m["le2a"].inverse_transform(m["stage2a"].predict(X_conifer_imp))

    deciduous_rows = group_pred == "DECIDUOUS"
    if deciduous_rows.any():
        X_deciduous_imp = m["imputer2bi"].transform(X.loc[deciduous_rows])
        lv_pred = m["le2bi"].inverse_transform(m["stage2bi"].predict(X_deciduous_imp))
        lv_vs_rest_pred[deciduous_rows] = lv_pred

        is_lv = lv_pred == "LV"
        deciduous_idx = np.where(deciduous_rows)[0]
        lv_idx = deciduous_idx[is_lv]
        rest_idx = deciduous_idx[~is_lv]

        final_pred[lv_idx] = "LV"
        if len(rest_idx) > 0:
            X_rest_imp = m["imputer2bii"].transform(X.iloc[rest_idx])
            final_pred[rest_idx] = m["le2bii"].inverse_transform(m["stage2bii"].predict(X_rest_imp))

    muu_rows = group_pred == "MUU"
    final_pred[muu_rows] = "MUU"

    return final_pred, group_pred, lv_vs_rest_pred


# =========================================================
# 1-4. LOAD, CLEAN, PIVOT, AUX (identical to EXP029/031)
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

train_wide = add_species_signature_features(train_wide, "train")
val_wide = add_species_signature_features(val_wide, "val")
test_wide = add_species_signature_features(test_wide, "test")

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

print("\nSpecies distribution (train):")
print(y_train.value_counts())

# =========================================================
# 5. TRAIN-ONLY MODEL -> VAL (informational, matches EXP029/030/031 convention)
# =========================================================

print("\n" + "=" * 60)
print("TRAIN-only hierarchy -> VALIDATION")
print("=" * 60)

m_trainonly = train_hierarchical(X_train, y_train, SAMPLING_STRATEGY_STAGE1, SAMPLING_STRATEGY, XGB_KWARGS)
val_final_pred, val_group_pred, val_lv_pred = predict_hierarchical(X_val, m_trainonly)

evaluate_model("EXP038 STAGE 1 (group) - VALIDATION", y_val.map(GROUP_MAP), val_group_pred)
val_metrics = evaluate_model("EXP038 hierarchical (combined) - VALIDATION", y_val, val_final_pred)

# =========================================================
# 6. FINAL MODEL: TRAIN+VAL -> TEST
# =========================================================

print("\n" + "=" * 60)
print("TRAIN+VAL hierarchy -> TEST")
print("=" * 60)

X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()
X_trainval = X_trainval[valid_cols_trainval].copy()
X_test_final = X_test[valid_cols_trainval].copy()

m_final = train_hierarchical(X_trainval, y_trainval, SAMPLING_STRATEGY_STAGE1, SAMPLING_STRATEGY, XGB_KWARGS)
test_final_pred, test_group_pred, test_lv_pred = predict_hierarchical(X_test_final, m_final)

test_group_true = y_test.map(GROUP_MAP)
stage1_test_metrics = evaluate_model("EXP038 STAGE 1 (group) - TEST", test_group_true, test_group_pred)
plot_confusion_matrix_pretty(
    test_group_true, test_group_pred, "EXP038 stage-1 group confusion matrix (test)",
    CM_STAGE1_OUT, normalize=True
)

# stage 2b-i evaluation, restricted to rows that ACTUALLY reached this
# stage (true deciduous AND correctly routed there by stage 1) - rows
# misrouted by stage 1 never got a stage-2b-i prediction (still NaN in
# test_lv_pred), so including them here would mix floats (NaN) with
# strings and break sklearn's metric functions, as well as conflating two
# different error sources into one number.
deciduous_test_mask = y_test.isin(DECIDUOUS_SPECIES).values
reached_stage2bi_mask = deciduous_test_mask & (test_group_pred == "DECIDUOUS")
y_lv_true_test = y_test[reached_stage2bi_mask].apply(lambda s: "LV" if s == "LV" else "REST")
y_lv_pred_test = pd.Series(test_lv_pred[reached_stage2bi_mask])
stage2bi_test_metrics = evaluate_model(
    "EXP038 STAGE 2b-i (LV vs REST, true-deciduous test rows) - TEST", y_lv_true_test, y_lv_pred_test
)
plot_confusion_matrix_pretty(
    y_lv_true_test, y_lv_pred_test, "EXP038 stage-2b-i LV-vs-rest confusion matrix (test)",
    CM_STAGE2BI_OUT, normalize=True
)

test_predictions_df = pd.DataFrame({
    ID_COL: test_wide[ID_COL].values,
    "y_true": y_test.values,
    "y_pred": test_final_pred,
    "stage1_group_true": test_group_true.values,
    "stage1_group_pred": test_group_pred,
    "stage2bi_lv_vs_rest_pred": test_lv_pred,
})
test_predictions_df.to_csv(TEST_PRED_OUT, index=False, encoding="utf-8-sig")
print("\nTest predictions saved:", TEST_PRED_OUT)

test_metrics = evaluate_model("EXP038 hierarchical (combined) - TEST", y_test, test_final_pred)
plot_confusion_matrix_pretty(
    y_test, test_final_pred, "EXP038 test CM (conifer/deciduous + LV-first split)",
    CM_OUT, normalize=True
)
test_metrics["hb_precision"] = precision_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_recall"] = recall_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_f1"] = f1_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
print(f"\nHB precision={test_metrics['hb_precision']:.4f}, "
      f"HB recall={test_metrics['hb_recall']:.4f}, HB F1={test_metrics['hb_f1']:.4f}")

# error attribution: how much final error comes from each stage misrouting?
group_misrouted = test_group_true.values != test_group_pred
n_group_misrouted = int(group_misrouted.sum())
lv_misrouted = np.zeros(len(y_test), dtype=bool)
lv_misrouted[reached_stage2bi_mask] = (y_lv_true_test.values != y_lv_pred_test.values)
n_lv_misrouted_correctly_grouped = int((lv_misrouted & ~group_misrouted).sum())
print(f"\nStage-1 (group) misrouted: {n_group_misrouted} / {len(y_test)}")
print(f"Stage-2b-i (LV-vs-rest) misrouted, among correctly-grouped deciduous rows: "
      f"{n_lv_misrouted_correctly_grouped} / {int((~group_misrouted & deciduous_test_mask).sum())}")

importance_df_stage1 = pd.DataFrame({
    "feature": valid_cols_trainval, "importance": m_final["stage1"].feature_importances_
}).sort_values("importance", ascending=False)
importance_df_stage1.to_csv(FI_OUT_STAGE1, index=False, encoding="utf-8-sig")

# NOTE: stage 2b-i's fitted feature count (730) differs from the global
# valid_cols_trainval (744) because SimpleImputer silently drops columns
# that are all-NaN specifically WITHIN the deciduous-only training subset
# (a different set than the ones dropped globally) - this doesn't affect
# correctness of the live pipeline (imputer2bi.transform always receives
# the full 744-column input and internally applies the same drop it
# learned at fit time), it just means the column NAMES for this specific
# sub-model's importances aren't cheaply recoverable here, so that
# diagnostic table is skipped (stage 1's FI table above is unaffected,
# since stage 1 trains on the full un-subset feature set).

joblib.dump(
    {
        "stage1_model": m_final["stage1"], "stage1_imputer": m_final["imputer1"],
        "stage2a_conifer_model": m_final["stage2a"], "stage2a_imputer": m_final["imputer2a"],
        "stage2bi_lv_vs_rest_model": m_final["stage2bi"], "stage2bi_imputer": m_final["imputer2bi"],
        "stage2bii_hb_ks_lm_model": m_final["stage2bii"], "stage2bii_imputer": m_final["imputer2bii"],
        "feature_cols": valid_cols_trainval,
        "id_col": ID_COL, "target_col": TARGET_COL,
        "group_map": GROUP_MAP,
        "sampling_strategy_stage1": SAMPLING_STRATEGY_STAGE1,
        "sampling_strategy_species": SAMPLING_STRATEGY,
        "test_metrics": test_metrics,
        "stage1_test_metrics": stage1_test_metrics,
        "stage2bi_test_metrics": stage2bi_test_metrics,
        "removed_clearcut_overlap_file": OVERLAP_FILE,
        "experiment": "EXP038_xgboost_hierarchical"
    },
    MODEL_OUT
)

print("\nDone.")
print("Model:", MODEL_OUT)
print("Confusion matrix (combined):", CM_OUT)
print("Confusion matrix (stage 1):", CM_STAGE1_OUT)
print("Confusion matrix (stage 2b-i):", CM_STAGE2BI_OUT)
