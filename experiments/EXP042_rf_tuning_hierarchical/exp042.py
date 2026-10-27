"""EXP042: hyperparameter tuning for Random Forest inside EXP032's exact
4-stage hierarchy - the RF-equivalent of EXP041's XGBoost tuning attempt.
Answers a real, previously-unfilled gap: EXP032's own RF hyperparameters
(n_estimators=400, max_depth=None, min_samples_leaf=2, class_weight=
"balanced") were carried forward unchanged from early in the project and
never actually tuned themselves, even as the features/architecture around
them kept changing. The one prior RF grid search in this repo
(`EXP019_auto_rf_search/autoRF.py`) ran on a completely different, now-
superseded flat pipeline (old elev_*/soil_main columns, no 10-day grid, no
species-signature meta-features) and its results were never logged - not
a usable reference point for the current architecture.

Same grid-search style/effort as EXP041 (hand-picked configs evaluated on
VAL, not exhaustive/CV) - reuses EXP019's original 8 RF parameter sets
verbatim (still a reasonable spread of tree count/depth/leaf-size/
max_features/class_weight choices) plus EXP032's exact current
hyperparameters as param set 1, a control. All 4 stages share one param
set per grid entry, selected by VAL macro F1 of the combined hierarchical
prediction, exactly like EXP041.

Compare against: EXP032 (untuned/carried-forward RF hierarchy, current
primary) Accuracy 0.8219, Macro F1 0.6246, Kappa 0.7534, HB F1 0.5161.
Also worth watching against EXP041's cautionary result (XGBoost tuning
made TEST worse despite winning on VAL, likely VAL-set overfitting) - the
same risk applies here.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

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
# FILES (identical to EXP032/038/040/041)
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

MODEL_OUT = os.path.join(PATHS.models_trained, "EXP042_rf_tuned_hierarchical.joblib")
SEARCH_RESULTS_OUT = os.path.join(PATHS.outputs_tables, "EXP042_search_results.csv")
FI_OUT_STAGE1 = os.path.join(PATHS.outputs_tables, "EXP042_feature_importance_stage1_group.csv")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "EXP042_test_predictions.csv")
CM_OUT = os.path.join(PATHS.outputs_figures, "EXP042_confusion_matrix_test_normalized.png")

# =========================================================
# CONFIG (identical to EXP032/038/040/041)
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
REST_OF_DECIDUOUS_SPECIES = {"HB", "KS", "LM"}

SAMPLING_STRATEGY_STAGE1 = {"MUU": 500}
GROUP_COL = "target_group"
LV_VS_REST_COL = "target_lv_vs_rest"

RANDOM_STATE = 42

# Param set 1 = EXP032's exact current hyperparameters (control). Sets
# 2-9 = EXP019's original 8-set RF grid, reused verbatim.
PARAM_GRID = [
    dict(n_estimators=400, max_depth=None, min_samples_leaf=2, min_samples_split=2,
         max_features="sqrt", class_weight="balanced"),  # = EXP032 baseline
    dict(n_estimators=400, max_depth=None, min_samples_leaf=1, min_samples_split=2,
         max_features="sqrt", class_weight="balanced"),
    dict(n_estimators=700, max_depth=None, min_samples_leaf=1, min_samples_split=2,
         max_features="sqrt", class_weight="balanced_subsample"),
    dict(n_estimators=700, max_depth=None, min_samples_leaf=2, min_samples_split=2,
         max_features="sqrt", class_weight="balanced"),
    dict(n_estimators=1000, max_depth=None, min_samples_leaf=2, min_samples_split=5,
         max_features="sqrt", class_weight="balanced_subsample"),
    dict(n_estimators=700, max_depth=40, min_samples_leaf=1, min_samples_split=2,
         max_features="log2", class_weight="balanced"),
    dict(n_estimators=1000, max_depth=60, min_samples_leaf=1, min_samples_split=5,
         max_features=0.3, class_weight="balanced_subsample"),
    dict(n_estimators=700, max_depth=30, min_samples_leaf=2, min_samples_split=5,
         max_features=0.5, class_weight="balanced"),
    dict(n_estimators=1000, max_depth=None, min_samples_leaf=4, min_samples_split=10,
         max_features="sqrt", class_weight="balanced_subsample"),
]

# =========================================================
# HELPER FUNCTIONS (species-signature features, identical to EXP032/038/040/041)
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


def fit_rf_stage(X_os_imp, y_os_str, rf_kwargs):
    model = RandomForestClassifier(**rf_kwargs, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_os_imp, y_os_str)
    return model


def train_hierarchical(X_train, y_train, sampling_strategy_stage1, sampling_strategy_species, rf_kwargs):
    y_group = y_train.map(GROUP_MAP)
    assert y_group.isna().sum() == 0, "Unmapped species found in GROUP_MAP"

    X_group_os, y_group_os = manual_oversample(
        X_train, y_group, sampling_strategy_stage1, GROUP_COL, random_state=42
    )
    imputer1 = SimpleImputer(strategy="median")
    X_group_os_imp = imputer1.fit_transform(X_group_os)
    stage1 = fit_rf_stage(X_group_os_imp, y_group_os, rf_kwargs)

    conifer_mask = y_train.isin(CONIFER_SPECIES)
    X_conifer_os, y_conifer_os = manual_oversample(
        X_train[conifer_mask], y_train[conifer_mask], sampling_strategy_species, TARGET_COL, random_state=42
    )
    imputer2a = SimpleImputer(strategy="median")
    X_conifer_os_imp = imputer2a.fit_transform(X_conifer_os)
    stage2a = fit_rf_stage(X_conifer_os_imp, y_conifer_os, rf_kwargs)

    deciduous_mask = y_train.isin(DECIDUOUS_SPECIES)
    y_lv_vs_rest = y_train[deciduous_mask].apply(lambda s: "LV" if s == "LV" else "REST")
    sampling_strategy_lv = {"LV": sampling_strategy_species.get("LV", 2000)}
    X_lv_os, y_lv_os = manual_oversample(
        X_train[deciduous_mask], y_lv_vs_rest, sampling_strategy_lv, LV_VS_REST_COL, random_state=42
    )
    imputer2bi = SimpleImputer(strategy="median")
    X_lv_os_imp = imputer2bi.fit_transform(X_lv_os)
    stage2bi = fit_rf_stage(X_lv_os_imp, y_lv_os, rf_kwargs)

    rest_mask = y_train.isin(REST_OF_DECIDUOUS_SPECIES)
    X_rest_os, y_rest_os = manual_oversample(
        X_train[rest_mask], y_train[rest_mask], sampling_strategy_species, TARGET_COL, random_state=42
    )
    imputer2bii = SimpleImputer(strategy="median")
    X_rest_os_imp = imputer2bii.fit_transform(X_rest_os)
    stage2bii = fit_rf_stage(X_rest_os_imp, y_rest_os, rf_kwargs)

    return {
        "stage1": stage1, "imputer1": imputer1,
        "stage2a": stage2a, "imputer2a": imputer2a,
        "stage2bi": stage2bi, "imputer2bi": imputer2bi,
        "stage2bii": stage2bii, "imputer2bii": imputer2bii,
    }


def predict_hierarchical(X, m):
    X_imp1 = m["imputer1"].transform(X)
    group_pred = m["stage1"].predict(X_imp1)

    final_pred = np.empty(len(X), dtype=object)
    lv_vs_rest_pred = np.full(len(X), np.nan, dtype=object)

    conifer_rows = group_pred == "CONIFER"
    if conifer_rows.any():
        X_conifer_imp = m["imputer2a"].transform(X.loc[conifer_rows])
        final_pred[conifer_rows] = m["stage2a"].predict(X_conifer_imp)

    deciduous_rows = group_pred == "DECIDUOUS"
    if deciduous_rows.any():
        X_deciduous_imp = m["imputer2bi"].transform(X.loc[deciduous_rows])
        lv_pred = m["stage2bi"].predict(X_deciduous_imp)
        lv_vs_rest_pred[deciduous_rows] = lv_pred

        is_lv = lv_pred == "LV"
        deciduous_idx = np.where(deciduous_rows)[0]
        lv_idx = deciduous_idx[is_lv]
        rest_idx = deciduous_idx[~is_lv]

        final_pred[lv_idx] = "LV"
        if len(rest_idx) > 0:
            X_rest_imp = m["imputer2bii"].transform(X.iloc[rest_idx])
            final_pred[rest_idx] = m["stage2bii"].predict(X_rest_imp)

    muu_rows = group_pred == "MUU"
    final_pred[muu_rows] = "MUU"

    return final_pred, group_pred, lv_vs_rest_pred


# =========================================================
# 1-4. LOAD, CLEAN, PIVOT, AUX (identical to EXP032/038/040/041)
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
# 5. GRID SEARCH: TRAIN-only hierarchy -> VAL macro F1 (combined)
# =========================================================

results = []
best = None

for i, params in enumerate(PARAM_GRID, start=1):
    print("\n" + "=" * 60)
    print(f"Param set {i}/{len(PARAM_GRID)}: {params}")
    print("=" * 60)

    t0 = time.time()
    m = train_hierarchical(X_train, y_train, SAMPLING_STRATEGY_STAGE1, SAMPLING_STRATEGY, params)
    val_final_pred, val_group_pred, val_lv_pred = predict_hierarchical(X_val, m)
    elapsed = time.time() - t0

    val_macro_f1 = f1_score(y_val, val_final_pred, average="macro")
    val_hb_f1 = f1_score(y_val, val_final_pred, labels=["HB"], average="macro", zero_division=0)

    print(f"VAL macro F1={val_macro_f1:.4f}, HB F1={val_hb_f1:.4f}, elapsed={elapsed:.1f}s")

    row = {"param_i": i, "val_macro_f1": val_macro_f1, "val_hb_f1": val_hb_f1, "elapsed_s": elapsed, **params}
    results.append(row)

    if best is None or val_macro_f1 > best["val_macro_f1"]:
        best = {"param_i": i, "val_macro_f1": val_macro_f1, "params": params}

    pd.DataFrame(results).to_csv(SEARCH_RESULTS_OUT, index=False, encoding="utf-8-sig")

results_df = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
results_df.to_csv(SEARCH_RESULTS_OUT, index=False, encoding="utf-8-sig")

print("\n================ GRID SEARCH RESULTS (sorted) ================")
print(results_df)

best_params = best["params"]
print("\nBest param set:", best["param_i"], json.dumps(best_params, indent=2))
print("Best VAL macro F1:", best["val_macro_f1"])

# =========================================================
# 6. FINAL MODEL: TRAIN+VAL -> TEST, using best param set
# =========================================================

print("\n" + "=" * 60)
print("TRAIN+VAL hierarchy (best params) -> TEST")
print("=" * 60)

X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()
X_trainval = X_trainval[valid_cols_trainval].copy()
X_test_final = X_test[valid_cols_trainval].copy()

m_final = train_hierarchical(X_trainval, y_trainval, SAMPLING_STRATEGY_STAGE1, SAMPLING_STRATEGY, best_params)
test_final_pred, test_group_pred, test_lv_pred = predict_hierarchical(X_test_final, m_final)

test_group_true = y_test.map(GROUP_MAP)
stage1_test_metrics = evaluate_model("EXP042 STAGE 1 (group) - TEST", test_group_true, test_group_pred)

deciduous_test_mask = y_test.isin(DECIDUOUS_SPECIES).values
reached_stage2bi_mask = deciduous_test_mask & (test_group_pred == "DECIDUOUS")
y_lv_true_test = y_test[reached_stage2bi_mask].apply(lambda s: "LV" if s == "LV" else "REST")
y_lv_pred_test = pd.Series(test_lv_pred[reached_stage2bi_mask])
stage2bi_test_metrics = evaluate_model(
    "EXP042 STAGE 2b-i (LV vs REST, true-deciduous test rows) - TEST", y_lv_true_test, y_lv_pred_test
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

test_metrics = evaluate_model("EXP042 hierarchical RF TUNED (combined) - TEST", y_test, test_final_pred)
plot_confusion_matrix_pretty(
    y_test, test_final_pred, "EXP042 test CM (tuned RF hierarchy)",
    CM_OUT, normalize=True
)
test_metrics["hb_precision"] = precision_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_recall"] = recall_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
test_metrics["hb_f1"] = f1_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
print(f"\nHB precision={test_metrics['hb_precision']:.4f}, "
      f"HB recall={test_metrics['hb_recall']:.4f}, HB F1={test_metrics['hb_f1']:.4f}")

importance_df_stage1 = pd.DataFrame({
    "feature": valid_cols_trainval, "importance": m_final["stage1"].feature_importances_
}).sort_values("importance", ascending=False)
importance_df_stage1.to_csv(FI_OUT_STAGE1, index=False, encoding="utf-8-sig")

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
        "best_params": best_params,
        "search_results_file": SEARCH_RESULTS_OUT,
        "test_metrics": test_metrics,
        "stage1_test_metrics": stage1_test_metrics,
        "stage2bi_test_metrics": stage2bi_test_metrics,
        "removed_clearcut_overlap_file": OVERLAP_FILE,
        "experiment": "EXP042_rf_tuning_hierarchical"
    },
    MODEL_OUT
)

print("\nDone.")
print("Search results:", SEARCH_RESULTS_OUT)
print("Model:", MODEL_OUT)
print("Confusion matrix:", CM_OUT)
