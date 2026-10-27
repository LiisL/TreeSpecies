"""EXP048: Random Forest, EXP032's exact 4-stage hierarchy and
species-signature meta-features, trained on a NEW, LARGER dataset that
combines PURE (>=80% purity) AND MIXED stands together - every prior
experiment (EXP001-042/044) trained on pure stands only. Isolates
"does adding mixed-stand examples to training help or hurt dominant-
species classification" as the one new variable; everything else
(architecture, features, oversampling policy) is unchanged from EXP032.

Data lineage (see experiments/EXP047_composition_daily_statistical and
[[project-treespecies-exp047-composition-classification]] for the full
story):
  1. EXP047 downloaded DAILY Statistical-API stats (not the old 10-day-
     grid download) for composition_base_2026_subregion2.gpkg's 10,309
     parcels (4,198 pure + 6,111 mixed) - a fresh dataset, not the old
     Subregion2_{train,val,test}_sentinel2_daily_timeseries.csv (only
     ~39% of that old dataset's pure-stand ids survive into this new
     export; re-download was the explicit user decision).
  2. This script aggregates those daily rows into the SAME 26 fixed
     10-day windows used throughout the rest of the project (via
     src.preprocessing.timeseries.aggregate_daily_to_windows), so the
     existing add_species_signature_features formulas (which hardcode
     specific window-start dates like "NDVI_mean_2025_05_01") keep
     working unmodified - a raw daily table would essentially never
     contain those exact dates.
  3. Height/soil AUX features come directly from
     composition_base_2026_subregion2.gpkg's own elev_*/soil_main
     columns (already computed by scripts/build_composition_base_dataset.py
     from the same national rasters used everywhere else) - renamed to
     height_*/soil_majority to match the existing merge_aux()/
     add_species_signature_features() column-naming convention.

Known limitation, not solved here: clearcut-overlap filtering
(remove_clearcut_ids) only catches whatever fraction of this dataset's
ids happen to also appear in the OLD clearcut-overlap CSV (built against
the old parcel set) - it is NOT a fresh clearcut check against the new
10,309 parcels. Treated as a partial safeguard, not full coverage.

Compare against: EXP032 (pure-only, primary, Macro F1 0.6246), EXP038
(pure-only XGBoost, Macro F1 0.6916), EXP044 (pixel-level RF on pure+mixed,
parcel-aggregated Macro F1 0.4727).
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

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
from src.models.sampling import manual_oversample
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix_pretty
from src.paths import PATHS

# =========================================================
# FILES
# =========================================================

DAILY_DIR = PATHS.intermediate / "EXP047_composition_daily"
TRAIN_FILE = DAILY_DIR / "composition_train_sentinel2_daily_timeseries.csv"
VAL_FILE = DAILY_DIR / "composition_val_sentinel2_daily_timeseries.csv"
TEST_FILE = DAILY_DIR / "composition_test_sentinel2_daily_timeseries.csv"

PARCEL_FILE = PATHS.raw / "composition_base_2026_subregion2.gpkg"  # source of height/soil AUX + composition cols

OVERLAP_FILE = os.path.join(
    PATHS.reference, "clearcut_overlap", "stands_overlapping_clearcuts_over20pct.csv"
)

MODEL_OUT = os.path.join(PATHS.models_trained, "EXP048_rf_composition_pure_mixed.joblib")
FI_OUT_STAGE1 = os.path.join(PATHS.outputs_tables, "EXP048_feature_importance_stage1_group.csv")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "EXP048_test_predictions.csv")
CM_OUT = os.path.join(PATHS.outputs_figures, "EXP048_confusion_matrix_test_normalized.png")
CM_STAGE1_OUT = os.path.join(PATHS.outputs_figures, "EXP048_stage1_group_confusion_matrix.png")
CM_STAGE2BI_OUT = os.path.join(PATHS.outputs_figures, "EXP048_stage2bi_lv_vs_rest_confusion_matrix.png")

# =========================================================
# CONFIG
# =========================================================

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

# same 26 fixed 10-day windows as EXP029/032/038/043 (note the real gap
# between 2025-03-12 and 2025-04-01 - not a bug, see aggregate_daily_to_windows)
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

SAMPLING_STRATEGY = {"HB": 1200, "LM": 1200, "KU": 1600, "LV": 2000, "MUU": 500}

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

RF_KWARGS = dict(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)

# =========================================================
# HELPER FUNCTIONS (species-signature features, byte-identical to EXP032/038/043)
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
    """Builds an AUX dataframe (height_*/soil_majority) from
    composition_base_2026_subregion2.gpkg's own elev_*/soil_main columns
    (computed by scripts/build_composition_base_dataset.py from the same
    national height/soil rasters used everywhere else), renamed to match
    merge_aux()'s expected column names.
    """
    import geopandas as gpd
    gdf = gpd.read_file(PARCEL_FILE, ignore_geometry=True)
    aux = gdf[[ID_COL, "elev_mean", "elev_median", "elev_std", "elev_min", "elev_max", "soil_main"]].copy()
    aux = aux.rename(columns={
        "elev_mean": "height_mean", "elev_median": "height_median", "elev_std": "height_std",
        "elev_min": "height_min", "elev_max": "height_max", "soil_main": "soil_majority",
    })
    return aux


def train_hierarchical(X_train, y_train, sampling_strategy_stage1, sampling_strategy_species, rf_kwargs):
    y_group = y_train.map(GROUP_MAP)
    assert y_group.isna().sum() == 0, "Unmapped species found in GROUP_MAP"

    X_group_os, y_group_os = manual_oversample(X_train, y_group, sampling_strategy_stage1, GROUP_COL, random_state=42)
    imputer1 = SimpleImputer(strategy="median")
    X_group_os_imp = imputer1.fit_transform(X_group_os)
    stage1 = RandomForestClassifier(**rf_kwargs)
    stage1.fit(X_group_os_imp, y_group_os)

    conifer_mask = y_train.isin(CONIFER_SPECIES)
    X_conifer_os, y_conifer_os = manual_oversample(X_train[conifer_mask], y_train[conifer_mask], sampling_strategy_species, TARGET_COL, random_state=42)
    imputer2a = SimpleImputer(strategy="median")
    X_conifer_os_imp = imputer2a.fit_transform(X_conifer_os)
    stage2a = RandomForestClassifier(**rf_kwargs)
    stage2a.fit(X_conifer_os_imp, y_conifer_os)

    deciduous_mask = y_train.isin(DECIDUOUS_SPECIES)
    y_lv_vs_rest = y_train[deciduous_mask].apply(lambda s: "LV" if s == "LV" else "REST")
    sampling_strategy_lv = {"LV": sampling_strategy_species.get("LV", 2000)}
    X_lv_os, y_lv_os = manual_oversample(X_train[deciduous_mask], y_lv_vs_rest, sampling_strategy_lv, LV_VS_REST_COL, random_state=42)
    imputer2bi = SimpleImputer(strategy="median")
    X_lv_os_imp = imputer2bi.fit_transform(X_lv_os)
    stage2bi = RandomForestClassifier(**rf_kwargs)
    stage2bi.fit(X_lv_os_imp, y_lv_os)

    rest_mask = y_train.isin(REST_OF_DECIDUOUS_SPECIES)
    X_rest_os, y_rest_os = manual_oversample(X_train[rest_mask], y_train[rest_mask], sampling_strategy_species, TARGET_COL, random_state=42)
    imputer2bii = SimpleImputer(strategy="median")
    X_rest_os_imp = imputer2bii.fit_transform(X_rest_os)
    stage2bii = RandomForestClassifier(**rf_kwargs)
    stage2bii.fit(X_rest_os_imp, y_rest_os)

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


def main():
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

    train_wide = add_species_signature_features(train_wide, "train")
    val_wide = add_species_signature_features(val_wide, "val")
    test_wide = add_species_signature_features(test_wide, "test")

    train_wide, val_wide, test_wide = encode_categoricals_consistently(train_wide, val_wide, test_wide, AUX_CATEGORICAL_COLS)
    train_wide, val_wide, test_wide, feature_cols = align_columns(train_wide, val_wide, test_wide, ID_COL, TARGET_COL)

    X_train = train_wide[feature_cols].copy()
    y_train = train_wide[TARGET_COL].copy()
    X_val = val_wide[feature_cols].copy()
    y_val = val_wide[TARGET_COL].copy()
    X_test = test_wide[feature_cols].copy()
    y_test = test_wide[TARGET_COL].copy()

    X_train, X_val, X_test, feature_cols_used = drop_all_nan_train_columns(X_train, X_val, X_test)

    print("\nSpecies distribution (train):")
    print(y_train.value_counts())

    print("\n" + "=" * 60)
    print("TRAIN-only hierarchy -> VALIDATION")
    print("=" * 60)

    m_trainonly = train_hierarchical(X_train, y_train, SAMPLING_STRATEGY_STAGE1, SAMPLING_STRATEGY, RF_KWARGS)
    val_final_pred, val_group_pred, val_lv_pred = predict_hierarchical(X_val, m_trainonly)

    evaluate_model("EXP048 STAGE 1 (group) - VALIDATION", y_val.map(GROUP_MAP), val_group_pred)
    evaluate_model("EXP048 hierarchical (combined) - VALIDATION", y_val, val_final_pred)

    print("\n" + "=" * 60)
    print("TRAIN+VAL hierarchy -> TEST")
    print("=" * 60)

    X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()
    X_trainval = X_trainval[valid_cols_trainval].copy()
    X_test_final = X_test[valid_cols_trainval].copy()

    m_final = train_hierarchical(X_trainval, y_trainval, SAMPLING_STRATEGY_STAGE1, SAMPLING_STRATEGY, RF_KWARGS)
    test_final_pred, test_group_pred, test_lv_pred = predict_hierarchical(X_test_final, m_final)

    test_group_true = y_test.map(GROUP_MAP)
    stage1_test_metrics = evaluate_model("EXP048 STAGE 1 (group) - TEST", test_group_true, test_group_pred)
    plot_confusion_matrix_pretty(test_group_true, test_group_pred, "EXP048 stage-1 group confusion matrix (test)", CM_STAGE1_OUT, normalize=True)

    deciduous_test_mask = y_test.isin(DECIDUOUS_SPECIES).values
    reached_stage2bi_mask = deciduous_test_mask & (test_group_pred == "DECIDUOUS")
    y_lv_true_test = y_test[reached_stage2bi_mask].apply(lambda s: "LV" if s == "LV" else "REST")
    y_lv_pred_test = pd.Series(test_lv_pred[reached_stage2bi_mask])
    stage2bi_test_metrics = evaluate_model("EXP048 STAGE 2b-i (LV vs REST, true-deciduous test rows) - TEST", y_lv_true_test, y_lv_pred_test)
    plot_confusion_matrix_pretty(y_lv_true_test, y_lv_pred_test, "EXP048 stage-2b-i LV-vs-rest confusion matrix (test)", CM_STAGE2BI_OUT, normalize=True)

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

    test_metrics = evaluate_model("EXP048 hierarchical (combined) - TEST", y_test, test_final_pred)
    plot_confusion_matrix_pretty(y_test, test_final_pred, "EXP048 test CM (pure+mixed stands)", CM_OUT, normalize=True)
    test_metrics["hb_precision"] = precision_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
    test_metrics["hb_recall"] = recall_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
    test_metrics["hb_f1"] = f1_score(y_test, test_final_pred, labels=["HB"], average="macro", zero_division=0)
    print(f"\nHB precision={test_metrics['hb_precision']:.4f}, HB recall={test_metrics['hb_recall']:.4f}, HB F1={test_metrics['hb_f1']:.4f}")

    importance_df_stage1 = pd.DataFrame({
        "feature": valid_cols_trainval, "importance": m_final["stage1"].feature_importances_
    }).sort_values("importance", ascending=False)
    importance_df_stage1.to_csv(FI_OUT_STAGE1, index=False, encoding="utf-8-sig")

    joblib.dump({
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
        "experiment": "EXP048_composition_rf",
    }, MODEL_OUT)

    print("\nDone.")
    print("Model:", MODEL_OUT)
    print("Compare against: EXP032=0.6246, EXP038=0.6916 (pure-only), EXP044=0.4727 (pixel-level, pure+mixed)")


if __name__ == "__main__":
    main()
