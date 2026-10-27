import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

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
from src.evaluation.metrics import evaluate_model
from src.paths import PATHS

# =========================================================
# FILES
# =========================================================

DATA_DIR = PATHS.intermediate

TRAIN_FILE = os.path.join(DATA_DIR, "Subregion2_train_sentinel2_10day_timeseries.csv")
VAL_FILE   = os.path.join(DATA_DIR, "Subregion2_val_sentinel2_10day_timeseries.csv")
TEST_FILE  = os.path.join(DATA_DIR, "Subregion2_test_sentinel2_10day_timeseries.csv")

AUX_FILE = os.path.join(DATA_DIR, "stand_aux_features.csv")

OVERLAP_FILE = os.path.join(
    PATHS.reference,
    "clearcut_overlap",
    "stands_overlapping_clearcuts_over20pct.csv"
)

MODEL_OUT = os.path.join(PATHS.models_trained, "EXP021_rf_species_signature_features.joblib")
FI_OUT = os.path.join(PATHS.outputs_tables, "EXP021_feature_importance.csv")
TEST_PRED_OUT = os.path.join(PATHS.outputs_tables, "EXP021_test_predictions.csv")

# =========================================================
# CONFIG
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
    "elev_mean",
    "elev_median",
    "elev_std",
    "elev_min",
    "elev_max"
]

AUX_CATEGORICAL_COLS = ["soil_main"]

SAMPLING_STRATEGY = {
    "HB": 1200,
    "LM": 1200,
    "KU": 1600,
    "LV": 2000,
    "MUU": 500
}

# =========================================================
# HELPER FUNCTIONS
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

    # -----------------------------------------------------
    # HB - aspen: height + spring + autumn
    # -----------------------------------------------------

    if "elev_max" in df.columns and "elev_std" in df.columns:
        df["HB_height_index"] = df["elev_max"] + df["elev_std"]
        df["HB_height_product"] = df["elev_max"] * df["elev_std"]
    else:
        df["HB_height_index"] = np.nan
        df["HB_height_product"] = np.nan

    if all(c in df.columns for c in ["elev_std", "elev_mean"]):
        df["HB_structure_ratio"] = df["elev_std"] / (df["elev_mean"] + 1)
    else:
        df["HB_structure_ratio"] = np.nan

    df["HB_spring_index"] = safe_mean(
        df,
        [
            "NDVI_mean_2025_05_01",
            "NDMI_mean_2025_05_01",
            "NBR_mean_2025_05_01"
        ]
    )

    df["HB_autumn_ndre"] = safe_mean(
        df,
        [
            "NDRE_mean_2025_09_28",
            "NDRE_mean_2025_10_08",
            "NDRE_mean_2025_10_18"
        ]
    )

    # -----------------------------------------------------
    # KS - birch: moisture + SWIR + NDRE September
    # -----------------------------------------------------

    df["KS_moisture_index"] = safe_mean(
        df,
        [
            "NDMI_mean_2025_09_28",
            "NDRE_mean_2025_09_28",
            "B11_mean_2025_09_28"
        ]
    )

    # -----------------------------------------------------
    # KU - spruce: winter/autumn signal
    # -----------------------------------------------------

    df["KU_winter_index"] = safe_mean(
        df,
        [
            "B11_mean_2025_02_20",
            "B12_mean_2025_02_20",
            "NDVI_mean_2025_02_20"
        ]
    )

    df["KU_autumn_index"] = safe_mean(
        df,
        [
            "NDRE_mean_2025_09_28",
            "NDVI_mean_2025_10_18",
            "NDMI_mean_2025_09_28"
        ]
    )

    # -----------------------------------------------------
    # LM - black alder: autumn phenology
    # -----------------------------------------------------

    df["LM_autumn_index"] = safe_mean(
        df,
        [
            "NDRE_mean_2025_09_28",
            "NDVI_mean_2025_09_28",
            "B8_mean_2025_09_28",
            "B7_mean_2025_09_28"
        ]
    )

    # -----------------------------------------------------
    # LV - grey alder: July NIR/red-edge/SWIR
    # -----------------------------------------------------

    df["LV_july_index"] = safe_mean(
        df,
        [
            "B6_mean_2025_07_20",
            "B7_mean_2025_07_20",
            "B8_mean_2025_07_20",
            "B8A_mean_2025_07_20"
        ]
    )

    df["LV_july_swir_index"] = safe_mean(
        df,
        [
            "B11_mean_2025_07_20",
            "B12_mean_2025_07_20"
        ]
    )

    # -----------------------------------------------------
    # MA - pine: spring NIR
    # -----------------------------------------------------

    df["MA_spring_nir"] = safe_mean(
        df,
        [
            "B8_mean_2025_05_01",
            "B8A_mean_2025_05_01",
            "B6_mean_2025_05_01",
            "NDVI_mean_2025_05_01"
        ]
    )

    # -----------------------------------------------------
    # Phenological change features
    # -----------------------------------------------------

    df["NDVI_change_spring"] = safe_diff(
        df,
        "NDVI_mean_2025_05_21",
        "NDVI_mean_2025_04_11"
    )

    df["NDVI_change_summer"] = safe_diff(
        df,
        "NDVI_mean_2025_07_20",
        "NDVI_mean_2025_05_21"
    )

    df["NDVI_change_autumn"] = safe_diff(
        df,
        "NDVI_mean_2025_10_18",
        "NDVI_mean_2025_09_08"
    )

    df["NDRE_change_autumn"] = safe_diff(
        df,
        "NDRE_mean_2025_10_18",
        "NDRE_mean_2025_09_08"
    )

    df["NDMI_change_autumn"] = safe_diff(
        df,
        "NDMI_mean_2025_10_18",
        "NDMI_mean_2025_09_08"
    )

    df["NBR_change_autumn"] = safe_diff(
        df,
        "NBR_mean_2025_10_18",
        "NBR_mean_2025_09_08"
    )

    new_cols = [
        "HB_height_index",
        "HB_height_product",
        "HB_structure_ratio",
        "HB_spring_index",
        "HB_autumn_ndre",
        "KS_moisture_index",
        "KU_winter_index",
        "KU_autumn_index",
        "LM_autumn_index",
        "LV_july_index",
        "LV_july_swir_index",
        "MA_spring_nir",
        "NDVI_change_spring",
        "NDVI_change_summer",
        "NDVI_change_autumn",
        "NDRE_change_autumn",
        "NDMI_change_autumn",
        "NBR_change_autumn"
    ]

    print("Meta-features added:", len(new_cols))
    print("Meta-feature NaN counts:")
    print(df[new_cols].isna().sum())

    return df


# =========================================================
# 1. LOAD DATA
# =========================================================

train_long = pd.read_csv(TRAIN_FILE)
val_long = pd.read_csv(VAL_FILE)
test_long = pd.read_csv(TEST_FILE)
aux = pd.read_csv(AUX_FILE)

print("Long-form read before cleaning:")
print("train:", train_long.shape)
print("val:", val_long.shape)
print("test:", test_long.shape)
print("aux:", aux.shape)

# =========================================================
# 2. REMOVE CLEARCUT-OVERLAPPING IDS
# =========================================================

train_long, val_long, test_long = remove_clearcut_ids(
    train_long,
    val_long,
    test_long,
    OVERLAP_FILE,
    ID_COL,
    TARGET_COL
)

# =========================================================
# 3. LONG -> WIDE
# =========================================================

train_wide = long_to_wide(train_long, "train", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
val_wide = long_to_wide(val_long, "val", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
test_wide = long_to_wide(test_long, "test", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)

print("\nWide-form before AUX:")
print("train:", train_wide.shape)
print("val:", val_wide.shape)
print("test:", test_wide.shape)

# =========================================================
# 4. AUX
# =========================================================

train_wide = merge_aux(train_wide, aux, "train", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
val_wide = merge_aux(val_wide, aux, "val", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
test_wide = merge_aux(test_wide, aux, "test", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)

# =========================================================
# 5. ADD SPECIES-SPECIFIC META-FEATURES
# =========================================================

train_wide = add_species_signature_features(train_wide, "train")
val_wide = add_species_signature_features(val_wide, "val")
test_wide = add_species_signature_features(test_wide, "test")

# =========================================================
# 6. SOIL ONE-HOT
# =========================================================

train_wide, val_wide, test_wide = encode_categoricals_consistently(
    train_wide,
    val_wide,
    test_wide,
    AUX_CATEGORICAL_COLS
)

# =========================================================
# 7. ALIGN COLUMNS
# =========================================================

train_wide, val_wide, test_wide, feature_cols = align_columns(
    train_wide,
    val_wide,
    test_wide,
    ID_COL,
    TARGET_COL
)

print("\nWide-form + AUX + meta-features:")
print("train:", train_wide.shape)
print("val:", val_wide.shape)
print("test:", test_wide.shape)

# =========================================================
# 8. X / y
# =========================================================

X_train = train_wide[feature_cols].copy()
y_train = train_wide[TARGET_COL].copy()

X_val = val_wide[feature_cols].copy()
y_val = val_wide[TARGET_COL].copy()

X_test = test_wide[feature_cols].copy()
y_test = test_wide[TARGET_COL].copy()

X_train, X_val, X_test, feature_cols_used = drop_all_nan_train_columns(
    X_train,
    X_val,
    X_test
)

print("\nFeature count after removing empty columns:")
print("train:", X_train.shape[1])
print("val:", X_val.shape[1])
print("test:", X_test.shape[1])

print("\nClass counts before oversampling:")
print(y_train.value_counts())

# =========================================================
# 9. OVERSAMPLING ON TRAIN
# =========================================================

print("\nPerforming manual oversampling on TRAIN data...")

X_train_os, y_train_os = manual_oversample(
    X_train,
    y_train,
    SAMPLING_STRATEGY,
    TARGET_COL,
    random_state=42
)

print("\nClass counts after oversampling:")
print(y_train_os.value_counts())

# =========================================================
# 10. IMPUTE + TRAIN -> VAL
# =========================================================

imputer = SimpleImputer(strategy="median")

X_train_os_imp = imputer.fit_transform(X_train_os)
X_val_imp = imputer.transform(X_val)

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining EXP021 model on TRAIN...")
rf.fit(X_train_os_imp, y_train_os)

val_pred = rf.predict(X_val_imp)

val_metrics = evaluate_model(
    "EXP021 VALIDATION - species signature features",
    y_val,
    val_pred
)

# =========================================================
# 11. TRAIN + VAL -> TEST
# =========================================================

X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()

X_trainval = X_trainval[valid_cols_trainval].copy()
X_test_final = X_test[valid_cols_trainval].copy()

print("\nPerforming manual oversampling on TRAIN+VAL data...")

X_trainval_os, y_trainval_os = manual_oversample(
    X_trainval,
    y_trainval,
    SAMPLING_STRATEGY,
    TARGET_COL,
    random_state=42
)

print("\nTRAIN+VAL class counts after oversampling:")
print(y_trainval_os.value_counts())

imputer_final = SimpleImputer(strategy="median")

X_trainval_os_imp = imputer_final.fit_transform(X_trainval_os)
X_test_imp = imputer_final.transform(X_test_final)

rf_final = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining EXP021 final model on TRAIN+VAL...")
rf_final.fit(X_trainval_os_imp, y_trainval_os)

test_pred = rf_final.predict(X_test_imp)

test_predictions_df = pd.DataFrame({
    ID_COL: test_wide[ID_COL].values,
    "y_true": y_test.values,
    "y_pred": test_pred
})
test_predictions_df.to_csv(TEST_PRED_OUT, index=False, encoding="utf-8-sig")
print("\nTest predictions saved:", TEST_PRED_OUT)

test_metrics = evaluate_model(
    "EXP021 TEST - species signature features",
    y_test,
    test_pred
)

# =========================================================
# 12. FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "feature": valid_cols_trainval,
    "importance": rf_final.feature_importances_
}).sort_values("importance", ascending=False)

print("\nEXP021 Top 80 feature importance:")
print(importance_df.head(80))

importance_df.to_csv(
    FI_OUT,
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 13. SAVE MODEL
# =========================================================

joblib.dump(
    {
        "model": rf_final,
        "imputer": imputer_final,
        "feature_cols": valid_cols_trainval,
        "id_col": ID_COL,
        "target_col": TARGET_COL,
        "timeseries_features": TS_FEATURE_COLS,
        "aux_numeric_cols": AUX_NUMERIC_COLS,
        "aux_categorical_cols": AUX_CATEGORICAL_COLS,
        "sampling_strategy": SAMPLING_STRATEGY,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "removed_clearcut_overlap_file": OVERLAP_FILE,
        "experiment": "EXP021_species_signature_features"
    },
    MODEL_OUT
)

print("\nDone.")
print("Feature importance:", FI_OUT)
print("Model:", MODEL_OUT)
