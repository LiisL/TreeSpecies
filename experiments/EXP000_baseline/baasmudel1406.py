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
    get_timeseries_feature_cols,
    long_to_wide,
    merge_aux,
    encode_categoricals_consistently,
    align_columns,
    drop_all_nan_train_columns,
)
from src.evaluation.metrics import evaluate_model, plot_confusion_matrix_pretty
from src.paths import PATHS

# =========================================================
# FILES
# =========================================================

TRAIN_FILE = os.path.join(PATHS.processed, "timeseries_ext_full_train_mean_median.csv")
VAL_FILE   = os.path.join(PATHS.processed, "timeseries_ext_full_val_mean_median.csv")
TEST_FILE  = os.path.join(PATHS.processed, "timeseries_ext_full_test_mean_median.csv")

AUX_FILE = os.path.join(PATHS.intermediate, "stand_aux_features.csv")

MODEL_OUT = os.path.join(PATHS.models_trained, "BASELINE_rf_balanced.joblib")
FI_OUT = os.path.join(PATHS.outputs_tables, "BASELINE_feature_importance.csv")

CM_VAL_OUT = os.path.join(PATHS.outputs_figures, "confusion_matrix_validation.png")
CM_TEST_OUT = os.path.join(PATHS.outputs_figures, "confusion_matrix_test.png")
CM_TEST_NORM_OUT = os.path.join(PATHS.outputs_figures, "confusion_matrix_test_normalized.png")

# =========================================================
# CONFIG
# =========================================================

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

train_long = pd.read_csv(TRAIN_FILE)
val_long = pd.read_csv(VAL_FILE)
test_long = pd.read_csv(TEST_FILE)
aux = pd.read_csv(AUX_FILE)

TS_FEATURE_COLS = get_timeseries_feature_cols(train_long, ID_COL, TARGET_COL, DATE_COL)

print("\nTime-series features used in the model:")
print(TS_FEATURE_COLS)
print("Total time-series features:", len(TS_FEATURE_COLS))

AUX_NUMERIC_COLS = [
    "elev_mean",
    "elev_median",
    "elev_std",
    "elev_min",
    "elev_max"
]

AUX_CATEGORICAL_COLS = ["soil_main"]

# =========================================================
# 1. LOAD DATA
# =========================================================

print("Long-form read:")
print("train:", train_long.shape)
print("val:", val_long.shape)
print("test:", test_long.shape)
print("aux:", aux.shape)

# =========================================================
# 2. LONG -> WIDE
# =========================================================

train_wide = long_to_wide(train_long, "train", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
val_wide = long_to_wide(val_long, "val", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
test_wide = long_to_wide(test_long, "test", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)

print("\nWide-form before AUX:")
print("train:", train_wide.shape)
print("val:", val_wide.shape)
print("test:", test_wide.shape)

# =========================================================
# 3. AUX FEATURES
# =========================================================

train_wide = merge_aux(train_wide, aux, "train", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
val_wide = merge_aux(val_wide, aux, "val", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)
test_wide = merge_aux(test_wide, aux, "test", ID_COL, AUX_NUMERIC_COLS, AUX_CATEGORICAL_COLS)

# =========================================================
# 4. SOIL ONE-HOT
# =========================================================

train_wide, val_wide, test_wide = encode_categoricals_consistently(
    train_wide,
    val_wide,
    test_wide,
    AUX_CATEGORICAL_COLS
)

# =========================================================
# 5. ALIGN COLUMNS
# =========================================================

train_wide, val_wide, test_wide, feature_cols = align_columns(
    train_wide,
    val_wide,
    test_wide,
    ID_COL,
    TARGET_COL
)

print("\nWide-form + AUX:")
print("train:", train_wide.shape)
print("val:", val_wide.shape)
print("test:", test_wide.shape)

# =========================================================
# 6. X / y
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

print("\nTRAIN class counts:")
print(y_train.value_counts())

print("\nVAL class counts:")
print(y_val.value_counts())

print("\nTEST class counts:")
print(y_test.value_counts())

# =========================================================
# 7. IMPUTE
# =========================================================

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_val_imp = imputer.transform(X_val)

# =========================================================
# 8. TRAIN -> VAL BASELINE MODEL
# =========================================================

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining BASELINE model on TRAIN...")
rf.fit(X_train_imp, y_train)

val_pred = rf.predict(X_val_imp)

val_metrics = evaluate_model(
    "BASELINE VALIDATION - Random Forest class_weight=balanced",
    y_val,
    val_pred

)

plot_confusion_matrix_pretty(
    y_val,
    val_pred,
    "Valideerimisandmestiku confusion matrix",
    CM_VAL_OUT,
    normalize=False
)

# =========================================================
# 9. TRAIN + VAL -> TEST BASELINE MODEL
# =========================================================

X_trainval = pd.concat([X_train, X_val], axis=0)
y_trainval = pd.concat([y_train, y_val], axis=0)

valid_cols_trainval = X_trainval.columns[~X_trainval.isna().all()].tolist()

X_trainval = X_trainval[valid_cols_trainval].copy()
X_test_final = X_test[valid_cols_trainval].copy()

print("\nTRAIN+VAL class counts:")
print(y_trainval.value_counts())

imputer_final = SimpleImputer(strategy="median")

X_trainval_imp = imputer_final.fit_transform(X_trainval)
X_test_imp = imputer_final.transform(X_test_final)

rf_final = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining BASELINE final model on TRAIN+VAL...")
rf_final.fit(X_trainval_imp, y_trainval)

test_pred = rf_final.predict(X_test_imp)

test_metrics = evaluate_model(
    "BASELINE TEST - Random Forest class_weight=balanced",
    y_test,
    test_pred
)
plot_confusion_matrix_pretty(
    y_test,
    test_pred,
    "Testandmestiku confusion matrix",
    CM_TEST_OUT,
    normalize=False
)

plot_confusion_matrix_pretty(
    y_test,
    test_pred,
    "Testandmestiku normaliseeritud confusion matrix",
    CM_TEST_NORM_OUT,
    normalize=True
)

# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "feature": valid_cols_trainval,
    "importance": rf_final.feature_importances_
}).sort_values("importance", ascending=False)

print("\nBASELINE Top 50 feature importance:")
print(importance_df.head(50))

importance_df.to_csv(
    FI_OUT,
    index=False,
    encoding="utf-8-sig"
)

# =========================================================
# 11. SAVE MODEL
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
        "class_weight": "balanced",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    },
    MODEL_OUT
)

print("\nDone.")
print("Feature importance:", FI_OUT)
print("Model:", MODEL_OUT)
