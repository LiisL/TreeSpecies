"""Produces a pixel-based dominant tree-species map for the Subregion 2
pilot tile, using the actual primary model (EXP032's 4-stage hierarchical
Random Forest), masked to forest area with data/raw/metsamask.tif.

This is a from-scratch adaptation of the older, pre-EXP-numbering example
(scripts/predict_pixel_species_map.py), which used a single flat RF model
with the old elev_*/soil_main_* feature naming and a small hand-drawn AOI.
Differences here:
  - Model: EXP032's 4-stage hierarchy (group split -> KU-vs-MA /
    LV-vs-rest / HB-vs-KS-vs-LM), loaded from its actual joblib bundle
    (stage{1,2a,2bi,2bii}_model/imputer + group_map), not a single
    model/imputer pair. Routing uses each stage's PREDICTED output,
    mirroring the training-time predict_hierarchical() discipline.
  - Features: height_*/soil_majority_* naming (matching EXP027's fixed
    height/soil pipeline, which is what EXP032 was actually trained on),
    plus the 18 species-signature meta-features (HB_height_index,
    KU_winter_index, etc.) - entirely missing from the old example script.
  - AOI: none needed - the raster tile IS the area of interest; only the
    forest mask restricts which pixels get a prediction.
  - Height zonal-stat approximation: uses fast scipy.ndimage filters
    (uniform/minimum/maximum/median_filter) instead of the old script's
    generic_filter with a Python callback, which does not scale to a
    2500x2500 pixel tile (would take a very long time).
  - Confidence: reported as the deciding stage's own predict_proba max
    for the row's final predicted class (not a calibrated cross-stage
    product like EXP037 - EXP032 itself was never calibrated).
"""

import os
import re
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import rasterio

from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, median_filter, distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# =========================================================
# INPUTS
# =========================================================

MODEL_FILE = PATHS.models_trained / "EXP032_rf_deciduous_further_split.joblib"

RASTER_DIR = PATHS.intermediate / "pixel_stack_rasters_subregion2_pilot"

HEIGHT_RASTER = PATHS.raw / "H_20_21_22_23.tif"
SOIL_RASTER = PATHS.raw / "soil_siffer_10m.tif"
FOREST_MASK_RASTER = PATHS.raw / "metsamask.tif"

OUT_DIR = PATHS.outputs_predictions / "pixel_species_map_exp032_pilot"

SPECIES_TIF = os.path.join(OUT_DIR, "species_map_forestmask.tif")
CONF_TIF = os.path.join(OUT_DIR, "confidence_map_forestmask.tif")
CLASS_CODES_CSV = os.path.join(OUT_DIR, "class_codes.csv")

# Once a tile's species/confidence map is safely written, the downloaded
# raw Sentinel-2 pixel-stack rasters for that tile (many GB) are no longer
# needed - delete them to save disk space. Set to False to keep them (e.g.
# while still debugging the pipeline on a given tile).
DELETE_PIXEL_STACK_AFTER_PREDICTION = True

# =========================================================
# CONFIG
# =========================================================

NODATA_CLASS = 255
NODATA_FLOAT = -9999.0

FOREST_VALUES = [1]

BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B11", "B12",
    "NDVI", "NDRE", "NDMI", "NBR"
]

# Same approximation logic as the old example script: pixel-level prediction
# has no stand polygon, so height zonal stats are approximated with a moving
# window matching a ~1ha area (the smallest stand the model ever saw,
# scripts/split.py's MIN_AREA=1.0 filter).
HEIGHT_ZONAL_AREA_M2 = 10000.0

GROUP_MAP = {
    "KU": "CONIFER", "MA": "CONIFER",
    "HB": "DECIDUOUS", "KS": "DECIDUOUS", "LM": "DECIDUOUS", "LV": "DECIDUOUS",
    "MUU": "MUU"
}
DECIDUOUS_SPECIES = {"HB", "KS", "LM", "LV"}

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def date_from_filename(path):
    name = os.path.basename(path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if not m:
        raise ValueError(f"No date found in filename: {name}")
    return m.group(1)


def get_reference_raster(stack_files):
    ref = stack_files[0]
    with rasterio.open(ref) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
    return ref, profile, height, width, transform, crs


def read_raster_aligned(path, ref_profile, resampling):
    with rasterio.open(path) as src:
        with WarpedVRT(
            src,
            crs=ref_profile["crs"],
            transform=ref_profile["transform"],
            width=ref_profile["width"],
            height=ref_profile["height"],
            resampling=resampling
        ) as vrt:
            arr = vrt.read(1).astype("float32")
            nodata = vrt.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            return arr


def make_forest_mask(path, ref_profile):
    forest_arr = read_raster_aligned(path, ref_profile, Resampling.nearest)
    forest_mask = np.isin(forest_arr, FOREST_VALUES) & ~np.isnan(forest_arr)
    return forest_mask


def fill_nearest(arr):
    """Fills NaN cells with the nearest non-NaN value - needed before the
    fast scipy.ndimage window filters below, which are not NaN-aware."""
    if not np.isnan(arr).any():
        return arr
    mask = np.isnan(arr)
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def windowed_height_stats_fast(height_arr, window_px):
    """Fast replacement for the old example script's generic_filter-based
    windowed zonal stats - uses vectorized scipy.ndimage filters instead of
    a per-pixel Python callback, which does not scale past a small AOI.
    NaNs are nearest-filled first (edge-of-data-extent only, in practice)."""
    filled = fill_nearest(height_arr)

    mean = uniform_filter(filled, size=window_px, mode="nearest")
    mean_sq = uniform_filter(filled ** 2, size=window_px, mode="nearest")
    var = np.clip(mean_sq - mean ** 2, 0, None)
    std = np.sqrt(var)

    median = median_filter(filled, size=window_px, mode="nearest")
    mn = minimum_filter(filled, size=window_px, mode="nearest")
    mx = maximum_filter(filled, size=window_px, mode="nearest")

    return mean, median, std, mn, mx


def height_window_size_px(transform):
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    px_area = pixel_size_x * pixel_size_y
    side_px = int(round((HEIGHT_ZONAL_AREA_M2 / px_area) ** 0.5))
    side_px = max(side_px, 1)
    if side_px % 2 == 0:
        side_px += 1
    return side_px


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
    """Byte-identical formulas to experiments/EXP032_deciduous_further_split/exp032.py's
    add_species_signature_features(), applied to the pixel-level wide table."""
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


def build_feature_table(stack_data, stack_dates, height_arr, soil_arr, feature_cols, transform):
    data = {}

    for date_str, arr in zip(stack_dates, stack_data):
        date_key = date_str.replace("-", "_")
        for band_idx, band_name in enumerate(BAND_NAMES):
            col = f"{band_name}_mean_{date_key}"
            band = arr[band_idx].astype("float32")
            band[band <= -9990] = np.nan
            data[col] = band.reshape(-1)

    window_px = height_window_size_px(transform)
    print(f"Height feature window size: {window_px} x {window_px} pixels (~{HEIGHT_ZONAL_AREA_M2/10000:.2f} ha)")
    h_mean, h_median, h_std, h_min, h_max = windowed_height_stats_fast(height_arr, window_px)
    data["height_mean"] = h_mean.reshape(-1)
    data["height_median"] = h_median.reshape(-1)
    data["height_std"] = h_std.reshape(-1)
    data["height_min"] = h_min.reshape(-1)
    data["height_max"] = h_max.reshape(-1)

    # Soil: per-pixel majority code (a pixel-level "majority" degenerates to
    # the pixel's own value), one-hot encoded to match soil_majority_<code>.
    soil_flat = soil_arr.reshape(-1)
    soil_series = pd.Series(soil_flat).round()
    soil_codes_present = soil_series.dropna().astype("Int64").astype(str).unique()

    soil_feature_cols_in_model = [c for c in feature_cols if c.startswith("soil_majority_")]
    soil_as_str = soil_series.astype("Int64").astype(str).values

    for col in soil_feature_cols_in_model:
        soil_code = col.replace("soil_majority_", "")
        data[col] = (soil_as_str == soil_code).astype("uint8")

    X = pd.DataFrame(data)

    # Species-signature meta-features (needs the spectral columns already in X)
    X = add_species_signature_features(X)

    missing_cols = [c for c in feature_cols if c not in X.columns]
    if missing_cols:
        print(f"Note: {len(missing_cols)} model feature(s) not producible from this pixel stack "
              f"(will be imputed): {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}")
        missing_df = pd.DataFrame(np.nan, index=X.index, columns=missing_cols)
        X = pd.concat([X, missing_df], axis=1)

    X = X[feature_cols].copy()
    return X


def delete_pixel_stack(raster_dir, stack_files):
    """Deletes the downloaded per-tile Sentinel-2 pixel-stack rasters (and
    the small band_order.csv alongside them) once the species/confidence
    maps have been written successfully - these are the large, disk-space-
    heavy files (many GB per tile); the final maps are tiny by comparison
    and are what's actually kept. Only called after both output rasters
    are confirmed written, so a failed run never deletes source data it
    might need to be re-run."""
    freed_bytes = 0
    for f in stack_files:
        try:
            freed_bytes += os.path.getsize(f)
            os.remove(f)
        except OSError as e:
            print(f"Warning: could not delete {f}: {e}")

    band_order_csv = os.path.join(raster_dir, "band_order.csv")
    if os.path.exists(band_order_csv):
        try:
            os.remove(band_order_csv)
        except OSError:
            pass

    try:
        if not os.listdir(raster_dir):
            os.rmdir(raster_dir)
    except OSError:
        pass

    print(f"\nDeleted {len(stack_files)} pixel-stack raster(s), freed ~{freed_bytes / 1e9:.2f} GB "
          f"from {raster_dir}")


def predict_hierarchical_with_confidence(X, m):
    """Same predicted-routing discipline as training (exp032.py's
    predict_hierarchical), plus per-row confidence = the deciding stage's
    own max predicted probability for the row's final class."""
    n = len(X)
    final_pred = np.empty(n, dtype=object)
    final_conf = np.full(n, np.nan, dtype="float32")

    X_imp1 = m["stage1_imputer"].transform(X)
    group_proba = m["stage1_model"].predict_proba(X_imp1)
    group_pred = m["stage1_model"].classes_[np.argmax(group_proba, axis=1)]
    group_conf = group_proba.max(axis=1)

    conifer_rows = group_pred == "CONIFER"
    if conifer_rows.any():
        X_conifer_imp = m["stage2a_imputer"].transform(X.loc[conifer_rows])
        proba = m["stage2a_conifer_model"].predict_proba(X_conifer_imp)
        pred = m["stage2a_conifer_model"].classes_[np.argmax(proba, axis=1)]
        final_pred[conifer_rows] = pred
        final_conf[conifer_rows] = proba.max(axis=1)

    deciduous_rows = group_pred == "DECIDUOUS"
    if deciduous_rows.any():
        X_dec_imp = m["stage2bi_imputer"].transform(X.loc[deciduous_rows])
        proba_bi = m["stage2bi_lv_vs_rest_model"].predict_proba(X_dec_imp)
        classes_bi = m["stage2bi_lv_vs_rest_model"].classes_
        lv_pred = classes_bi[np.argmax(proba_bi, axis=1)]
        lv_conf = proba_bi.max(axis=1)

        dec_idx = np.where(deciduous_rows)[0]
        is_lv = lv_pred == "LV"
        lv_idx = dec_idx[is_lv]
        rest_idx = dec_idx[~is_lv]

        final_pred[lv_idx] = "LV"
        final_conf[lv_idx] = lv_conf[is_lv]

        if len(rest_idx) > 0:
            X_rest_imp = m["stage2bii_imputer"].transform(X.iloc[rest_idx])
            proba_bii = m["stage2bii_hb_ks_lm_model"].predict_proba(X_rest_imp)
            classes_bii = m["stage2bii_hb_ks_lm_model"].classes_
            pred_bii = classes_bii[np.argmax(proba_bii, axis=1)]
            conf_bii = proba_bii.max(axis=1)
            final_pred[rest_idx] = pred_bii
            final_conf[rest_idx] = conf_bii

    muu_rows = group_pred == "MUU"
    final_pred[muu_rows] = "MUU"
    final_conf[muu_rows] = group_conf[muu_rows]

    return final_pred, final_conf


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    m = joblib.load(MODEL_FILE)
    feature_cols = m["feature_cols"]
    print("Model loaded. Feature count:", len(feature_cols))

    stack_files = sorted(glob.glob(os.path.join(RASTER_DIR, "pixel_stack_*.tif")))
    print("Raster stack files found:", len(stack_files))

    if not stack_files:
        raise RuntimeError(f"No pixel stack rasters found in {RASTER_DIR}")

    ref_file, profile, height, width, transform, crs = get_reference_raster(stack_files)
    print("Reference raster:", ref_file, "| size:", width, "x", height, "| CRS:", crs)

    print("\nReading and aligning forest mask...")
    forest_mask = make_forest_mask(FOREST_MASK_RASTER, profile)
    print("Forest pixels in tile:", int(forest_mask.sum()), "/", height * width,
          f"({100*forest_mask.sum()/(height*width):.1f}%)")

    stack_data, stack_dates = [], []
    for path in stack_files:
        date_str = date_from_filename(path)
        stack_dates.append(date_str)
        with rasterio.open(path) as src:
            arr = src.read().astype("float32")
            if src.count != len(BAND_NAMES):
                raise RuntimeError(f"{path}: expected {len(BAND_NAMES)} bands, got {src.count}")
            stack_data.append(arr)
    print("\nDates loaded:", len(stack_dates))

    print("\nReading and aligning height raster...")
    height_arr = read_raster_aligned(HEIGHT_RASTER, profile, Resampling.bilinear)

    print("Reading and aligning soil raster...")
    soil_arr = read_raster_aligned(SOIL_RASTER, profile, Resampling.nearest)

    print("\nBuilding pixel feature table...")
    X = build_feature_table(stack_data, stack_dates, height_arr, soil_arr, feature_cols, transform)
    print("X shape:", X.shape)

    valid_mask_2d = forest_mask
    valid_mask = valid_mask_2d.reshape(-1)

    sentinel_cols = [c for c in feature_cols if any(c.startswith(p + "_mean_") for p in BAND_NAMES)]
    has_s2_data = X[sentinel_cols].notna().any(axis=1).values
    valid_mask = valid_mask & has_s2_data

    print("\nTotal pixels:", height * width)
    print("Pixels with forest + S2 data:", int(valid_mask.sum()))

    if valid_mask.sum() == 0:
        raise RuntimeError("No predictable pixels found. Check the forest mask alignment and rasters.")

    species_code_flat = np.full(height * width, NODATA_CLASS, dtype="uint8")
    conf_flat = np.full(height * width, NODATA_FLOAT, dtype="float32")

    X_valid = X.loc[valid_mask].reset_index(drop=True)

    print("\nPredicting (4-stage hierarchy)...")
    pred_labels, pred_conf = predict_hierarchical_with_confidence(X_valid, m)

    classes = sorted(set(GROUP_MAP.keys()))
    class_to_code = {cls: i + 1 for i, cls in enumerate(classes)}
    pred_codes = np.array([class_to_code[p] for p in pred_labels], dtype="uint8")

    species_code_flat[valid_mask] = pred_codes
    conf_flat[valid_mask] = pred_conf.astype("float32")

    species_map = species_code_flat.reshape(height, width)
    conf_map = conf_flat.reshape(height, width)

    class_df = pd.DataFrame({"class_code": list(class_to_code.values()), "class_name": list(class_to_code.keys())})
    class_df.to_csv(CLASS_CODES_CSV, index=False, encoding="utf-8-sig")
    print("\nClass codes:\n", class_df)

    species_profile = profile.copy()
    species_profile.update(count=1, dtype="uint8", nodata=NODATA_CLASS, compress="lzw")
    with rasterio.open(SPECIES_TIF, "w", **species_profile) as dst:
        dst.write(species_map, 1)
    print("\nSpecies map saved:", SPECIES_TIF)

    conf_profile = profile.copy()
    conf_profile.update(count=1, dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    with rasterio.open(CONF_TIF, "w", **conf_profile) as dst:
        dst.write(conf_map, 1)
    print("Confidence map saved:", CONF_TIF)

    # Only delete source rasters once both output files are confirmed
    # written to disk - never on a failed/partial run.
    if DELETE_PIXEL_STACK_AFTER_PREDICTION and os.path.exists(SPECIES_TIF) and os.path.exists(CONF_TIF):
        delete_pixel_stack(RASTER_DIR, stack_files)

    print("\nDone.")


if __name__ == "__main__":
    main()
