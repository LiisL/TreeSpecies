"""Scales the pixel-based species mapping pipeline (piloted in
scripts/download_pixel_stack_subregion2_pilot.py +
scripts/predict_pixel_species_map_exp032.py) up to all of Subregion 2's
forest area, using EXP032 (the primary model).

Processes ONE 25km x 25km tile at a time: download all 26 dates -> build
that tile's species/confidence map -> delete the tile's raw rasters ->
move to the next tile. This keeps disk usage bounded to roughly one
tile's worth of data (~4-9GB) at a time, rather than accumulating the
full ~150GB Subregion 2 total, matching the disk-space-driven cleanup
behavior added to the per-tile prediction scripts.

The pilot tile (tx=1, ty=2) is already done (see
outputs/predictions/pixel_species_map_exp032_pilot/) and is skipped here;
its output is included in the final mosaic step.

Resumable: if a tile's output already exists, both the download and
prediction steps for that tile are skipped - safe to re-run this script
after an interruption.

After all tiles are processed, mosaics every tile's species_map and
confidence_map into one seamless Subregion 2-wide GeoTIFF pair.
"""

import os
import sys
import time
import glob
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, median_filter, distance_transform_edt
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# =========================================================
# CONFIG
# =========================================================

SUBREGION2_BOUNDS = (461678.49, 6456097.71, 528234.01, 6592745.26)  # EPSG:3301
TILE_M = 25000
FOREST_FRACTION_THRESHOLD = 0.005  # skip tiles with <0.5% forest (coarse pre-check)

PILOT_BBOX = (486678.0, 6506098.0, 511678.0, 6531098.0)
PILOT_OUT_DIR = PATHS.outputs_predictions / "pixel_species_map_exp032_pilot"

RESOLUTION = 10
TARGET_EPSG = 3301
WINDOW_DAYS = 10

DATES = [
    "2025-01-31", "2025-02-10", "2025-02-20", "2025-03-02", "2025-03-12",
    "2025-04-01", "2025-04-11", "2025-04-21", "2025-05-01", "2025-05-11",
    "2025-05-21", "2025-05-31", "2025-06-10", "2025-06-20", "2025-06-30",
    "2025-07-10", "2025-07-20", "2025-07-30", "2025-08-09", "2025-08-19",
    "2025-08-29", "2025-09-08", "2025-09-18", "2025-09-28", "2025-10-08",
    "2025-10-18",
]

BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B11", "B12",
    "NDVI", "NDRE", "NDMI", "NBR"
]

MODEL_FILE = PATHS.models_trained / "EXP032_rf_deciduous_further_split.joblib"
HEIGHT_RASTER = PATHS.raw / "H_20_21_22_23.tif"
SOIL_RASTER = PATHS.raw / "soil_siffer_10m.tif"
FOREST_MASK_RASTER = PATHS.raw / "metsamask.tif"

TILES_ROOT_RASTER_DIR = PATHS.intermediate / "pixel_stack_rasters_sr2_tiles"
TILES_ROOT_OUT_DIR = PATHS.outputs_predictions / "pixel_species_map_exp032_sr2_tiles"
FINAL_OUT_DIR = PATHS.outputs_predictions / "pixel_species_map_exp032_subregion2"

NODATA_CLASS = 255
NODATA_FLOAT = -9999.0
FOREST_VALUES = [1]
HEIGHT_ZONAL_AREA_M2 = 10000.0

GROUP_MAP = {
    "KU": "CONIFER", "MA": "CONIFER",
    "HB": "DECIDUOUS", "KS": "DECIDUOUS", "LM": "DECIDUOUS", "LV": "DECIDUOUS",
    "MUU": "MUU"
}

CLASS_COLORMAP = {
    0: (255, 255, 255, 0),
    1: (230, 171, 2, 255),   # HB
    2: (102, 194, 165, 255), # KS
    3: (27, 77, 46, 255),    # KU
    4: (140, 81, 10, 255),   # LM
    5: (166, 206, 227, 255), # LV
    6: (51, 105, 30, 255),   # MA
    7: (153, 153, 153, 255), # MUU
    255: (0, 0, 0, 0),
}

EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "SCL", "dataMask"],
      units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE",
              "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE",
              "DN", "DN"]
    }],
    output: { bands: 14, sampleType: "FLOAT32", nodataValue: -9999 }
  };
}
function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);
  if (good) {
    return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07,
            sample.B08, sample.B8A, sample.B11, sample.B12, ndvi, ndre, ndmi, nbr];
  } else {
    return [-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999,-9999];
  }
}
"""

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def load_env_file(env_path=None):
    if env_path is None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
    else:
        env_path = Path(env_path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def fetch_access_token(client_id, client_secret, token_url, timeout=60):
    r = requests.post(token_url, data={
        "grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret
    }, timeout=timeout)
    if r.status_code != 200:
        print(r.text)
        raise RuntimeError("Failed to obtain access token.")
    return r.json()["access_token"]


load_env_file()
CLIENT_ID = os.environ.get("CDSE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET")
if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET missing from .env")


# =========================================================
# TILE GRID
# =========================================================

def compute_forest_tiles():
    minx, miny, maxx, maxy = SUBREGION2_BOUNDS
    nx = int(np.ceil((maxx - minx) / TILE_M))
    ny = int(np.ceil((maxy - miny) / TILE_M))

    with rasterio.open(FOREST_MASK_RASTER) as src:
        window = rasterio.windows.from_bounds(*SUBREGION2_BOUNDS, transform=src.transform)
        scale = 10
        out_h, out_w = int(window.height / scale), int(window.width / scale)
        data = src.read(1, window=window, out_shape=(out_h, out_w), resampling=Resampling.nearest)

    px_per_tile = int(TILE_M / 100)
    tiles = []
    for ty in range(ny):
        for tx in range(nx):
            y0, y1 = ty * px_per_tile, min((ty + 1) * px_per_tile, out_h)
            x0, x1 = tx * px_per_tile, min((tx + 1) * px_per_tile, out_w)
            if y0 >= out_h or x0 >= out_w:
                continue
            sub = data[y0:y1, x0:x1]
            if sub.size == 0:
                continue
            frac = (sub == 1).mean()
            if frac <= FOREST_FRACTION_THRESHOLD:
                continue
            tile_minx = minx + tx * TILE_M
            tile_miny = miny + ty * TILE_M
            tile_maxx = min(tile_minx + TILE_M, maxx)
            tile_maxy = min(tile_miny + TILE_M, maxy)
            tiles.append({
                "tx": tx, "ty": ty, "forest_frac": frac,
                "bbox": [tile_minx, tile_miny, tile_maxx, tile_maxy]
            })

    tiles.sort(key=lambda t: -t["forest_frac"])
    return tiles


def is_pilot_tile(bbox):
    return abs(bbox[0] - PILOT_BBOX[0]) < 1 and abs(bbox[1] - PILOT_BBOX[1]) < 1


# =========================================================
# DOWNLOAD (per tile)
# =========================================================

def tile_size_px(bbox):
    minx, miny, maxx, maxy = bbox
    return int(round((maxx - minx) / RESOLUTION)), int(round((maxy - miny) / RESOLUTION))


def date_window(date_str):
    start = datetime.strptime(date_str, "%Y-%m-%d")
    end = start + timedelta(days=WINDOW_DAYS)
    return start.strftime("%Y-%m-%dT00:00:00Z"), end.strftime("%Y-%m-%dT23:59:59Z")


def make_process_payload(bbox, width, height, start_time, end_time):
    return {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{TARGET_EPSG}"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {
                "timeRange": {"from": start_time, "to": end_time},
                "maxCloudCoverage": 30, "mosaickingOrder": "leastCC"
            }}]
        },
        "output": {"width": width, "height": height,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": EVALSCRIPT
    }


def download_tile(bbox, out_dir, token_holder):
    os.makedirs(out_dir, exist_ok=True)
    width, height = tile_size_px(bbox)

    for i, date_str in enumerate(DATES, start=1):
        out_tif = os.path.join(out_dir, f"pixel_stack_{date_str}.tif")
        if os.path.exists(out_tif):
            continue

        start_time, end_time = date_window(date_str)
        payload = make_process_payload(bbox, width, height, start_time, end_time)
        headers = {
            "Authorization": f"Bearer {token_holder['token']}",
            "Content-Type": "application/json", "Accept": "image/tiff"
        }

        for attempt in range(1, 4):
            r = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=300)

            if r.status_code == 401:
                print("  Token expired, refreshing...")
                token_holder["token"] = fetch_access_token(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)
                headers["Authorization"] = f"Bearer {token_holder['token']}"
                continue

            if r.status_code == 429:
                wait = 30 * attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if r.status_code >= 400:
                print(f"  [{i}/{len(DATES)}] {date_str}: error {r.status_code}: {r.text[:300]}")
                if attempt == 3:
                    raise RuntimeError(f"Process API error {r.status_code} for {date_str} after 3 attempts")
                time.sleep(10 * attempt)
                continue

            with open(out_tif, "wb") as f:
                f.write(r.content)
            print(f"  [{i}/{len(DATES)}] {date_str} saved ({os.path.getsize(out_tif)/1e6:.0f} MB)")
            break
        else:
            raise RuntimeError(f"Failed to download {date_str}")

        time.sleep(1)


# =========================================================
# PREDICTION (per tile) - same logic as predict_pixel_species_map_exp032.py
# =========================================================

def date_from_filename(path):
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1)


def read_raster_aligned(path, ref_profile, resampling):
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=ref_profile["crs"], transform=ref_profile["transform"],
                        width=ref_profile["width"], height=ref_profile["height"], resampling=resampling) as vrt:
            arr = vrt.read(1).astype("float32")
            if vrt.nodata is not None:
                arr[arr == vrt.nodata] = np.nan
            return arr


def make_forest_mask(ref_profile):
    forest_arr = read_raster_aligned(FOREST_MASK_RASTER, ref_profile, Resampling.nearest)
    return np.isin(forest_arr, FOREST_VALUES) & ~np.isnan(forest_arr)


def fill_nearest(arr):
    if not np.isnan(arr).any():
        return arr
    idx = distance_transform_edt(np.isnan(arr), return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def height_window_size_px(transform):
    px_area = abs(transform.a) * abs(transform.e)
    side_px = max(int(round((HEIGHT_ZONAL_AREA_M2 / px_area) ** 0.5)), 1)
    return side_px + 1 if side_px % 2 == 0 else side_px


def windowed_height_stats_fast(height_arr, window_px):
    filled = fill_nearest(height_arr)
    mean = uniform_filter(filled, size=window_px, mode="nearest")
    mean_sq = uniform_filter(filled ** 2, size=window_px, mode="nearest")
    std = np.sqrt(np.clip(mean_sq - mean ** 2, 0, None))
    median = median_filter(filled, size=window_px, mode="nearest")
    mn = minimum_filter(filled, size=window_px, mode="nearest")
    mx = maximum_filter(filled, size=window_px, mode="nearest")
    return mean, median, std, mn, mx


def safe_mean(df, cols):
    existing = [c for c in cols if c in df.columns]
    return df[existing].mean(axis=1) if existing else np.nan


def safe_diff(df, col_late, col_early):
    return df[col_late] - df[col_early] if col_late in df.columns and col_early in df.columns else np.nan


def add_species_signature_features(df):
    if "height_max" in df.columns and "height_std" in df.columns:
        df["HB_height_index"] = df["height_max"] + df["height_std"]
        df["HB_height_product"] = df["height_max"] * df["height_std"]
    else:
        df["HB_height_index"] = np.nan
        df["HB_height_product"] = np.nan
    df["HB_structure_ratio"] = (df["height_std"] / (df["height_mean"] + 1)
                                 if all(c in df.columns for c in ["height_std", "height_mean"]) else np.nan)
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
            band = arr[band_idx].astype("float32")
            band[band <= -9990] = np.nan
            data[f"{band_name}_mean_{date_key}"] = band.reshape(-1)

    window_px = height_window_size_px(transform)
    h_mean, h_median, h_std, h_min, h_max = windowed_height_stats_fast(height_arr, window_px)
    data["height_mean"] = h_mean.reshape(-1)
    data["height_median"] = h_median.reshape(-1)
    data["height_std"] = h_std.reshape(-1)
    data["height_min"] = h_min.reshape(-1)
    data["height_max"] = h_max.reshape(-1)

    soil_series = pd.Series(soil_arr.reshape(-1)).round()
    soil_as_str = soil_series.astype("Int64").astype(str).values
    for col in [c for c in feature_cols if c.startswith("soil_majority_")]:
        soil_code = col.replace("soil_majority_", "")
        data[col] = (soil_as_str == soil_code).astype("uint8")

    X = pd.DataFrame(data)
    X = add_species_signature_features(X)

    missing_cols = [c for c in feature_cols if c not in X.columns]
    if missing_cols:
        X = pd.concat([X, pd.DataFrame(np.nan, index=X.index, columns=missing_cols)], axis=1)
    return X[feature_cols].copy()


def predict_hierarchical_with_confidence(X, m):
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
        final_pred[conifer_rows] = m["stage2a_conifer_model"].classes_[np.argmax(proba, axis=1)]
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
        lv_idx, rest_idx = dec_idx[is_lv], dec_idx[~is_lv]
        final_pred[lv_idx] = "LV"
        final_conf[lv_idx] = lv_conf[is_lv]

        if len(rest_idx) > 0:
            X_rest_imp = m["stage2bii_imputer"].transform(X.iloc[rest_idx])
            proba_bii = m["stage2bii_hb_ks_lm_model"].predict_proba(X_rest_imp)
            classes_bii = m["stage2bii_hb_ks_lm_model"].classes_
            final_pred[rest_idx] = classes_bii[np.argmax(proba_bii, axis=1)]
            final_conf[rest_idx] = proba_bii.max(axis=1)

    muu_rows = group_pred == "MUU"
    final_pred[muu_rows] = "MUU"
    final_conf[muu_rows] = group_conf[muu_rows]
    return final_pred, final_conf


def predict_tile(raster_dir, out_dir, model_bundle):
    os.makedirs(out_dir, exist_ok=True)
    species_tif = os.path.join(out_dir, "species_map_forestmask.tif")
    conf_tif = os.path.join(out_dir, "confidence_map_forestmask.tif")
    class_csv = os.path.join(out_dir, "class_codes.csv")

    feature_cols = model_bundle["feature_cols"]
    stack_files = sorted(glob.glob(os.path.join(raster_dir, "pixel_stack_*.tif")))
    if not stack_files:
        raise RuntimeError(f"No pixel stack rasters in {raster_dir}")

    with rasterio.open(stack_files[0]) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        transform = src.transform

    forest_mask = make_forest_mask(profile)
    print(f"  Forest pixels: {int(forest_mask.sum())}/{height*width} ({100*forest_mask.sum()/(height*width):.1f}%)")

    stack_data, stack_dates = [], []
    for path in stack_files:
        stack_dates.append(date_from_filename(path))
        with rasterio.open(path) as src:
            stack_data.append(src.read().astype("float32"))

    height_arr = read_raster_aligned(HEIGHT_RASTER, profile, Resampling.bilinear)
    soil_arr = read_raster_aligned(SOIL_RASTER, profile, Resampling.nearest)

    X = build_feature_table(stack_data, stack_dates, height_arr, soil_arr, feature_cols, transform)

    valid_mask = forest_mask.reshape(-1)
    sentinel_cols = [c for c in feature_cols if any(c.startswith(p + "_mean_") for p in BAND_NAMES)]
    valid_mask = valid_mask & X[sentinel_cols].notna().any(axis=1).values

    print(f"  Predictable pixels: {int(valid_mask.sum())}")
    if valid_mask.sum() == 0:
        print("  No predictable pixels - skipping tile output.")
        # Mark this tile as done-but-empty so future runs don't waste quota
        # re-downloading it - the coarse 100m forest-fraction pre-check that
        # selects tiles can overestimate a tile's real (10m) forest content.
        with open(os.path.join(out_dir, "NO_FOREST_PIXELS.marker"), "w") as f:
            f.write("Tile had 0 predictable forest pixels at full resolution - no map produced.\n")
        return None

    species_code_flat = np.full(height * width, NODATA_CLASS, dtype="uint8")
    conf_flat = np.full(height * width, NODATA_FLOAT, dtype="float32")

    X_valid = X.loc[valid_mask].reset_index(drop=True)
    pred_labels, pred_conf = predict_hierarchical_with_confidence(X_valid, model_bundle)

    classes = sorted(GROUP_MAP.keys())
    class_to_code = {cls: i + 1 for i, cls in enumerate(classes)}
    species_code_flat[valid_mask] = np.array([class_to_code[p] for p in pred_labels], dtype="uint8")
    conf_flat[valid_mask] = pred_conf.astype("float32")

    class_df = pd.DataFrame({"class_code": list(class_to_code.values()), "class_name": list(class_to_code.keys())})
    class_df.to_csv(class_csv, index=False, encoding="utf-8-sig")

    species_profile = profile.copy()
    species_profile.update(count=1, dtype="uint8", nodata=NODATA_CLASS, compress="lzw")
    with rasterio.open(species_tif, "w", **species_profile) as dst:
        dst.write(species_code_flat.reshape(height, width), 1)
        dst.write_colormap(1, CLASS_COLORMAP)

    conf_profile = profile.copy()
    conf_profile.update(count=1, dtype="float32", nodata=NODATA_FLOAT, compress="lzw")
    with rasterio.open(conf_tif, "w", **conf_profile) as dst:
        dst.write(conf_flat.reshape(height, width), 1)

    return species_tif, conf_tif


def delete_raster_dir(raster_dir):
    if os.path.exists(raster_dir):
        freed = sum(os.path.getsize(f) for f in glob.glob(os.path.join(raster_dir, "*")))
        shutil.rmtree(raster_dir)
        print(f"  Deleted tile rasters, freed ~{freed/1e9:.2f} GB")


# =========================================================
# MOSAIC
# =========================================================

def mosaic_all_tiles():
    print("\n" + "=" * 60)
    print("Mosaicking all tiles into one Subregion 2 map")
    print("=" * 60)

    os.makedirs(FINAL_OUT_DIR, exist_ok=True)

    species_paths = [os.path.join(PILOT_OUT_DIR, "species_map_forestmask.tif")]
    conf_paths = [os.path.join(PILOT_OUT_DIR, "confidence_map_forestmask.tif")]

    for d in sorted(glob.glob(os.path.join(TILES_ROOT_OUT_DIR, "*"))):
        sp = os.path.join(d, "species_map_forestmask.tif")
        cf = os.path.join(d, "confidence_map_forestmask.tif")
        if os.path.exists(sp):
            species_paths.append(sp)
            conf_paths.append(cf)

    print(f"Tiles to mosaic: {len(species_paths)}")

    srcs = [rasterio.open(p) for p in species_paths]
    mosaic, out_transform = merge(srcs, nodata=NODATA_CLASS)
    out_profile = srcs[0].profile.copy()
    out_profile.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=out_transform,
                        compress="lzw", nodata=NODATA_CLASS)
    out_species = os.path.join(FINAL_OUT_DIR, "species_map_subregion2.tif")
    with rasterio.open(out_species, "w", **out_profile) as dst:
        dst.write(mosaic)
        dst.write_colormap(1, CLASS_COLORMAP)
    for s in srcs:
        s.close()
    print("Saved:", out_species)

    srcs = [rasterio.open(p) for p in conf_paths]
    mosaic_c, out_transform_c = merge(srcs, nodata=NODATA_FLOAT)
    out_profile_c = srcs[0].profile.copy()
    out_profile_c.update(height=mosaic_c.shape[1], width=mosaic_c.shape[2], transform=out_transform_c,
                          compress="lzw", nodata=NODATA_FLOAT)
    out_conf = os.path.join(FINAL_OUT_DIR, "confidence_map_subregion2.tif")
    with rasterio.open(out_conf, "w", **out_profile_c) as dst:
        dst.write(mosaic_c)
    for s in srcs:
        s.close()
    print("Saved:", out_conf)


# =========================================================
# MAIN
# =========================================================

def main():
    tiles = compute_forest_tiles()
    remaining = [t for t in tiles if not is_pilot_tile(t["bbox"])]
    print(f"Total forest tiles: {len(tiles)} (1 pilot already done, {len(remaining)} remaining)")

    model_bundle = joblib.load(MODEL_FILE)
    token_holder = {"token": fetch_access_token(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)}

    for i, tile in enumerate(remaining, start=1):
        tx, ty, bbox = tile["tx"], tile["ty"], tile["bbox"]
        tile_name = f"tile_{tx}_{ty}"
        raster_dir = os.path.join(TILES_ROOT_RASTER_DIR, tile_name)
        out_dir = os.path.join(TILES_ROOT_OUT_DIR, tile_name)
        species_tif = os.path.join(out_dir, "species_map_forestmask.tif")
        no_forest_marker = os.path.join(out_dir, "NO_FOREST_PIXELS.marker")

        print(f"\n{'#'*60}\n# TILE {i}/{len(remaining)}: {tile_name} (forest {tile['forest_frac']:.1%})\n{'#'*60}")

        if os.path.exists(species_tif):
            print(f"  Already done ({species_tif}), skipping.")
            continue

        if os.path.exists(no_forest_marker):
            print(f"  Already checked, confirmed no forest pixels at full resolution - skipping.")
            continue

        try:
            print(f"Downloading {len(DATES)} dates for bbox {bbox}...")
            download_tile(bbox, raster_dir, token_holder)

            print("Predicting...")
            result = predict_tile(raster_dir, out_dir, model_bundle)
            if result:
                print("  Species map:", result[0])

            delete_raster_dir(raster_dir)

        except Exception as e:
            print(f"  ERROR on {tile_name}: {e}")
            print("  Continuing to next tile.")
            continue

    mosaic_all_tiles()
    print("\nAll done.")


if __name__ == "__main__":
    main()
