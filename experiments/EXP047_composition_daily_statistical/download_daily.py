"""EXP047: downloads DAILY per-parcel Sentinel-2 statistics (mean + median,
Statistical API - the same approach as scripts/download_sentinel2_daily_statistical.py,
NOT the raw-pixel Process API used by experiments/EXP043_pixel_level_data)
for the NEW composition_base_2026_subregion2.gpkg parcel set: both pure
(>=80% purity) AND mixed stands, 10,309 parcels total.

Why a new download instead of reusing the old daily dataset: only ~39% of
the new dataset's pure-stand ids match the old daily dataset's ids (most
stands have been re-inventoried/renumbered since the original query - see
scripts/build_composition_base_dataset.py's docstring for the same finding
at the full-composition-CSV level). Per explicit user decision, this
downloads EVERYTHING (pure + mixed) fresh from the new dataset rather than
mixing two differently-sourced pure-stand sets, and writes to its own,
clearly separate output files/directory so it never collides with the old
Subregion2_{train,val,test}_sentinel2_daily_timeseries.csv files.

Purpose: train Random Forest + XGBoost classifiers on pure+mixed stands
together (dominant-species classification), and separately a multi-output
regression model predicting full species composition (the
liik_1..5/osakaal_1..5 columns already present on the parcel file) -
the Klehr et al. 2025-style approach discussed and cited in the thesis.
This script only produces the spectral timeseries; composition columns
are joined back in from PARCEL_FILE at training time, not carried through
here (parse_stats(), reused as-is from src/sentinel_api/statistical_client.py,
only carries id/target/split/subregion through).

Unlike scripts/download_sentinel2_daily_statistical.py (which requires
manually editing SELECT_SPLIT + OUTPUT and re-running once per split),
this loops over train/val/test automatically in one run - both to reduce
manual steps and because, given how many CDSE account switches this
project's downloads have already needed, a single resumable entry point
is more convenient to re-launch after a quota interruption.
"""

import os
import sys
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd

from sentinelhub import DataCollection, Geometry, CRS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sentinel_api.auth import get_config, check_token, load_env_file
from src.sentinel_api.statistical_client import apply_filters, request_with_retry, parse_stats
from src.paths import PATHS

# =========================================================
# INPUT / OUTPUT
# =========================================================

PARCEL_FILE = PATHS.raw / "composition_base_2026_subregion2.gpkg"
ID_COL = "id"
TARGET_COL = "peapuuliik"  # renamed to "target" internally for parse_stats/apply_filters compatibility

OUT_DIR = PATHS.intermediate / "EXP047_composition_daily"
SPLITS = ["train", "val", "test"]


def output_path(split):
    return OUT_DIR / f"composition_{split}_sentinel2_daily_timeseries.csv"


def failed_log_path(split):
    return OUT_DIR / f"composition_{split}_sentinel2_daily_timeseries_failed_ids.txt"


# =========================================================
# TIME RANGE - identical to scripts/download_sentinel2_daily_statistical.py
# =========================================================

START = "2025-01-01"
END = "2025-12-31"
AGGREGATION_INTERVAL = "P1D"

# =========================================================
# AUTHENTICATION
# =========================================================

load_env_file()
DIRECT_CLIENT_ID = os.environ.get("CDSE_CLIENT_ID")
DIRECT_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET")
if not DIRECT_CLIENT_ID or not DIRECT_CLIENT_SECRET:
    raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET missing from .env")

SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

CDSE_S2L2A = DataCollection.SENTINEL2_L2A.define_from(name="s2l2a_cdse", service_url=SH_BASE_URL)

# =========================================================
# SETTINGS - identical to scripts/download_sentinel2_daily_statistical.py
# =========================================================

MAX_RETRIES = 12
SLEEP_BETWEEN_REQUESTS = 0.5
SAVE_EVERY_N_STANDS = 10
SKIP_ALREADY_PROCESSED = True

# =========================================================
# EVALSCRIPT - byte-identical to scripts/download_sentinel2_daily_statistical.py
# (10 reflectance bands + NDVI/NDRE/NDMI/NBR/EVI/CIRE, SCL 4/5/6 cloud mask)
# =========================================================

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
    output: [
      { id: "bands", bands: 10, sampleType: "FLOAT32" },
      { id: "indices", bands: 6, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1, sampleType: "UINT8" }
    ]
  };
}

function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);

  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  let ndre = (sample.B8A - sample.B05) / (sample.B8A + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);

  let evi = 2.5 * ((sample.B08 - sample.B04) /
            (sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0 + 1e-6));

  let cire = (sample.B08 / (sample.B05 + 1e-6)) - 1.0;

  if (good) {
    return {
      bands: [
        sample.B02, sample.B03, sample.B04, sample.B05, sample.B06,
        sample.B07, sample.B08, sample.B8A, sample.B11, sample.B12
      ],
      indices: [ndvi, ndre, ndmi, nbr, evi, cire],
      dataMask: [1]
    };
  } else {
    return {
      bands: [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
      indices: [NaN, NaN, NaN, NaN, NaN, NaN],
      dataMask: [0]
    };
  }
}
"""

# =========================================================
# HELPER FUNCTIONS - same logic as scripts/download_sentinel2_daily_statistical.py
# =========================================================


def load_processed_ids(output_csv, id_col="id"):
    if not os.path.exists(output_csv):
        return set()
    try:
        df_existing = pd.read_csv(output_csv, usecols=[id_col])
        processed_ids = set(df_existing[id_col].dropna().astype(str).unique())
        print(f"Found {len(processed_ids)} already-processed IDs in the existing CSV.")
        return processed_ids
    except Exception as e:
        print(f"Could not read the existing CSV: {e}")
        return set()


def write_results(df_part, output_path, first_write):
    if df_part.empty:
        return
    mode = "w" if first_write else "a"
    df_part.to_csv(output_path, mode=mode, header=first_write, index=False, encoding="utf-8-sig")


def download_split(gdf_all, split, config):
    gdf = apply_filters(gdf_all, subregion=None, split=split)
    if gdf.empty:
        print(f"\n{split}: no parcels, skipping.")
        return

    out_csv = str(output_path(split))
    failed_log = str(failed_log_path(split))

    print(f"\n{'=' * 60}\nSPLIT: {split} ({len(gdf)} parcels)\n{'=' * 60}")

    processed_ids = set()
    if SKIP_ALREADY_PROCESSED and os.path.exists(out_csv):
        processed_ids = load_processed_ids(out_csv, id_col="id")
    first_write = not os.path.exists(out_csv)

    results_buffer = []
    failed_ids = []
    processed_count = 0
    skipped_count = 0
    total_objects = len(gdf)

    for i, row in enumerate(gdf.itertuples(index=False), start=1):
        row = row._asdict()
        row_id = row["id"]
        row_id_str = str(row_id)

        if SKIP_ALREADY_PROCESSED and row_id_str in processed_ids:
            skipped_count += 1
            continue

        try:
            geom = Geometry(row["geometry"], CRS.WGS84)
            data, config = request_with_retry(
                row_id, geom, config, EVALSCRIPT, (START, END), AGGREGATION_INTERVAL, CDSE_S2L2A,
                refresh_config_fn=lambda: get_config(DIRECT_CLIENT_ID, DIRECT_CLIENT_SECRET, SH_BASE_URL, SH_TOKEN_URL),
                max_retries=MAX_RETRIES,
            )

            if data is not None:
                parsed = parse_stats(data, row)
                results_buffer.extend(parsed)
                processed_count += 1
                processed_ids.add(row_id_str)
                print(f"OK {processed_count}/{total_objects} ({split}): {row_id} | rows added: {len(parsed)}")
            else:
                failed_ids.append(row_id_str)
                print(f"FAILED {i}/{total_objects} ({split}): {row_id}")

            if processed_count > 0 and processed_count % SAVE_EVERY_N_STANDS == 0 and results_buffer:
                write_results(pd.DataFrame(results_buffer), out_csv, first_write)
                first_write = False
                results_buffer = []

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"Error {row_id} ({split}): {e}")
            failed_ids.append(row_id_str)

    if results_buffer:
        write_results(pd.DataFrame(results_buffer), out_csv, first_write)

    if failed_ids:
        with open(failed_log, "w", encoding="utf-8") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")
        print(f"Failed IDs saved: {failed_log}")

    print(f"\n{split} DONE - processed: {processed_count}, skipped (already done): {skipped_count}, failed: {len(failed_ids)}")
    return config


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gdf = gpd.read_file(PARCEL_FILE)
    print("Columns:", gdf.columns.tolist())
    print("Input CRS:", gdf.crs)
    print("Total parcels:", len(gdf))

    if ID_COL not in gdf.columns:
        raise ValueError(f"Missing ID column: {ID_COL}")
    gdf = gdf.rename(columns={ID_COL: "id", TARGET_COL: "target"})
    if "subregion" not in gdf.columns:
        gdf["subregion"] = "Subregion2"

    print(gdf.groupby(["subregion", "dataset_split", "puistu_tyyp"]).size())

    if gdf.crs is None:
        raise RuntimeError("Input file has no CRS.")
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(4326)
    print("CRS after reprojection:", gdf.crs)

    config = get_config(DIRECT_CLIENT_ID, DIRECT_CLIENT_SECRET, SH_BASE_URL, SH_TOKEN_URL)
    check_token(config)

    for split in SPLITS:
        config = download_split(gdf, split, config)

    print("\nALL SPLITS DONE.")


if __name__ == "__main__":
    main()
