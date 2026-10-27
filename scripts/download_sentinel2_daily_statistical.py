import os
import sys
import time
from pathlib import Path
import geopandas as gpd
import pandas as pd

from sentinelhub import (
    DataCollection,
    Geometry,
    CRS
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sentinel_api.auth import get_config, check_token, load_env_file
from src.sentinel_api.statistical_client import (
    apply_filters,
    request_with_retry,
    parse_stats,
)
from src.paths import PATHS

# =========================================================
# INPUT / OUTPUT
# =========================================================

INPUT = str(PATHS.processed / "final_split_clean.gpkg")
OUTPUT = str(PATHS.intermediate / "Subregion2_val_sentinel2_daily_timeseries.csv")

FAILED_LOG = OUTPUT.replace(".csv", "_failed_ids.txt")

ID_COL = "id_left"

# =========================================================
# FILTERS
# =========================================================

SELECT_SUBREGION = "Subregion2"   # e.g. "Subregion2" or None
SELECT_SPLIT = "val"             # e.g. "train", "val", "test" or None

# =========================================================
# TIME RANGE
# =========================================================

START = "2025-01-01"
END   = "2025-12-31"

# Daily query
AGGREGATION_INTERVAL = "P1D"

# =========================================================
# AUTHENTICATION - from .env file (see .env.example)
# =========================================================

load_env_file()

DIRECT_CLIENT_ID = os.environ.get("CDSE_CLIENT_ID")
DIRECT_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET")

if not DIRECT_CLIENT_ID or not DIRECT_CLIENT_SECRET:
    raise RuntimeError(
        "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are missing. Create a .env file "
        "at the repo root (see .env.example) or set them as environment variables."
    )

SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

CDSE_S2L2A = DataCollection.SENTINEL2_L2A.define_from(
    name="s2l2a_cdse",
    service_url=SH_BASE_URL
)

# =========================================================
# SETTINGS
# =========================================================

MAX_RETRIES = 12
SLEEP_BETWEEN_REQUESTS = 0.5

SAVE_EVERY_N_STANDS = 10

OVERWRITE_OUTPUT = False
SKIP_ALREADY_PROCESSED = True

DEBUG = False
DEBUG_N_PER_SPLIT = 5

# =========================================================
# EVALSCRIPT
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
# HELPER FUNCTIONS
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
    header = first_write

    df_part.to_csv(
        output_path,
        mode=mode,
        header=header,
        index=False,
        encoding="utf-8-sig"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    gdf = gpd.read_file(INPUT)

    print("Columns:")
    print(gdf.columns.tolist())
    print("Input CRS:", gdf.crs)
    print("Bounds before reprojection:", gdf.total_bounds)

    if ID_COL not in gdf.columns:
        raise ValueError(f"Missing ID column: {ID_COL}")

    gdf = gdf.rename(columns={ID_COL: "id"})

    gdf = apply_filters(gdf, SELECT_SUBREGION, SELECT_SPLIT)

    if gdf.empty:
        raise ValueError("No objects left after filtering.")

    print("\nSelected subregions:")
    print(sorted(gdf["subregion"].dropna().unique().tolist()))

    print("\nSelected splits:")
    print(sorted(gdf["dataset_split"].dropna().unique().tolist()))

    print("\nTotal objects:", len(gdf))
    print(gdf.groupby(["subregion", "dataset_split"]).size())

    if DEBUG:
        if "dataset_split" in gdf.columns:
            groups = []
            for _, part in gdf.groupby("dataset_split"):
                n = min(DEBUG_N_PER_SPLIT, len(part))
                groups.append(part.sample(n=n, random_state=42))
            gdf = pd.concat(groups, ignore_index=True)
        else:
            gdf = gdf.sample(n=min(DEBUG_N_PER_SPLIT, len(gdf)), random_state=42)

        print("\nDEBUG objects:", len(gdf))

    if gdf.crs is None:
        raise RuntimeError("Input file has no CRS.")

    # WGS84 for Sentinel Hub Geometry
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(4326)

    print("CRS after reprojection:", gdf.crs)
    print("Bounds after reprojection:", gdf.total_bounds)

    if OVERWRITE_OUTPUT and os.path.exists(OUTPUT):
        os.remove(OUTPUT)

    if os.path.exists(FAILED_LOG):
        os.remove(FAILED_LOG)

    processed_ids = set()

    if SKIP_ALREADY_PROCESSED and os.path.exists(OUTPUT):
        processed_ids = load_processed_ids(OUTPUT, id_col="id")

    first_write = not os.path.exists(OUTPUT)

    config = get_config(DIRECT_CLIENT_ID, DIRECT_CLIENT_SECRET, SH_BASE_URL, SH_TOKEN_URL)
    check_token(config)

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
            print(f"Skip already processed ({skipped_count}): {row_id}")
            continue

        try:
            geom = Geometry(row["geometry"], CRS.WGS84)

            data, config = request_with_retry(
                row_id, geom, config,
                EVALSCRIPT, (START, END), AGGREGATION_INTERVAL, CDSE_S2L2A,
                refresh_config_fn=lambda: get_config(DIRECT_CLIENT_ID, DIRECT_CLIENT_SECRET, SH_BASE_URL, SH_TOKEN_URL),
                max_retries=MAX_RETRIES
            )

            if data is not None:
                parsed = parse_stats(data, row)
                results_buffer.extend(parsed)

                processed_count += 1
                processed_ids.add(row_id_str)

                print(f"OK {processed_count}/{total_objects}: {row_id} | rows added: {len(parsed)}")
            else:
                failed_ids.append(row_id_str)
                print(f"FAILED {i}/{total_objects}: {row_id}")

            if processed_count > 0 and processed_count % SAVE_EVERY_N_STANDS == 0 and results_buffer:
                df_part = pd.DataFrame(results_buffer)
                write_results(df_part, OUTPUT, first_write)
                first_write = False
                results_buffer = []
                print(f"Intermediate save done: {OUTPUT}")

            time.sleep(SLEEP_BETWEEN_REQUESTS)

        except Exception as e:
            print(f"Error {row_id}: {e}")
            failed_ids.append(row_id_str)

    if results_buffer:
        df_part = pd.DataFrame(results_buffer)
        write_results(df_part, OUTPUT, first_write)

    if failed_ids:
        with open(FAILED_LOG, "w", encoding="utf-8") as f:
            for fid in failed_ids:
                f.write(f"{fid}\n")

        print(f"Failed IDs saved: {FAILED_LOG}")

    print("\nDONE")
    print(f"Processed this run: {processed_count}")
    print(f"Skipped already processed: {skipped_count}")
    print(f"Failed: {len(failed_ids)}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
