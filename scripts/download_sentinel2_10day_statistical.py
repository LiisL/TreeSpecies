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
OUTPUT = str(PATHS.intermediate / "Subregion2_test_sentinel2_10day_timeseries.csv")

ID_COL = "id_left"

# Filter if desired
SELECT_SUBREGION = "Subregion2"   # e.g. "Subregion2" or None
SELECT_SPLIT = "test"            # e.g. "train", "val", "test" or None

# =========================================================
# TIME RANGE
# =========================================================

START = "2025-01-01"
END   = "2025-12-31"

# =========================================================
# API / DOWNLOAD CONFIG
# =========================================================

MAX_RETRIES = 10
SLEEP_BETWEEN_REQUESTS = 0.15

DEBUG = False
DEBUG_N_PER_SPLIT = 5

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
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);

  let evi = 2.5 * ((sample.B08 - sample.B04) / (sample.B08 + 6.0 * sample.B04 - 7.5 * sample.B02 + 1.0 + 1e-6));
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

    if DEBUG:
        if "dataset_split" in gdf.columns:
            gdf = gdf.groupby("dataset_split", group_keys=False).sample(
                n=min(DEBUG_N_PER_SPLIT, gdf.groupby("dataset_split").size().min()),
                random_state=42
            )
        else:
            gdf = gdf.sample(n=min(20, len(gdf)), random_state=42)

    print("Total objects:", len(gdf))

    # Reproject to WGS84 for the API
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(4326)

    print("CRS after reprojection:", gdf.crs)
    print("Bounds after reprojection:", gdf.total_bounds)

    # create config once
    config = get_config(DIRECT_CLIENT_ID, DIRECT_CLIENT_SECRET, SH_BASE_URL, SH_TOKEN_URL)

    # test token right away
    check_token(config)

    results = []

    for i, row in enumerate(gdf.itertuples(index=False), start=1):
        row = row._asdict()
        row_id = row["id"]

        geom = Geometry(row["geometry"], CRS.WGS84)
        data, config = request_with_retry(
            row_id, geom, config,
            EVALSCRIPT, (START, END), "P10D", CDSE_S2L2A,
            refresh_config_fn=lambda: get_config(DIRECT_CLIENT_ID, DIRECT_CLIENT_SECRET, SH_BASE_URL, SH_TOKEN_URL),
            max_retries=MAX_RETRIES
        )

        if data is not None:
            parsed = parse_stats(data, row)
            results.extend(parsed)
            print(f"OK {i}/{len(gdf)}: {row_id}")
        else:
            print(f"FAILED {i}/{len(gdf)}: {row_id}")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if i % 100 == 0:
            df_tmp = pd.DataFrame(results)
            df_tmp.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
            print(f"Intermediate save: {OUTPUT} | rows: {len(df_tmp)}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\nDONE")
    print(f"Rows written: {len(df)}")
    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
