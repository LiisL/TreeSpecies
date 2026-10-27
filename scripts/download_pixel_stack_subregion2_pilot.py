"""Downloads Sentinel-2 pixel-level raster stacks for ONE 25km x 25km pilot
tile within Subregion 2, across all 26 ten-day windows EXP032 was actually
trained on (after empty-column removal - the model's real feature_cols span
2025-01-31 to 2025-10-18, not the full nominal 31-window range).

Pilot tile chosen as the highest-forest-fraction full-size tile in Subregion
2's 25km tiling grid (61.2% forest, per data/raw/metsamask.tif): bbox
(486678, 6506098, 511678, 6531098) in EPSG:3301.

This is a copy of scripts/download_pixel_stack.py's evalscript/download logic
(same 14-band output: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,NDVI,NDRE,NDMI,NBR),
parameterized for this tile's bbox and the full 26-date list instead of the
original 7-date "first test" subset and the small pixel_test_area AOI.
"""

import os
import sys
import time
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS


def load_env_file(env_path=None):
    """Minimal .env loader - avoids importing src.sentinel_api.auth, which
    pulls in the `sentinelhub` package (not installed in this environment)."""
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
    r = requests.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        },
        timeout=timeout
    )
    print("TOKEN STATUS:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        raise RuntimeError("Failed to obtain access token.")
    return r.json()["access_token"]

# =========================================================
# INPUTS
# =========================================================

OUT_DIR = str(PATHS.intermediate / "pixel_stack_rasters_subregion2_pilot")

load_env_file()

CLIENT_ID = os.environ.get("CDSE_CLIENT_ID")
CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    raise RuntimeError(
        "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are missing. Create a .env file "
        "at the repo root (see .env.example) or set them as environment variables."
    )

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# =========================================================
# CONFIG
# =========================================================

# Pilot tile bbox (EPSG:3301) - highest-forest-fraction 25km tile in
# Subregion 2's tiling grid (tx=1, ty=2), 61.2% forest per metsamask.tif.
TILE_BBOX = [486678.0, 6506098.0, 511678.0, 6531098.0]

# The 26 ten-day windows that actually survive in EXP032's trained
# feature_cols (extracted directly from the saved model bundle - some
# early-winter windows were dropped during training as all-NaN).
DATES = [
    "2025-01-31", "2025-02-10", "2025-02-20", "2025-03-02", "2025-03-12",
    "2025-04-01", "2025-04-11", "2025-04-21", "2025-05-01", "2025-05-11",
    "2025-05-21", "2025-05-31", "2025-06-10", "2025-06-20", "2025-06-30",
    "2025-07-10", "2025-07-20", "2025-07-30", "2025-08-09", "2025-08-19",
    "2025-08-29", "2025-09-08", "2025-09-18", "2025-09-28", "2025-10-08",
    "2025-10-18",
]

WINDOW_DAYS = 10
RESOLUTION = 10  # meters
TARGET_EPSG = 3301

BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B11", "B12",
    "NDVI", "NDRE", "NDMI", "NBR"
]

# =========================================================
# EVALSCRIPT (identical to scripts/download_pixel_stack.py)
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
    output: {
      bands: 14,
      sampleType: "FLOAT32",
      nodataValue: -9999
    }
  };
}

function evaluatePixel(sample) {
  let good = sample.dataMask == 1 && (sample.SCL == 4 || sample.SCL == 5 || sample.SCL == 6);

  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-6);
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-6);
  let ndmi = (sample.B08 - sample.B11) / (sample.B08 + sample.B11 + 1e-6);
  let nbr  = (sample.B08 - sample.B12) / (sample.B08 + sample.B12 + 1e-6);

  if (good) {
    return [
      sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07,
      sample.B08, sample.B8A, sample.B11, sample.B12,
      ndvi, ndre, ndmi, nbr
    ];
  } else {
    return [
      -9999, -9999, -9999, -9999, -9999, -9999,
      -9999, -9999, -9999, -9999,
      -9999, -9999, -9999, -9999
    ];
  }
}
"""

# =========================================================
# FUNCTIONS
# =========================================================

def tile_size_px(bbox):
    minx, miny, maxx, maxy = bbox
    width = int(round((maxx - minx) / RESOLUTION))
    height = int(round((maxy - miny) / RESOLUTION))
    return width, height


def date_window(date_str):
    start = datetime.strptime(date_str, "%Y-%m-%d")
    end = start + timedelta(days=WINDOW_DAYS)
    return (
        start.strftime("%Y-%m-%dT00:00:00Z"),
        end.strftime("%Y-%m-%dT23:59:59Z")
    )


def make_process_payload(bbox, width, height, start_time, end_time):
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{TARGET_EPSG}"}
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {"from": start_time, "to": end_time},
                        "maxCloudCoverage": 30,
                        "mosaickingOrder": "leastCC"
                    }
                }
            ]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
        },
        "evalscript": EVALSCRIPT
    }


def get_token():
    return fetch_access_token(CLIENT_ID, CLIENT_SECRET, TOKEN_URL)


def download_one_date(token, date_str, bbox, width, height, retries=3):
    start_time, end_time = date_window(date_str)
    out_tif = os.path.join(OUT_DIR, f"pixel_stack_{date_str}.tif")

    if os.path.exists(out_tif):
        print(f"Already exists, skipping: {out_tif}")
        return out_tif

    payload = make_process_payload(bbox, width, height, start_time, end_time)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "image/tiff"
    }

    print(f"\nDownloading date: {date_str} (window {start_time} to {end_time})")

    for attempt in range(1, retries + 1):
        r = requests.post(PROCESS_URL, headers=headers, json=payload, timeout=300)
        print("STATUS:", r.status_code, f"(attempt {attempt}/{retries})")

        if r.status_code == 401:
            raise RuntimeError("401 Unauthorized. Token expired or invalid - restart the script to refetch.")

        if r.status_code == 429:
            wait = 30 * attempt
            print(f"Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue

        if r.status_code >= 400:
            print(r.text[:1000])
            if attempt == retries:
                raise RuntimeError(f"Process API error: {r.status_code} after {retries} attempts")
            time.sleep(10 * attempt)
            continue

        with open(out_tif, "wb") as f:
            f.write(r.content)
        print("Saved:", out_tif, f"({os.path.getsize(out_tif)/1e6:.1f} MB)")
        return out_tif

    raise RuntimeError(f"Failed to download {date_str} after {retries} attempts")


def save_band_metadata():
    meta_csv = os.path.join(OUT_DIR, "band_order.csv")
    df = pd.DataFrame({
        "band_index": list(range(1, len(BAND_NAMES) + 1)),
        "band_name": BAND_NAMES
    })
    df.to_csv(meta_csv, index=False, encoding="utf-8-sig")
    print("Band order saved:", meta_csv)


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    width, height = tile_size_px(TILE_BBOX)
    print("Pilot tile bbox (EPSG:3301):", TILE_BBOX)
    print("Raster size:", width, "x", height, f"({width*height/1e6:.1f} megapixels per band)")
    print("Dates to download:", len(DATES))

    token = get_token()
    save_band_metadata()

    for i, date_str in enumerate(DATES, start=1):
        print(f"\n=== [{i}/{len(DATES)}] ===")
        try:
            download_one_date(token, date_str, TILE_BBOX, width, height)
        except RuntimeError as e:
            if "401" in str(e):
                print("Refreshing token...")
                token = get_token()
                download_one_date(token, date_str, TILE_BBOX, width, height)
            else:
                print(f"Error on date {date_str}: {e}")
        time.sleep(1)

    print("\nDone.")
    print("Rasters are in folder:", OUT_DIR)


if __name__ == "__main__":
    main()
