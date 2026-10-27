"""Adds per-parcel Copernicus Tree Cover Density (TCD) statistics to the
Subregion2 parcel table, as a structural/canopy-density feature - motivated
by the LV (grey alder) age-invariance finding (script 09): LV's near-flat
young/mature accuracy gap traces to a structural feature (height) that
doesn't need canopy maturity to express, unlike the spectral/phenological
signatures most species (especially MA/pine, the worst age gap) rely on.
Canopy density is a second, more general structural signal worth testing
the same way.

Source: Copernicus Data Space Ecosystem STAC catalog, collection
`clms_vlcc_tree-cover-density_europe_10m_yearly_v1` (10m, EPSG:3035,
updated yearly). 2023 is the newest vintage covering our study area (a
~2-year gap to the 2025 Sentinel-2 imagery, vs. 7 years for the older,
free-without-auth 2018 EEA mirror - documented as a known limitation
either way, since TCD is a snapshot and stands can change).

6 tiles needed to cover Subregion2 (100km x 100km each, EPSG:3035 grid) -
found by searching the STAC catalog with Subregion2's ACTUAL bounding box
(23.33-24.50 E, 58.24-59.47 N; a first attempt guessed a bbox from memory
without checking the real parcel extent and downloaded 4 entirely wrong,
non-overlapping tiles - always derive the search bbox from the real data,
never assume): E50N39, E50N40, E50N41, E51N39, E51N40, E51N41 - downloaded
via the CDSE OData API
(password-grant OAuth, `cdse-public` client - NOT the client_credentials
flow used for Sentinel Hub elsewhere in this project; the OData download
API requires a real user login token, confirmed empirically after a
`client_credentials` token was rejected with "Token audience not allowed").
Saved to data/raw/tcd_2023/ (not re-downloaded by this script - a one-time
manual step, since it needed an interactively-obtained user token).

Value semantics (from the STAC item's `classification:classes`): 0 = "all
non-tree covered areas", 255 = "outside_area" (nodata sentinel - values
1-100 are the percent tree cover density for actual tree-covered pixels;
note 0 is a REAL value ("no tree cover"), not missing data).

Outputs:
  - tables/subregion2_parcel_tcd.csv
        one row per parcel: id, tcd_mean/median/std/min/max/pixel_count
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.paths import PATHS

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "tables"

STANDS_FILE = PATHS.processed / "final_split_clean.gpkg"
TCD_DIR = PATHS.raw / "tcd_2023"
TCD_TILES = [
    "CLMS_HRLVLCC_TCD_S2023_R10m_E50N39_03035_V01_R00.tif",
    "CLMS_HRLVLCC_TCD_S2023_R10m_E50N40_03035_V01_R00.tif",
    "CLMS_HRLVLCC_TCD_S2023_R10m_E50N41_03035_V01_R00.tif",
    "CLMS_HRLVLCC_TCD_S2023_R10m_E51N39_03035_V01_R00.tif",
    "CLMS_HRLVLCC_TCD_S2023_R10m_E51N40_03035_V01_R00.tif",
    "CLMS_HRLVLCC_TCD_S2023_R10m_E51N41_03035_V01_R00.tif",
]

ID_COL = "id_left"
TCD_NODATA = 255

PARCEL_TABLE_OUT = TABLES_DIR / "subregion2_parcel_tcd.csv"
MOSAIC_OUT = TCD_DIR / "tcd_2023_subregion2_mosaic.tif"


def load_subregion2_parcels():
    gdf = gpd.read_file(STANDS_FILE)
    gdf = gdf[gdf["subregion"] == "Subregion2"].copy()
    print(f"Subregion2 parcels in {STANDS_FILE.name}: {len(gdf)}")
    return gdf


def build_mosaic():
    """Merges the 4 downloaded tiles into a single raster covering
    Subregion2 (parcels can straddle tile boundaries, so a single mosaic
    avoids having to split zonal_stats calls per-tile per-parcel).
    """
    srcs = [rasterio.open(TCD_DIR / t) for t in TCD_TILES]
    mosaic_arr, mosaic_transform = merge(srcs, nodata=TCD_NODATA)

    meta = srcs[0].meta.copy()
    meta.update({
        "height": mosaic_arr.shape[1],
        "width": mosaic_arr.shape[2],
        "transform": mosaic_transform,
        "nodata": TCD_NODATA,
    })

    with rasterio.open(MOSAIC_OUT, "w", **meta) as dst:
        dst.write(mosaic_arr)

    for s in srcs:
        s.close()

    print(f"Mosaic saved: {MOSAIC_OUT} (shape {mosaic_arr.shape})")
    return MOSAIC_OUT


def compute_tcd_stats(gdf, mosaic_path):
    with rasterio.open(mosaic_path) as src:
        crs = src.crs

    gdf_proj = gdf.to_crs(crs)

    print("Computing TCD zonal statistics (mean, median, std, min, max)...")
    stats = zonal_stats(
        gdf_proj,
        mosaic_path,
        stats=["mean", "median", "std", "min", "max", "count"],
        nodata=TCD_NODATA,
        all_touched=False,
    )

    df = pd.DataFrame(stats)
    df = df.rename(columns={
        "mean": "tcd_mean",
        "median": "tcd_median",
        "std": "tcd_std",
        "min": "tcd_min",
        "max": "tcd_max",
        "count": "tcd_pixel_count",
    })
    return df[["tcd_mean", "tcd_median", "tcd_std", "tcd_min", "tcd_max", "tcd_pixel_count"]]


def build_parcel_table():
    gdf = load_subregion2_parcels()
    mosaic_path = build_mosaic()
    tcd_df = compute_tcd_stats(gdf, mosaic_path)

    out = gdf[[ID_COL, "target", "dataset_split", "subregion"]].reset_index(drop=True)
    out = out.rename(columns={ID_COL: "id", "dataset_split": "split"})
    out = pd.concat([out, tcd_df.reset_index(drop=True)], axis=1)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PARCEL_TABLE_OUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved parcel-level TCD table: {PARCEL_TABLE_OUT}")
    print(out.groupby("split").size())
    print("\nMissing TCD values per split:")
    print(out.groupby("split")[["tcd_mean", "tcd_median"]].apply(lambda d: d.isna().sum()))
    print("\nTCD summary stats (all parcels):")
    print(out[["tcd_mean", "tcd_median", "tcd_std"]].describe())

    return out


if __name__ == "__main__":
    build_parcel_table()
    print("\nDone.")
