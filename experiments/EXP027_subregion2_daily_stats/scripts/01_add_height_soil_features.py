"""
Adds per-parcel height (mean/median/std/min/max) and soil (majority code)
features to the Subregion2 daily Sentinel-2 time series (train/val/test).

Height source: data/raw/H_20_21_22_23.tif (elevation/height raster, EPSG:3301)
Soil source:   data/raw/soil_siffer_10m.tif (soil "siffer" code raster, EPSG:3301)
Parcel geometries: data/processed/final_split_clean.gpkg (id_left matches the
    daily CSVs' "id" column almost exactly - confirmed empirically: 1613/1613
    Subregion2 test IDs from the daily CSV are present in this file).

Outputs:
  - tables/subregion2_parcel_height_soil.csv
        one row per parcel: id, split, target, subregion, height stats, soil code
  - data/processed/Subregion2_{split}_sentinel2_daily_timeseries_with_height_soil.csv
        original daily CSV + the height/soil columns broadcast onto every row
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.paths import PATHS

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "tables"

STANDS_FILE = PATHS.processed / "final_split_clean.gpkg"
HEIGHT_RASTER = PATHS.raw / "H_20_21_22_23.tif"
SOIL_RASTER = PATHS.raw / "soil_siffer_10m.tif"

ID_COL = "id_left"
SPLITS = ["train", "val", "test"]

DAILY_CSV = {
    split: PATHS.intermediate / f"Subregion2_{split}_sentinel2_daily_timeseries.csv"
    for split in SPLITS
}
ENRICHED_CSV = {
    split: PATHS.processed / f"Subregion2_{split}_sentinel2_daily_timeseries_with_height_soil.csv"
    for split in SPLITS
}

PARCEL_TABLE_OUT = TABLES_DIR / "subregion2_parcel_height_soil.csv"


def load_subregion2_parcels():
    gdf = gpd.read_file(STANDS_FILE)
    gdf = gdf[gdf["subregion"] == "Subregion2"].copy()
    print(f"Subregion2 parcels in {STANDS_FILE.name}: {len(gdf)}")
    return gdf


def compute_height_stats(gdf):
    with rasterio.open(HEIGHT_RASTER) as src:
        crs = src.crs
        nodata = src.nodata

    gdf_proj = gdf.to_crs(crs)

    print("Computing height zonal statistics (mean, median, std, min, max)...")
    stats = zonal_stats(
        gdf_proj,
        HEIGHT_RASTER,
        stats=["mean", "median", "std", "min", "max", "count"],
        nodata=nodata,
        all_touched=False,
    )

    # NOTE: rasterstats does NOT return dict keys in the order the `stats`
    # list was given (it has its own internal order) - rename by key name,
    # never positionally, or columns silently get mislabeled.
    df = pd.DataFrame(stats)
    df = df.rename(columns={
        "mean": "height_mean",
        "median": "height_median",
        "std": "height_std",
        "min": "height_min",
        "max": "height_max",
        "count": "height_pixel_count",
    })
    return df[["height_mean", "height_median", "height_std", "height_min", "height_max", "height_pixel_count"]]


def compute_soil_stats(gdf):
    with rasterio.open(SOIL_RASTER) as src:
        crs = src.crs
        nodata = src.nodata

    gdf_proj = gdf.to_crs(crs)

    print("Computing soil zonal statistics (majority code)...")
    stats = zonal_stats(
        gdf_proj,
        SOIL_RASTER,
        stats=["majority", "count"],
        nodata=nodata,
        all_touched=False,
    )

    df = pd.DataFrame(stats)
    df = df.rename(columns={"majority": "soil_majority", "count": "soil_pixel_count"})
    df["soil_majority"] = df["soil_majority"].astype("Int64")
    return df[["soil_majority", "soil_pixel_count"]]


def build_parcel_table():
    gdf = load_subregion2_parcels()

    height_df = compute_height_stats(gdf)
    soil_df = compute_soil_stats(gdf)

    out = gdf[[ID_COL, "target", "dataset_split", "subregion"]].reset_index(drop=True)
    out = out.rename(columns={
        ID_COL: "id",
        "dataset_split": "split",
    })
    out = pd.concat([out, height_df.reset_index(drop=True), soil_df.reset_index(drop=True)], axis=1)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PARCEL_TABLE_OUT, index=False, encoding="utf-8-sig")
    print(f"\nSaved parcel-level height/soil table: {PARCEL_TABLE_OUT}")
    print(out.groupby("split").size())
    print("\nMissing height/soil values per split:")
    print(out.groupby("split")[["height_mean", "height_median", "soil_majority"]].apply(lambda d: d.isna().sum()))

    return out


def merge_into_daily_csvs(parcel_df):
    aux_cols = [
        "id", "height_mean", "height_median", "height_std", "height_min", "height_max",
        "height_pixel_count", "soil_majority", "soil_pixel_count",
    ]
    aux = parcel_df[aux_cols].copy()
    aux["id"] = aux["id"].astype(float)

    for split in SPLITS:
        daily_path = DAILY_CSV[split]
        print(f"\nLoading daily time series: {daily_path.name}")
        daily = pd.read_csv(daily_path)
        daily["id"] = daily["id"].astype(float)

        before_cols = daily.shape[1]
        merged = daily.merge(aux, on="id", how="left")

        n_missing = merged["height_mean"].isna().sum()
        print(f"  Rows: {len(merged)}, columns {before_cols} -> {merged.shape[1]}")
        print(f"  Rows with no height/soil match: {n_missing} ({n_missing / len(merged):.2%})")

        out_path = ENRICHED_CSV[split]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parcel_df = build_parcel_table()
    merge_into_daily_csvs(parcel_df)
    print("\nDone.")
