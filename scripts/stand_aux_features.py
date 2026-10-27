import os
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# =========================================================
# INPUTS
# =========================================================

STANDS_FILE = PATHS.processed / "final_split_clean.gpkg"

SOIL_RASTER = PATHS.raw / "soil_siffer_10m.tif"
ELEV_RASTER = PATHS.raw / "H_20_21_22_23.tif"

OUTPUT_CSV = PATHS.intermediate / "stand_aux_features0806.csv"

# =========================================================
# FILTERS
# =========================================================

SELECT_SUBREGION = None
SELECT_SPLIT = None

# =========================================================
# OUTPUT COLUMN NAMES
# =========================================================

ID_COL = "id_left"
TARGET_COL = "target"
SPLIT_COL = "dataset_split"
SUBREGION_COL = "subregion"

# =========================================================
# ZONAL STATS CONFIG
# =========================================================

# If True, includes every pixel touched by the polygon.
# This is usually better for small/narrow stands.
ALL_TOUCHED = True

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_filter_to_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def apply_filters(gdf):
    subregions = normalize_filter_to_list(SELECT_SUBREGION)
    splits = normalize_filter_to_list(SELECT_SPLIT)

    if subregions is not None:
        gdf = gdf[gdf[SUBREGION_COL].isin(subregions)].copy()

    if splits is not None:
        gdf = gdf[gdf[SPLIT_COL].isin(splits)].copy()

    return gdf


def print_raster_info(path, name):
    with rasterio.open(path) as src:
        print(f"\n{name} raster info:")
        print(f"  CRS: {src.crs}")
        print(f"  Resolution: {src.res}")
        print(f"  Width x Height: {src.width} x {src.height}")
        print(f"  Nodata: {src.nodata}")
        print(f"  Dtype: {src.dtypes[0]}")
        return src.crs, src.nodata


def main():
    gdf = gpd.read_file(STANDS_FILE)

    needed = [ID_COL, TARGET_COL, SPLIT_COL, SUBREGION_COL, "geometry"]
    missing = [c for c in needed if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if gdf.crs is None:
        raise RuntimeError("Stands file has no CRS.")

    gdf = apply_filters(gdf)

    if gdf.empty:
        raise RuntimeError("No objects left after filtering.")

    print("Total objects:", len(gdf))
    print("Stands CRS:", gdf.crs)
    print(gdf.groupby([SUBREGION_COL, SPLIT_COL]).size())

    # -----------------------------------------------------
    # Elevation raster info
    # -----------------------------------------------------
    elev_crs, elev_nodata = print_raster_info(ELEV_RASTER, "Elevation")

    gdf_elev = gdf.to_crs(elev_crs)

    print("\nComputing elevation features...")
    elev_stats = zonal_stats(
        vectors=gdf_elev,
        raster=ELEV_RASTER,
        stats=["count", "mean", "median", "std", "min", "max"],
        nodata=elev_nodata,
        all_touched=ALL_TOUCHED
    )

    elev_df = pd.DataFrame(elev_stats).rename(columns={
        "count": "elev_pixel_count",
        "mean": "elev_mean",
        "median": "elev_median",
        "std": "elev_std",
        "min": "elev_min",
        "max": "elev_max"
    })

    # -----------------------------------------------------
    # Soil raster info
    # -----------------------------------------------------
    soil_crs, soil_nodata = print_raster_info(SOIL_RASTER, "Soil")

    gdf_soil = gdf.to_crs(soil_crs)

    print("\nComputing soil features...")
    soil_stats = zonal_stats(
        vectors=gdf_soil,
        raster=SOIL_RASTER,
        stats=["count", "majority"],
        nodata=soil_nodata,
        all_touched=ALL_TOUCHED
    )

    soil_df = pd.DataFrame(soil_stats).rename(columns={
        "count": "soil_pixel_count",
        "majority": "soil_main"
    })

    # -----------------------------------------------------
    # Build output
    # -----------------------------------------------------
    out_df = gdf[[ID_COL, TARGET_COL, SPLIT_COL, SUBREGION_COL]].copy()
    out_df = out_df.rename(columns={
        ID_COL: "id",
        TARGET_COL: "target",
        SPLIT_COL: "split",
        SUBREGION_COL: "subregion"
    })

    out_df = pd.concat(
        [
            out_df.reset_index(drop=True),
            elev_df.reset_index(drop=True),
            soil_df.reset_index(drop=True)
        ],
        axis=1
    )

    # convert soil code to string
    out_df["soil_main"] = out_df["soil_main"].astype("Int64").astype(str)
    out_df.loc[out_df["soil_main"] == "<NA>", "soil_main"] = None

    # -----------------------------------------------------
    # Checks
    # -----------------------------------------------------
    print("\nElevation feature summary:")
    print(out_df[[
        "elev_pixel_count", "elev_mean", "elev_median", "elev_std", "elev_min", "elev_max"
    ]].describe())

    print("\nSoil pixel count summary:")
    print(out_df["soil_pixel_count"].describe())

    print("\nExamples of stands with very few pixels:")
    print(
        out_df.loc[out_df["elev_pixel_count"].fillna(0) <= 3,
                   ["id", "target", "split", "subregion", "elev_pixel_count", "elev_mean", "elev_median"]]
        .head(20)
    )

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nDone: {OUTPUT_CSV}")
    print("\nSample:")
    print(out_df.head())


if __name__ == "__main__":
    main()
