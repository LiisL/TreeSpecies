"""Builds a NATIONWIDE parcel base dataset from the new Metsaregister export
that includes full stand composition (not just the dominant species) and
both pure (>=80% purity) and mixed stands - the dataset requested for the
EXP043-046 pixel-level experiment family, but built for all 8 subregions at
once so it's ready whenever training/downloading expands beyond Subregion 2.

Inputs (both delivered by the user, joined by `id`):
  - data/raw/data-1785007358127/data-1785007358127.csv: composition table
    (peapuuliik, liik_1..5/osakaal_1..5, puistu_tyyp, ...). Geometry is
    decoded directly from this file's own `ala` WKB column (EPSG:3301) -
    NOT from data/raw/eraldised.shp, since that shapefile may be an older
    snapshot (confirmed: eraldised.shp matches the OLD SubR2_pl.shp's ids
    at 97%, but the new CSV only matches those same old ids at 18%,
    meaning eraldised.shp and the new CSV are from different points in
    time; the CSV's own embedded geometry avoids that mismatch entirely).
  - data/raw/Subregions/Subregion{1-8}.shp: the project's 8 official
    subregion polygons (Subregion5 is stored in EPSG:4326, reprojected
    here to match the rest).

Rows with puistu_tyyp == "andmed puuduvad" (no composition breakdown
recorded) are dropped - confirmed to be exactly the rows missing
liik_1..5, and excluded per explicit user decision since composition is
central to why this dataset exists.

Species codes are normalized to the project's established 7-class scheme
(HB/KS/KU/LM/LV/MA/MUU) - all rare codes (TA, RE, SA, VA, PN, LH, JA, PP,
PI, NU, TO, TS, KP, TL, TM) collapse into MUU, matching every prior
experiment's label scheme.

Split methodology mirrors scripts/split.py EXACTLY (same 5km spatial-grid
grouping, same GroupShuffleSplit random_state=42, same 70/15/15
train/val/test proportions, split independently per subregion) so results
stay methodologically comparable to the rest of the project - but this is
a FRESH split over the new parcel set, not a re-application of the old
Subregion2_{train,val,test} split, since only ~18% of the old dataset's
ids survive into the new export (most stands have presumably been
re-inventoried/harvested/renumbered since the original query).

The minimum-area filter is 0.3ha here, NOT scripts/split.py's 1.0ha - this
new query skews much smaller (median 0.84ha vs. the old dataset's 1.82ha,
which had nothing below 1ha at all); keeping 1.0ha would have dropped 57%
of the newly available data, disproportionately the small mixed stands
this dataset exists to capture. 0.3ha ~= 30 Sentinel-2 pixels at 10m
resolution, chosen for pixel-level usability, not parity with the old
pipeline.

Height/soil auxiliary features (same rasters/stats as
src/feature_engineering/stand_aux_features.py) are computed here so the
basefile is immediately usable for future parcel-level (mean/median-style)
experiments on any subregion, without needing a separate aux-join pass.

Output is nationwide; EXP043 (Subregion 2 only, for now) should point its
PARCEL_FILE at a Subregion2 slice of this file (see
export_subregion2_for_exp043() at the bottom).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.wkb
import rasterio
from rasterstats import zonal_stats
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# =========================================================
# INPUTS
# =========================================================

COMPOSITION_CSV = PATHS.raw / "data-1785007358127" / "data-1785007358127.csv"
SUBREGION_FOLDER = PATHS.raw / "Subregions"
SOIL_RASTER = PATHS.raw / "soil_siffer_10m.tif"
ELEV_RASTER = PATHS.raw / "H_20_21_22_23.tif"

SRID = 3301

# =========================================================
# OUTPUTS - named to be clearly distinct from prior experiments'
# final_split.gpkg / final_split_clean.gpkg
# =========================================================

OUTPUT_GPKG = PATHS.processed / "composition_base_2026.gpkg"
OUTPUT_CSV = PATHS.processed / "composition_base_2026.csv"

EXP043_PARCEL_FILE = PATHS.raw / "composition_base_2026_subregion2.gpkg"

# =========================================================
# SPECIES NORMALIZATION (matches every prior experiment's 7-class scheme)
# =========================================================

KNOWN_SPECIES = {"HB", "KS", "KU", "LM", "LV", "MA"}
COMPOSITION_SPECIES_COLS = ["peapuuliik", "liik_1", "liik_2", "liik_3", "liik_4", "liik_5"]


def normalize_species(code):
    if pd.isna(code):
        return code
    return code if code in KNOWN_SPECIES else "MUU"


# =========================================================
# SPLIT (mirrors scripts/split.py exactly)
# =========================================================

GRID_SIZE = 5000
RANDOM_STATE = 42

# The old parcel-level pipeline (scripts/split.py) used 1.0ha here, but this
# new composition query skews much smaller (median 0.84ha vs. the old
# dataset's 1.82ha, which had nothing below 1ha at all) - keeping 1.0ha
# would drop 57% of the newly available data, disproportionately the small
# mixed stands this dataset exists to capture. 0.3ha = ~30 Sentinel-2
# pixels at 10m resolution, a floor chosen for pixel-level usability
# (enough pixels for a meaningful sample / partly-padded CNN patch)
# rather than for consistency with the parcel-mean/median pipeline.
MIN_AREA_HA = 0.3


def split_subregion(gdf):
    cent = gdf.geometry.centroid
    gx = (cent.x // GRID_SIZE).astype(int)
    gy = (cent.y // GRID_SIZE).astype(int)
    group = (gx.astype(str) + "_" + gy.astype(str)).to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(gdf, groups=group))

    train = gdf.iloc[train_idx].copy()
    temp = gdf.iloc[temp_idx].copy()
    temp_group = group[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(temp, groups=temp_group))

    val = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()

    train["dataset_split"] = "train"
    val["dataset_split"] = "val"
    test["dataset_split"] = "test"

    return pd.concat([train, val, test])


# =========================================================
# STEPS
# =========================================================

def load_composition():
    df = pd.read_csv(COMPOSITION_CSV)
    before = len(df)
    df = df[df["puistu_tyyp"] != "andmed puuduvad"].copy()
    print(f"Dropped {before - len(df)} rows with missing composition breakdown "
          f"(andmed puuduvad); {len(df)} remain")

    df["geometry"] = df["ala"].apply(lambda h: shapely.wkb.loads(bytes.fromhex(h)))
    df = df.drop(columns=["ala"])
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=f"EPSG:{SRID}")

    for col in COMPOSITION_SPECIES_COLS:
        gdf[col] = gdf[col].apply(normalize_species)

    gdf["osakaal"] = gdf["suurim_osakaal"]

    gdf["pindala_ha"] = gdf.geometry.area / 10000.0
    before = len(gdf)
    gdf = gdf[gdf["pindala_ha"] >= MIN_AREA_HA].copy()
    print(f"Dropped {before - len(gdf)} parcels below {MIN_AREA_HA} ha; {len(gdf)} remain")

    return gdf


def assign_and_split_by_subregion(gdf):
    sub_files = sorted(Path(SUBREGION_FOLDER).glob("*.shp"))
    all_results = []

    for sub_file in sub_files:
        name = sub_file.stem
        subregion = gpd.read_file(sub_file)
        if subregion.crs != gdf.crs:
            subregion = subregion.to_crs(gdf.crs)

        sub_gdf = gpd.sjoin(gdf, subregion[["geometry"]], predicate="intersects", how="inner")
        sub_gdf = sub_gdf.drop(columns=["index_right"])

        if len(sub_gdf) < 20:
            print(f"{name}: too few parcels ({len(sub_gdf)}), skipping")
            continue

        result = split_subregion(sub_gdf)
        result["subregion"] = name
        all_results.append(result)
        print(f"{name}: {len(result)} total -> {dict(result['dataset_split'].value_counts())}")

    final = pd.concat(all_results)
    return gpd.GeoDataFrame(final, geometry="geometry", crs=gdf.crs)


def add_aux_features(gdf):
    with rasterio.open(ELEV_RASTER) as src:
        elev_crs, elev_nodata = src.crs, src.nodata
    gdf_elev = gdf.to_crs(elev_crs)
    print("Computing elevation/height zonal stats (this can take a while)...")
    elev_stats = zonal_stats(
        gdf_elev, ELEV_RASTER,
        stats=["count", "mean", "median", "std", "min", "max"],
        nodata=elev_nodata, all_touched=True,
    )
    elev_df = pd.DataFrame(elev_stats).rename(columns={
        "count": "elev_pixel_count", "mean": "elev_mean", "median": "elev_median",
        "std": "elev_std", "min": "elev_min", "max": "elev_max",
    })

    with rasterio.open(SOIL_RASTER) as src:
        soil_crs, soil_nodata = src.crs, src.nodata
    gdf_soil = gdf.to_crs(soil_crs)
    print("Computing soil zonal stats...")
    soil_stats = zonal_stats(
        gdf_soil, SOIL_RASTER,
        stats=["count", "majority"],
        nodata=soil_nodata, all_touched=True,
    )
    soil_df = pd.DataFrame(soil_stats).rename(columns={
        "count": "soil_pixel_count", "majority": "soil_main",
    })

    gdf = gdf.reset_index(drop=True)
    out = pd.concat([gdf, elev_df.reset_index(drop=True), soil_df.reset_index(drop=True)], axis=1)
    out["soil_main"] = out["soil_main"].astype("Int64").astype(str)
    out.loc[out["soil_main"] == "<NA>", "soil_main"] = None

    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def export_subregion2_for_exp043(final_gdf):
    sr2 = final_gdf[final_gdf["subregion"] == "Subregion2"].copy()
    sr2.to_file(EXP043_PARCEL_FILE, driver="GPKG")
    print(f"\nSubregion2 slice for EXP043: {len(sr2)} parcels -> {EXP043_PARCEL_FILE}")
    print(sr2["dataset_split"].value_counts())
    print(sr2["puistu_tyyp"].value_counts())


def main():
    gdf = load_composition()
    print(f"\nTotal usable parcels nationwide: {len(gdf)}")
    print(gdf["peapuuliik"].value_counts())

    final = assign_and_split_by_subregion(gdf)
    print(f"\nAfter subregion assignment + split: {len(final)} parcel-rows "
          f"(a parcel straddling two subregion boundaries can appear twice, "
          f"same as scripts/split.py)")

    final = add_aux_features(final)

    PATHS.processed.mkdir(parents=True, exist_ok=True)
    final.to_file(OUTPUT_GPKG, driver="GPKG")
    final.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nDONE -> {OUTPUT_GPKG}")
    print(final.groupby(["subregion", "dataset_split"]).size())

    export_subregion2_for_exp043(final)


if __name__ == "__main__":
    main()
