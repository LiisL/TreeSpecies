import sys
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# =========================
# CONFIG
# =========================

# Stand file (tree species)
STANDS_FILE = PATHS.raw / "puuliik_filt_pindala" / "puuliik_filt_pindala2.shp"

# Subregion folder (7 shp files)
SUBREGION_FOLDER = PATHS.raw / "Subregions"

# Output
OUTPUT_FILE = PATHS.processed / "final_split.gpkg"

# Fields
TARGET_COL = "puuliigi_k"
PURITY_COL = "osakaal"
AREA_COL = "pindala"

# Filters
MIN_PURITY = 80
MIN_AREA = 1.0

GRID_SIZE = 5000
RANDOM_STATE = 42

# =========================
# FUNCTION: SPLIT
# =========================

def split_subregion(gdf, name):

    # filters
    gdf = gdf[
        (gdf[TARGET_COL].notna()) &
        (gdf[PURITY_COL] >= MIN_PURITY) &
        (gdf[AREA_COL] >= MIN_AREA)
    ].copy()

    if len(gdf) < 20:
        print(f"⚠ {name}: not enough data")
        return None

    # spatial group
    gdf_m = gdf.to_crs(3301)
    cent = gdf_m.geometry.centroid

    gdf_m["gx"] = (cent.x // GRID_SIZE).astype(int)
    gdf_m["gy"] = (cent.y // GRID_SIZE).astype(int)
    gdf_m["group"] = gdf_m["gx"].astype(str) + "_" + gdf_m["gy"].astype(str)

    # split
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(gdf_m, groups=gdf_m["group"]))

    train = gdf_m.iloc[train_idx]
    temp = gdf_m.iloc[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(temp, groups=temp["group"]))

    val = temp.iloc[val_idx]
    test = temp.iloc[test_idx]

    train["dataset_split"] = "train"
    val["dataset_split"] = "val"
    test["dataset_split"] = "test"

    out = pd.concat([train, val, test])
    out = out.to_crs(gdf.crs)
    out["subregion"] = name

    print(f"\n{name}")
    print(out["dataset_split"].value_counts())

    return out

# =========================
# MAIN
# =========================

def main():

    stands = gpd.read_file(STANDS_FILE)

    all_results = []

    sub_files = sorted(Path(SUBREGION_FOLDER).glob("*.shp"))

    for sub_file in sub_files:
        print(f"\n=== {sub_file.name} ===")

        subregion = gpd.read_file(sub_file)

        # same CRS
        if stands.crs != subregion.crs:
            subregion = subregion.to_crs(stands.crs)

        # clip stands into subregion
        stands_sub = gpd.sjoin(stands, subregion, predicate="intersects")

        if len(stands_sub) == 0:
            print("⚠ empty subregion")
            continue

        result = split_subregion(stands_sub, sub_file.stem)

        if result is not None:
            all_results.append(result)

    final = pd.concat(all_results)

    final.to_file(OUTPUT_FILE, driver="GPKG")

    print("\nDONE")
    print(final["dataset_split"].value_counts())



if __name__ == "__main__":
    main()
