"""Validates a pixel-based species map against known Forest Register
(Metsaregister) parcel data: for every parcel (eraldis) that overlaps the
map extent, counts how many map pixels of each species fall inside that
parcel's boundary, and compares the pixel-majority species against the
parcel's own recorded dominant species ("peapuuliik").

This checks something genuinely different from the original parcel-level
train/test evaluation (EXP032/EXP038's own reported Accuracy/Macro F1):
those metrics describe how well the model predicts a SINGLE label per
parcel using per-parcel AGGREGATED input features (the same kind of
signal used in training). This script instead builds the prediction
bottom-up from independent PIXEL classifications and checks whether
aggregating them back up to the parcel level recovers the known answer -
a genuine end-to-end sanity check of the whole pixel-mapping pipeline
(pixel download, feature reconstruction, hierarchical routing, forest
masking), not just the classifier itself.

Terminology note (per Metsa korraldamise juhend, RT I, 31.05.2024, 11,
paragrahv 11): "peapuuliik" (the parcel attribute used here) is technically
the economically-preferred first-canopy species given site conditions,
while "enamuspuuliik" is strictly the species with the largest growing
stock (tagavara) - per the regulation, they usually coincide ("üldjuhul on
enamuspuuliik ka peapuuliik") except in specific listed cases. This
project (and this script) follows the same informal convention used
throughout - comparing the pixel-majority species against the recorded
"peapuuliik" field - since a separate tagavara-based recomputation isn't
available in this parcel data.

Rare species codes present in the parcel data but outside this project's
7-class scheme (TA, SA, LH, RE, PN, ...) are mapped to "MUU" for a fair
comparison, matching how the classification models themselves treat them.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# =========================================================
# INPUTS (edit PARCEL_FILE here once older/additional parcel data is added)
# =========================================================

PARCEL_FILE = PATHS.raw / "SubR2_pl.shp"

MAPS_TO_VALIDATE = {
    "EXP032_RF_Subregion2_full": {
        "species_tif": PATHS.outputs_predictions / "pixel_species_map_exp032_subregion2" / "species_map_subregion2.tif",
        # the mosaic step doesn't copy a class_codes.csv of its own - every
        # tile's is identical (class_code 1-7 -> HB/KS/KU/LM/LV/MA/MUU), so
        # any one of them works.
        "class_codes_csv": PATHS.outputs_predictions / "pixel_species_map_exp032_sr2_tiles" / "tile_0_1" / "class_codes.csv",
    },
    "EXP032_RF_pilot": {
        "species_tif": PATHS.outputs_predictions / "pixel_species_map_exp032_pilot" / "species_map_forestmask.tif",
        "class_codes_csv": PATHS.outputs_predictions / "pixel_species_map_exp032_pilot" / "class_codes.csv",
    },
    "EXP038_XGB_pilot": {
        "species_tif": PATHS.outputs_predictions / "pixel_species_map_exp038_pilot" / "species_map_forestmask.tif",
        "class_codes_csv": PATHS.outputs_predictions / "pixel_species_map_exp038_pilot" / "class_codes.csv",
    },
}

OUT_DIR = PATHS.outputs_tables

NODATA_CLASS = 255

KNOWN_SPECIES = {"HB", "KS", "KU", "LM", "LV", "MA", "MUU"}

# Minimum valid (non-nodata) pixels a parcel needs to have a trustworthy
# majority vote - a 1-2 pixel parcel's "majority" is not meaningful.
MIN_PIXELS_FOR_RELIABLE_VOTE = 5

# =========================================================
# FUNCTIONS
# =========================================================

def map_to_known_species(code):
    if pd.isna(code):
        return np.nan
    code = str(code).strip().upper()
    return code if code in KNOWN_SPECIES else "MUU"


def load_parcels(parcel_file, raster_crs):
    gdf = gpd.read_file(parcel_file)
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    gdf["peapuuliik_mapped"] = gdf["peapuuliik"].apply(map_to_known_species)
    return gdf


def count_pixels_per_parcel(gdf, raster_path, class_code_to_name):
    stats = zonal_stats(
        gdf,
        str(raster_path),
        categorical=True,
        nodata=NODATA_CLASS,
        geojson_out=False,
    )

    rows = []
    for (_, parcel), stat in zip(gdf.iterrows(), stats):
        counts = {class_code_to_name.get(int(k), f"code_{k}"): v for k, v in stat.items() if int(k) != NODATA_CLASS}
        total_valid = sum(counts.values())

        if total_valid == 0:
            predicted = np.nan
        else:
            predicted = max(counts, key=counts.get)

        counts_str = ", ".join(f"{sp}:{n}" for sp, n in sorted(counts.items(), key=lambda x: -x[1]))

        rows.append({
            "id": parcel.get("id"),
            "eraldise_n": parcel.get("eraldise_n"),
            "peapuuliik": parcel.get("peapuuliik"),
            "peapuuliik_mapped": parcel.get("peapuuliik_mapped"),
            "vanus": parcel.get("vanus"),
            "pindala_ha": parcel.get("pindala"),
            "pixel_counts": counts_str,
            "n_valid_pixels": total_valid,
            "predicted_dominant": predicted,
            "match": (predicted == parcel.get("peapuuliik_mapped")) if total_valid > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def summarize(df, label):
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {label}")
    print("=" * 60)

    total_parcels = len(df)
    with_pixels = df[df["n_valid_pixels"] > 0]
    print(f"Total parcels overlapping map extent: {total_parcels}")
    print(f"Parcels with >=1 valid pixel: {len(with_pixels)}")

    reliable = with_pixels[with_pixels["n_valid_pixels"] >= MIN_PIXELS_FOR_RELIABLE_VOTE]
    print(f"Parcels with >={MIN_PIXELS_FOR_RELIABLE_VOTE} valid pixels (reliable vote): {len(reliable)}")

    for subset, subset_label in [(with_pixels, "ALL parcels with data"), (reliable, f"parcels with >={MIN_PIXELS_FOR_RELIABLE_VOTE} px")]:
        if len(subset) == 0:
            continue
        acc = subset["match"].mean()
        print(f"\n--- {subset_label} (n={len(subset)}) ---")
        print(f"Overall dominant-species accuracy: {acc:.3f} ({subset['match'].sum()}/{len(subset)})")

        per_species = subset.groupby("peapuuliik_mapped")["match"].apply(
            lambda s: pd.Series({"accuracy": float(pd.Series(s, dtype="float64").mean()), "n_parcels": int(len(s))})
        ).unstack()
        per_species = per_species.sort_values("n_parcels", ascending=False)
        per_species["accuracy"] = per_species["accuracy"].round(3)
        print("\nPer true-species accuracy:")
        print(per_species)

    # Confusion matrix (true peapuuliik_mapped vs predicted_dominant), reliable subset
    if len(reliable) > 0:
        print(f"\nConfusion matrix (rows=true peapuuliik, cols=predicted dominant), n>={MIN_PIXELS_FOR_RELIABLE_VOTE}px subset:")
        cm = pd.crosstab(reliable["peapuuliik_mapped"], reliable["predicted_dominant"])
        print(cm)

        # Age-stratified accuracy, matching this project's established age-effect analysis
        if "vanus" in reliable.columns and reliable["vanus"].notna().any():
            age_bins = pd.cut(reliable["vanus"], bins=[0, 20, 200], labels=["<20a", ">=20a"])
            age_acc = reliable.groupby(age_bins, observed=True)["match"].apply(
                lambda s: pd.Series({"accuracy": float(pd.Series(s, dtype="float64").mean()), "n_parcels": int(len(s))})
            ).unstack()
            age_acc["accuracy"] = age_acc["accuracy"].round(3)
            print("\nAge-stratified accuracy:")
            print(age_acc)


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for label, cfg in MAPS_TO_VALIDATE.items():
        species_tif = cfg["species_tif"]
        class_codes_csv = cfg["class_codes_csv"]

        if not os.path.exists(species_tif):
            print(f"Skipping {label}: {species_tif} not found.")
            continue

        print(f"\nLoading class codes for {label}...")
        class_df = pd.read_csv(class_codes_csv)
        class_code_to_name = dict(zip(class_df.class_code, class_df.class_name))

        with rasterio.open(species_tif) as src:
            raster_crs = src.crs

        print(f"Loading parcels from {PARCEL_FILE}...")
        gdf = load_parcels(PARCEL_FILE, raster_crs)

        # Keep only parcels that actually intersect the raster's extent
        with rasterio.open(species_tif) as src:
            bounds = src.bounds
        from shapely.geometry import box as shapely_box
        map_box = shapely_box(*bounds)
        gdf = gdf[gdf.intersects(map_box)].reset_index(drop=True)
        print(f"Parcels intersecting map extent: {len(gdf)}")

        print(f"Counting pixels per parcel for {label}...")
        result_df = count_pixels_per_parcel(gdf, species_tif, class_code_to_name)

        out_csv = os.path.join(OUT_DIR, f"pixel_map_validation_{label}.csv")
        result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Per-parcel results saved: {out_csv}")

        summarize(result_df, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
