import os
import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# =========================================================
# INPUTS
# =========================================================

STANDS_FILE = PATHS.raw / "puuliik_pindala_SR2.shp"
# The national clearcut-polygon file isn't part of this repo's own data - see data/external/README.md
CLEARCUT_FILE = PATHS.external / "EestiLRjaan2025_kuni_veebr2026" / "EestiLRjaan2025_kuni_veebr2026.shp"

OUT_DIR = PATHS.reference / "clearcut_overlap"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "stands_overlapping_clearcuts_over20pct.csv")
OUT_GPKG = os.path.join(OUT_DIR, "stands_overlapping_clearcuts_over20pct.gpkg")

# =========================================================
# CONFIG
# =========================================================

ID_COL = "id_left"      # if your file uses "id" instead, set it to "id"
TARGET_COL = "target"
SPLIT_COL = "dataset_split"
SUBREGION_COL = "subregion"

OVERLAP_THRESHOLD = 20.0  # percent

# =========================================================
# 1. LOAD DATA
# =========================================================

stands = gpd.read_file(STANDS_FILE)
cuts = gpd.read_file(CLEARCUT_FILE)

print("Stands:", stands.shape)
print("Clearcuts:", cuts.shape)

# check
needed_cols = [ID_COL, TARGET_COL, SPLIT_COL, SUBREGION_COL, "geometry"]
missing = [c for c in needed_cols if c not in stands.columns]
if missing:
    raise ValueError(f"Stands file is missing columns: {missing}")

if stands.crs is None:
    raise ValueError("Stands file has no CRS.")

if cuts.crs is None:
    raise ValueError("Clearcut file has no CRS.")

# =========================================================
# 2. MATCH CRS
# =========================================================

cuts = cuts.to_crs(stands.crs)

# =========================================================
# 3. FIX GEOMETRIES
# =========================================================

stands["geometry"] = stands.geometry.buffer(0)
cuts["geometry"] = cuts.geometry.buffer(0)

stands = stands[~stands.geometry.is_empty].copy()
cuts = cuts[~cuts.geometry.is_empty].copy()

# =========================================================
# 4. STAND AREA
# =========================================================

stands["stand_area_m2"] = stands.geometry.area

# =========================================================
# 5. INTERSECT
# =========================================================

print("Computing overlaps...")

intersections = gpd.overlay(
    stands[[ID_COL, TARGET_COL, SPLIT_COL, SUBREGION_COL, "stand_area_m2", "geometry"]],
    cuts[["geometry"]],
    how="intersection",
    keep_geom_type=False
)

print("Intersections found:", intersections.shape)

if intersections.empty:
    print("No overlaps found.")
    raise SystemExit(0)

intersections["overlap_area_m2"] = intersections.geometry.area

# =========================================================
# 6. SUM OVERLAP PER STAND
# =========================================================

overlap_sum = (
    intersections
    .groupby(ID_COL, as_index=False)["overlap_area_m2"]
    .sum()
)

result = stands.merge(
    overlap_sum,
    on=ID_COL,
    how="left"
)

result["overlap_area_m2"] = result["overlap_area_m2"].fillna(0)

result["overlap_pct"] = (
    result["overlap_area_m2"] / result["stand_area_m2"] * 100
)

# =========================================================
# 7. FILTER OVER 20%
# =========================================================

bad = result[result["overlap_pct"] > OVERLAP_THRESHOLD].copy()

print("\nStands overlapping >20%:", len(bad))

print("\nBy class:")
print(bad[TARGET_COL].value_counts())

print("\nBy split:")
print(bad[SPLIT_COL].value_counts())

print("\nBy subregion:")
print(bad[SUBREGION_COL].value_counts())

# =========================================================
# 8. SAVE
# =========================================================

out_cols = [
    ID_COL,
    TARGET_COL,
    SPLIT_COL,
    SUBREGION_COL,
    "stand_area_m2",
    "overlap_area_m2",
    "overlap_pct"
]

bad[out_cols].sort_values("overlap_pct", ascending=False).to_csv(
    OUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

bad[out_cols + ["geometry"]].to_file(
    OUT_GPKG,
    driver="GPKG"
)

print("\nSaved:")
print(OUT_CSV)
print(OUT_GPKG)

print("\nSample:")
print(bad[out_cols].sort_values("overlap_pct", ascending=False).head(20))
