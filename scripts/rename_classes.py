import sys
from pathlib import Path
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import PATHS

# file
FILE = PATHS.processed / "final_split.gpkg"

df = gpd.read_file(FILE)

# classes we keep
main_classes = ["MA", "KS", "KU", "LV", "HB", "LM"]

# new column (recommend not overwriting)
df["target"] = df["puuliigi_k"].apply(
    lambda x: x if x in main_classes else "MUU"
)

# check
print(df["target"].value_counts())
print(df.groupby(["dataset_split", "target"]).size())

OUT = PATHS.processed / "final_split_clean.gpkg"

df.to_file(OUT, driver="GPKG")
