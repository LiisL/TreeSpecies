"""Empirical forestry-characteristic profile per species, from THIS
project's own Subregion2 data - not textbook/dendrology facts (a
different, unverifiable-from-our-data kind of claim - see the README note
below), but what the actual stands in this study area look like.

Fields used, all unambiguous, standard Estonian forestry metrics straight
from the Metsaregister stand file (data/processed/final_split_clean.gpkg)
- deliberately NOT decoding `arengukl_k` (development class) or `klass`
into named categories, since no verified code lookup for them was found
in this repo (checked - no embedded raster attribute table on the soil
raster either, consistent with the EXP027 script 01 finding that
soil_majority can't currently be decoded to Boniteet/Loimis names):
  - height (m) - from analysis/EXP027_subregion2_daily_stats/tables/
    subregion2_parcel_height_soil.csv (already computed via zonal_stats
    on H_20_21_22_23.tif)
  - vanus (age, years)
  - tagavara (standing timber volume, m3/ha)
  - pindala (parcel area, ha)
  - osakaal (main species' share of the stand, % - a purity check: this
    project already filters to >=some purity threshold, this shows what
    that actually looks like per species)

Outputs:
  - tables/subregion2_species_forestry_profile.csv   (summary stats per species)
  - figures/species_forestry_profile_boxplots.png     (height/age/volume/area, 4-panel)
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.paths import PATHS

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"

STANDS_FILE = PATHS.processed / "final_split_clean.gpkg"
HEIGHT_SOIL_FILE = TABLES_DIR / "subregion2_parcel_height_soil.csv"

SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA"]
SPECIES_LABEL_ET = {
    "HB": "haab (HB)", "KS": "kask (KS)", "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)", "LV": "lepp (LV)", "MA": "mänd (MA)",
}
SPECIES_COLOR = {"HB": "#2a78d6", "KS": "#eb6834", "KU": "#1baf7a", "LM": "#eda100", "LV": "#e87ba4", "MA": "#008300"}

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FIELDS = {
    "height_median": "Kõrgus (m)",
    "vanus": "Vanus (a)",
    "tagavara": "Tagavara (m³/ha)",
    "pindala": "Pindala (ha)",
    "osakaal": "Enamuspuuliigi osakaal (%)",
}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.yaxis.grid(True, color=GRID_HAIRLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(STANDS_FILE)
    gdf = gdf[(gdf["subregion"] == "Subregion2") & (gdf["target"].isin(SPECIES_ORDER))].copy()
    print(f"Subregion2 parcels (6 main species): {len(gdf)}")

    height = pd.read_csv(HEIGHT_SOIL_FILE)[["id", "height_median"]]
    gdf["id_left"] = gdf["id_left"].astype(float)
    height["id"] = height["id"].astype(float)
    df = gdf.merge(height, left_on="id_left", right_on="id", how="left")

    print(f"Parcels with a height value: {df['height_median'].notna().sum()} / {len(df)}")

    # =====================================================
    # Summary table
    # =====================================================

    summary_rows = []
    for species in SPECIES_ORDER:
        sub = df[df["target"] == species]
        row = {"species": species, "n_parcels": len(sub)}
        for field in FIELDS:
            vals = sub[field].dropna()
            row[f"{field}_median"] = vals.median()
            row[f"{field}_p25"] = vals.quantile(0.25)
            row[f"{field}_p75"] = vals.quantile(0.75)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    out_table = TABLES_DIR / "subregion2_species_forestry_profile.csv"
    summary.to_csv(out_table, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_table}")

    print("\nSpecies forestry profile (median [25th-75th percentile]):")
    for _, row in summary.iterrows():
        print(f"\n{SPECIES_LABEL_ET[row['species']]} (n={row['n_parcels']}):")
        for field, label_et in FIELDS.items():
            print(f"  {label_et}: {row[f'{field}_median']:.1f} "
                  f"[{row[f'{field}_p25']:.1f}-{row[f'{field}_p75']:.1f}]")

    # =====================================================
    # Boxplot grid
    # =====================================================

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    axes = axes.flatten()

    for ax, (field, label_et) in zip(axes, FIELDS.items()):
        style_axes(ax)
        data = [df.loc[df["target"] == s, field].dropna() for s in SPECIES_ORDER]
        bp = ax.boxplot(data, tick_labels=[SPECIES_LABEL_ET[s] for s in SPECIES_ORDER],
                         patch_artist=True, showfliers=False, medianprops={"color": INK_PRIMARY})
        for patch, s in zip(bp["boxes"], SPECIES_ORDER):
            patch.set_facecolor(SPECIES_COLOR[s])
            patch.set_alpha(0.75)
        ax.set_ylabel(label_et, color=INK_MUTED)
        ax.tick_params(axis="x", labelsize=8, rotation=30)

    axes[-1].axis("off")

    fig.suptitle("Puuliikide metsanduslikud tunnused Subregion2 andmestikus", fontsize=13, color=INK_PRIMARY, y=1.0)
    fig.tight_layout()
    out = FIGURES_DIR / "species_forestry_profile_boxplots.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
