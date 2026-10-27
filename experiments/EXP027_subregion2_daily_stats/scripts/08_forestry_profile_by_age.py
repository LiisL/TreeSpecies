"""Species forestry profile (script 07), stratified by age group.

Uses the SAME 20-year threshold already established elsewhere in this
thesis - experiments/EXP021_species_signature_features/age_split_analysis.py
found a dramatic classification accuracy jump at this exact cutoff
(0.653 for stands <20y vs. 0.886 for >=20y), independently confirmed by
Lang et al. (2018)'s own 20-year threshold (already cited in this thesis).
Using a different split here (e.g. 10 years) would introduce an
inconsistency with that existing, already-validated analysis - deliberately
not done, even though sample sizes would support a 10-year split too
(checked: minimum group size at either threshold is well above 70 parcels
for every species).

Answers a concrete question directly: species height (and other forestry
metrics) depend heavily on stand age, so a single unstratified median (as
in script 07) can understate a species' typical MATURE size if the local
sample skews young (as it does for haab/HB and lepp/LV in this dataset).

Outputs:
  - tables/subregion2_species_forestry_profile_by_age.csv
  - figures/species_forestry_profile_by_age.png   (height + tagavara, 2-panel)
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.paths import PATHS

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"

STANDS_FILE = PATHS.processed / "final_split_clean.gpkg"
HEIGHT_SOIL_FILE = TABLES_DIR / "subregion2_parcel_height_soil.csv"

AGE_THRESHOLD = 20  # same as EXP021's age_split_analysis.py / Lang et al. (2018)
AGE_LABELS = {"young": f"< {AGE_THRESHOLD} aastat", "old": f"≥ {AGE_THRESHOLD} aastat"}

SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA"]
SPECIES_LABEL_ET = {
    "HB": "haab (HB)", "KS": "kask (KS)", "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)", "LV": "lepp (LV)", "MA": "mänd (MA)",
}
YOUNG_COLOR = "#86b6ef"
OLD_COLOR = "#184f95"

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


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

    height = pd.read_csv(HEIGHT_SOIL_FILE)[["id", "height_median"]]
    gdf["id_left"] = gdf["id_left"].astype(float)
    height["id"] = height["id"].astype(float)
    df = gdf.merge(height, left_on="id_left", right_on="id", how="left")

    df["age_group"] = np.where(df["vanus"] < AGE_THRESHOLD, "young", "old")

    # =====================================================
    # Summary table
    # =====================================================

    fields = {"height_median": "Kõrgus (m)", "tagavara": "Tagavara (m³/ha)", "pindala": "Pindala (ha)"}

    rows = []
    for species in SPECIES_ORDER:
        for age_key in ["young", "old"]:
            sub = df[(df["target"] == species) & (df["age_group"] == age_key)]
            row = {"species": species, "age_group": AGE_LABELS[age_key], "n_parcels": len(sub)}
            for field in fields:
                vals = sub[field].dropna()
                row[f"{field}_median"] = vals.median() if len(vals) else np.nan
                row[f"{field}_p25"] = vals.quantile(0.25) if len(vals) else np.nan
                row[f"{field}_p75"] = vals.quantile(0.75) if len(vals) else np.nan
            rows.append(row)

    summary = pd.DataFrame(rows)
    out_table = TABLES_DIR / "subregion2_species_forestry_profile_by_age.csv"
    summary.to_csv(out_table, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_table}")
    print(f"\nGroup sizes (age threshold {AGE_THRESHOLD}y):")
    print(summary.pivot(index="species", columns="age_group", values="n_parcels"))

    print("\nHeight median [IQR] by species and age group:")
    for _, row in summary.iterrows():
        print(f"  {SPECIES_LABEL_ET[row['species']]}, {row['age_group']} (n={row['n_parcels']}): "
              f"{row['height_median_median']:.1f} [{row['height_median_p25']:.1f}-{row['height_median_p75']:.1f}] m")

    # =====================================================
    # Grouped boxplot: height + tagavara, young vs old per species
    # =====================================================

    def grouped_boxplot(ax, field, ylabel):
        style_axes(ax)
        positions_young = [i - 0.22 for i in range(len(SPECIES_ORDER))]
        positions_old = [i + 0.22 for i in range(len(SPECIES_ORDER))]

        data_young = [df.loc[(df["target"] == s) & (df["age_group"] == "young"), field].dropna() for s in SPECIES_ORDER]
        data_old = [df.loc[(df["target"] == s) & (df["age_group"] == "old"), field].dropna() for s in SPECIES_ORDER]

        bp_y = ax.boxplot(data_young, positions=positions_young, widths=0.38, patch_artist=True,
                           showfliers=False, medianprops={"color": INK_PRIMARY})
        bp_o = ax.boxplot(data_old, positions=positions_old, widths=0.38, patch_artist=True,
                           showfliers=False, medianprops={"color": "white"})
        for box in bp_y["boxes"]:
            box.set_facecolor(YOUNG_COLOR)
        for box in bp_o["boxes"]:
            box.set_facecolor(OLD_COLOR)

        ax.set_xticks(range(len(SPECIES_ORDER)))
        ax.set_xticklabels([SPECIES_LABEL_ET[s] for s in SPECIES_ORDER], rotation=30, ha="right")
        ax.set_ylabel(ylabel, color=INK_MUTED)
        return bp_y["boxes"][0], bp_o["boxes"][0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)
    handles = grouped_boxplot(axes[0], "height_median", "Kõrgus (m)")
    grouped_boxplot(axes[1], "tagavara", "Tagavara (m³/ha)")

    fig.legend(handles, [AGE_LABELS["young"], AGE_LABELS["old"]], loc="lower center",
               ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Puuliikide tunnused vanuseklasside kaupa (lävi {AGE_THRESHOLD} aastat)",
                 fontsize=13, color=INK_PRIMARY, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "species_forestry_profile_by_age.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
