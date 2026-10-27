"""
Subregion2 dataset statistics: how many parcels fall into the train/val/test
split, and how they break down by main tree species (target class).

Both the pure-stand base dataset (ptk 3.2.1, Tabel 3.1) and the mixed-stand
extended dataset (ptk 3.2.2) go through the same clearcut-overlap removal
(src.preprocessing.clearcut.remove_clearcut_ids) that the actual training
scripts (exp038.py, exp048.py) apply - without it, parcel counts are the
pre-filter numbers (9776 / 10 309ish) rather than the numbers the thesis
text and Tabel 3.1/3.2 actually report (9117 / ~10 282).

Outputs:
  - tables/subregion2_split_species_counts.csv
  - tables/subregion2_extended_split_species_counts.csv
  - figures/subregion2_parcels_per_split.png            (parcel counts per split, base dataset)
  - figures/subregion2_species_by_split.png              (grouped bar: species x split, base dataset)
  - figures/subregion2_extended_species_by_split.png     (grouped bar: species x split, extended/mixed-stand dataset)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.paths import PATHS
from src.preprocessing.clearcut import remove_clearcut_ids

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"

OVERLAP_FILE = str(PATHS.reference / "clearcut_overlap" / "stands_overlapping_clearcuts_over20pct.csv")

SPLITS = ["train", "val", "test"]
# 10-day (not daily) files: this is what exp038.py itself reads before
# clearcut filtering, and after filtering it reproduces Tabel 3.1 exactly
# (6437 / 1170 / 1510), unlike the daily file which is off by 1-3 parcels
# per class (a handful of parcels have zero valid daily observations).
BASE_CSV = {
    split: PATHS.intermediate / f"Subregion2_{split}_sentinel2_10day_timeseries.csv"
    for split in SPLITS
}
EXTENDED_CSV = {
    split: PATHS.intermediate / "EXP047_composition_daily" / f"composition_{split}_sentinel2_daily_timeseries.csv"
    for split in SPLITS
}

# Fixed categorical color order (dataviz skill palette, light mode), assigned
# by species code so identity stays consistent across every figure in this folder.
SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA", "MUU"]
SPECIES_COLOR = {
    "HB": "#2a78d6",   # blue
    "KS": "#eb6834",   # orange
    "KU": "#1baf7a",   # aqua
    "LM": "#eda100",   # yellow
    "LV": "#e87ba4",   # magenta
    "MA": "#008300",   # green
    "MUU": "#4a3aa7",  # violet
}
SPECIES_LABEL_ET = {
    "HB": "haab (HB)",
    "KS": "kask (KS)",
    "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)",
    "LV": "hall lepp (LV)",
    "MA": "mänd (MA)",
    "MUU": "muu (MUU)",
}

SPLIT_LABEL_ET = {
    "train": "Treeningandmestik",
    "val": "Valideerimisandmestik",
    "test": "Testandmestik",
}
SPLIT_ORDER = ["train", "val", "test"]

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_base_parcels():
    raw = {}
    for split in SPLITS:
        df = pd.read_csv(BASE_CSV[split], usecols=["id", "target", "subregion"])
        df = df.drop_duplicates("id").copy()
        assert (df["subregion"] == "Subregion2").all()
        raw[split] = df

    filtered = remove_clearcut_ids(raw["train"], raw["val"], raw["test"], OVERLAP_FILE, "id", "target")
    frames = []
    for split, df in zip(SPLITS, filtered):
        df = df.copy()
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_extended_parcels():
    raw = {}
    for split in SPLITS:
        df = pd.read_csv(EXTENDED_CSV[split], usecols=["id", "target"])
        df = df.drop_duplicates("id").copy()
        raw[split] = df

    filtered = remove_clearcut_ids(raw["train"], raw["val"], raw["test"], OVERLAP_FILE, "id", "target")
    frames = []
    for split, df in zip(SPLITS, filtered):
        df = df.copy()
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.yaxis.grid(True, color=GRID_HAIRLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def plot_parcels_per_split(counts):
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    style_axes(ax)

    values = [counts[s] for s in SPLIT_ORDER]
    labels = [SPLIT_LABEL_ET[s] for s in SPLIT_ORDER]
    bars = ax.bar(labels, values, color="#2a78d6", width=0.55, zorder=3)

    for bar, v in zip(bars, values):
        ax.annotate(
            f"{v:,}".replace(",", " "),
            (bar.get_x() + bar.get_width() / 2, v),
            textcoords="offset points", xytext=(0, 4),
            ha="center", va="bottom", color=INK_PRIMARY, fontsize=10,
        )

    ax.set_ylabel("Eraldiste arv", color=INK_MUTED)
    ax.set_title("Alamregioon 2 eraldiste arv treening-, valideerimis- ja testandmestikus", color=INK_PRIMARY, fontsize=11, pad=12)
    ax.set_ylim(0, max(values) * 1.15)

    fig.tight_layout()
    out = FIGURES_DIR / "subregion2_parcels_per_split.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_species_by_split(pivot, title, out_name):
    species = [s for s in SPECIES_ORDER if s in pivot.columns]
    x = range(len(SPLIT_ORDER))
    n_species = len(species)
    bar_width = 0.8 / n_species

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    style_axes(ax)

    for i, sp in enumerate(species):
        offsets = [xi - 0.4 + bar_width * (i + 0.5) for xi in x]
        values = [pivot.loc[s, sp] if s in pivot.index else 0 for s in SPLIT_ORDER]
        ax.bar(
            offsets, values, width=bar_width * 0.9,
            color=SPECIES_COLOR[sp], label=SPECIES_LABEL_ET[sp], zorder=3,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([SPLIT_LABEL_ET[s] for s in SPLIT_ORDER])
    ax.set_ylabel("Eraldiste arv", color=INK_MUTED)
    ax.set_xlabel("Andmestik", color=INK_MUTED)
    ax.set_title(title, color=INK_PRIMARY, fontsize=11, pad=12)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9)

    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Base (pure-stand) dataset, ptk 3.2.1 / Tabel 3.1 ---
    parcels = load_base_parcels()

    counts = parcels.groupby("split").size()
    print("Base dataset - parcels per split:")
    print(counts)

    pivot = parcels.pivot_table(index="split", columns="target", aggfunc="size", fill_value=0)
    print("\nBase dataset - parcels per split x species:")
    print(pivot)

    out_table = TABLES_DIR / "subregion2_split_species_counts.csv"
    pivot.assign(total=pivot.sum(axis=1)).to_csv(out_table, encoding="utf-8-sig")
    print(f"\nSaved table: {out_table}")

    plot_parcels_per_split(counts)
    plot_species_by_split(
        pivot,
        "Alamregioon 2 eraldiste jaotus enamuspuuliigi järgi",
        "subregion2_species_by_split.png",
    )

    # --- Extended (mixed-stand) dataset, ptk 3.2.2 / Tabel 3.2 ---
    ext_parcels = load_extended_parcels()

    ext_counts = ext_parcels.groupby("split").size()
    print("\nExtended dataset - parcels per split:")
    print(ext_counts)

    ext_pivot = ext_parcels.pivot_table(index="split", columns="target", aggfunc="size", fill_value=0)
    print("\nExtended dataset - parcels per split x species:")
    print(ext_pivot)

    ext_out_table = TABLES_DIR / "subregion2_extended_split_species_counts.csv"
    ext_pivot.assign(total=ext_pivot.sum(axis=1)).to_csv(ext_out_table, encoding="utf-8-sig")
    print(f"\nSaved table: {ext_out_table}")

    plot_species_by_split(
        ext_pivot,
        "Alamregioon 2 laiendatud andmestiku eraldiste jaotus enamuspuuliigi järgi",
        "subregion2_extended_species_by_split.png",
    )


if __name__ == "__main__":
    main()
