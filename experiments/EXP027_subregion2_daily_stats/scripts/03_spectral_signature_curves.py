"""
Spectral signature curves: for every main tree species, shows how each
Sentinel-2 band/index behaves across the 2025 growing-season year, using the
raw daily observation dates (no 10-day binning).

Aggregation: for each (date, species) pair, take the MEDIAN across all
parcels of that species of the parcel-level spatial MEDIAN value
(the "<band>_median" columns already in the daily CSVs). Median-of-medians
is used throughout because it is far less sensitive than a mean to the
per-parcel outliers a raw daily dataset is full of (partial cloud/shadow
contamination, mixed edge pixels, single mislabeled parcels) - see the
analysis README for the mean-vs-median discussion.

Train + val + test are combined here (this is a descriptive picture of what
each species looks like spectrally, not a model split), using only rows
belonging to the six main species (MUU excluded - it is a catch-all "other"
class, not a coherent spectral signature).

Two complementary views are produced:
  - "by band" (panel = band, lines = species) - answers "how do species differ
    on this one band, through the year?"
  - "by species" (panel = species, lines = bands) - answers "how does this one
    species move across all its bands together, through the year?" Since the
    bands/indices sit on very different raw scales (reflectance ~0-0.5,
    EVI up to ~1.6, CIRE up to ~6), each band is min-max normalized to 0-1
    *within that species' own year* before plotting, so line shape/timing is
    comparable - the normalized axis is for shape comparison only, not
    absolute magnitude (see the "by band" figures for that).

Outputs:
  - tables/subregion2_species_band_daily_signature.csv   (long-form aggregated table)
  - figures/spectral_signature_reflectance_bands.png       (10-panel grid, B2..B12; panel=band, lines=species)
  - figures/spectral_signature_vegetation_indices.png      (6-panel grid, NDVI..CIRE; panel=band, lines=species)
  - figures/spectral_signature_by_species_reflectance.png  (6-panel grid, one per species; panel=species, lines=B2..B12, normalized)
  - figures/spectral_signature_by_species_indices.png      (6-panel grid, one per species; panel=species, lines=NDVI..CIRE, normalized)
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.paths import PATHS

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ANALYSIS_DIR / "tables"
FIGURES_DIR = ANALYSIS_DIR / "figures"

SPLITS = ["train", "val", "test"]
DAILY_CSV = {
    split: PATHS.intermediate / f"Subregion2_{split}_sentinel2_daily_timeseries.csv"
    for split in SPLITS
}

REFLECTANCE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
VEGETATION_INDICES = ["NDVI", "NDRE", "NDMI", "NBR", "EVI", "CIRE"]
ALL_BANDS = REFLECTANCE_BANDS + VEGETATION_INDICES

SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA"]
SPECIES_COLOR = {
    "HB": "#2a78d6",   # blue
    "KS": "#eb6834",   # orange
    "KU": "#1baf7a",   # aqua
    "LM": "#eda100",   # yellow
    "LV": "#e87ba4",   # magenta
    "MA": "#008300",   # green
}
SPECIES_LABEL_ET = {
    "HB": "haab (HB)",
    "KS": "kask (KS)",
    "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)",
    "LV": "hall lepp (LV)",
    "MA": "mänd (MA)",
}

# A sequential blue ramp was tried first (ordered by wavelength) but 10 shades
# of the same hue were too close to tell apart at a glance in a 10-line
# overlay - switched to 10 distinct categorical hues instead (readability
# over strict physical-ordering encoding).
REFLECTANCE_BAND_COLOR = {
    "B2": "#1f77b4", "B3": "#ff7f0e", "B4": "#2ca02c", "B5": "#d62728", "B6": "#9467bd",
    "B7": "#8c564b", "B8": "#e377c2", "B8A": "#7f7f7f", "B11": "#bcbd22", "B12": "#17becf",
}
# Vegetation indices are unrelated formulas (no shared physical ordering), so
# they get fixed categorical hues instead, same slot order as the species palette.
INDEX_COLOR = {
    "NDVI": "#2a78d6",  # blue
    "NDRE": "#eb6834",  # orange
    "NDMI": "#1baf7a",  # aqua
    "NBR": "#eda100",   # yellow
    "EVI": "#e87ba4",   # magenta
    "CIRE": "#008300",  # green
}

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

EST_MONTHS = ["", "Jaan", "Veebr", "Märts", "Apr", "Mai", "Juuni",
              "Juuli", "Aug", "Sept", "Okt", "Nov", "Dets"]


def load_combined():
    usecols = ["id", "date_from", "target"] + [f"{b}_median" for b in ALL_BANDS]
    frames = []
    for split in SPLITS:
        df = pd.read_csv(DAILY_CSV[split], usecols=usecols)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["target"].isin(SPECIES_ORDER)].copy()
    combined["date_from"] = pd.to_datetime(combined["date_from"]).dt.tz_localize(None).dt.normalize()
    return combined


def build_signature_table(combined):
    value_cols = [f"{b}_median" for b in ALL_BANDS]
    grouped = combined.groupby(["date_from", "target"])[value_cols].median().reset_index()
    n_parcels = combined.groupby(["date_from", "target"])["id"].nunique().rename("n_parcels")
    grouped = grouped.merge(n_parcels, on=["date_from", "target"])
    grouped = grouped.rename(columns={f"{b}_median": b for b in ALL_BANDS})
    return grouped


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=8)
    ax.yaxis.grid(True, color=GRID_HAIRLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: EST_MONTHS[mdates.num2date(x).month]))


def plot_grid(sig, bands, title, out_name, ncols, nrows, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150, sharex=True)
    axes = axes.flatten()

    for ax, band in zip(axes, bands):
        style_axes(ax)
        for sp in SPECIES_ORDER:
            sub = sig[sig["target"] == sp].sort_values("date_from")
            ax.plot(
                sub["date_from"], sub[band],
                color=SPECIES_COLOR[sp], linewidth=1.3,
                marker="o", markersize=2.5, alpha=0.9,
                label=SPECIES_LABEL_ET[sp],
            )
        ax.set_title(band, fontsize=10, color=INK_PRIMARY, loc="left")

    for ax in axes[len(bands):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(SPECIES_ORDER),
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, fontsize=13, color=INK_PRIMARY, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    out = FIGURES_DIR / out_name
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_species_grid(sig, bands, band_colors, title, out_name, ncols, nrows, figsize):
    """One panel per species; within a panel, one normalized (0-1, min-max
    over that species' own year) line per band - shows how a single species
    moves across all its bands together, through the year."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150, sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, sp in zip(axes, SPECIES_ORDER):
        style_axes(ax)
        sub = sig[sig["target"] == sp].sort_values("date_from")
        for band in bands:
            vals = sub[band]
            vmin, vmax = vals.min(), vals.max()
            norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else vals * 0
            ax.plot(
                sub["date_from"], norm,
                color=band_colors[band], linewidth=1.3,
                marker="o", markersize=2.5, alpha=0.9,
                label=band,
            )
        ax.set_title(SPECIES_LABEL_ET[sp], fontsize=10, color=INK_PRIMARY, loc="left")
        ax.set_ylim(-0.05, 1.05)

    for ax in axes[len(SPECIES_ORDER):]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=min(len(bands), 6),
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03),
    )
    fig.text(0.0, 0.5, "Normaliseeritud väärtus (0-1, liigi enda aasta miinimum-maksimum)",
              rotation=90, va="center", ha="left", fontsize=9, color=INK_MUTED)
    fig.suptitle(title, fontsize=13, color=INK_PRIMARY, y=1.0)
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.98])

    out = FIGURES_DIR / out_name
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_combined()
    print(f"Combined rows (6 main species, all splits): {len(combined)}")
    print(f"Parcels: {combined['id'].nunique()}")

    sig = build_signature_table(combined)

    out_table = TABLES_DIR / "subregion2_species_band_daily_signature.csv"
    sig.to_csv(out_table, index=False, encoding="utf-8-sig")
    print(f"Saved signature table: {out_table} (shape {sig.shape})")

    plot_grid(
        sig, REFLECTANCE_BANDS,
        "Spektraalne signatuur puuliigiti - Sentinel-2 peegelduskanalid (2025)",
        "spectral_signature_reflectance_bands.png",
        ncols=2, nrows=5, figsize=(10, 16),
    )
    plot_grid(
        sig, VEGETATION_INDICES,
        "Spektraalne signatuur puuliigiti - vegetatsiooniindeksid (2025)",
        "spectral_signature_vegetation_indices.png",
        ncols=2, nrows=3, figsize=(10, 10),
    )

    plot_species_grid(
        sig, REFLECTANCE_BANDS, REFLECTANCE_BAND_COLOR,
        "Puuliikide signatuur kõigi peegelduskanalite lõikes (2025, normaliseeritud)",
        "spectral_signature_by_species_reflectance.png",
        ncols=2, nrows=3, figsize=(10, 12),
    )
    plot_species_grid(
        sig, VEGETATION_INDICES, INDEX_COLOR,
        "Puuliikide signatuur kõigi vegetatsiooniindeksite lõikes (2025, normaliseeritud)",
        "spectral_signature_by_species_indices.png",
        ncols=2, nrows=3, figsize=(10, 12),
    )


if __name__ == "__main__":
    main()
