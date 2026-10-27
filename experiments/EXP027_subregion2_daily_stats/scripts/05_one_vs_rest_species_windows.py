"""One-vs-rest species signature analysis: for EVERY species, find the
10-day windows and bands where it separates most from all five other
species combined - not single dates (script 04's finding: a pure
"strongest single date" ranking clusters on one narrow event and misses
season-wide complementary signal, which cost LM's October signature
entirely in the top30_separability_features.py follow-up), and not
pairwise (script 04's Cohen's d was species-A-vs-species-B; this is
species-vs-everyone-else, directly answering "what makes THIS species
distinct").

Method:
  1. Bin the raw daily dates into 10-day windows across the year (day-of-
     year // 10), then take the per-parcel MEDIAN of each band/index
     within a window - smooths single-date noise while staying much finer
     than "whole season" and avoiding script 04's single-date redundancy
     problem (adjacent near-duplicate dates collapsing into one signal).
  2. For every (species, band, window), one-vs-rest Cohen's d: this
     species' parcels vs. every other species' parcels combined (MUU
     excluded throughout, as in every other EXP027/EXP028 analysis).
  3. Same reliability guard as script 04 (minimum real, non-imputed
     parcel count in both the "this species" and "rest" groups) - windows
     are much better populated than single dates, but the guard is kept
     for consistency/safety.
  4. Rank per species; report each species' own top windows/bands and
     visualize all six as one comparable heatmap grid (band x window,
     shared color scale across species so relative "how distinct is this
     species at its best" is directly comparable).

Outputs:
  - tables/subregion2_one_vs_rest_by_species_window_band.csv   (full ranked table)
  - tables/subregion2_one_vs_rest_top10_per_species.csv         (top 10 windows/bands per species)
  - figures/one_vs_rest_heatmap_grid.png                         (6-panel band x window heatmap)
  - figures/one_vs_rest_best_score_per_species.png                (best achievable one-vs-rest |d| per species)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

REFLECTANCE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
VEGETATION_INDICES = ["NDVI", "NDRE", "NDMI", "NBR", "EVI", "CIRE"]
ALL_BANDS = REFLECTANCE_BANDS + VEGETATION_INDICES
MEDIAN_COLS = [f"{b}_median" for b in ALL_BANDS]

SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA"]
SPECIES_LABEL_ET = {
    "HB": "haab (HB)", "KS": "kask (KS)", "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)", "LV": "lepp (LV)", "MA": "mänd (MA)",
}

MIN_REAL_N = 20
WINDOW_DAYS = 10

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"


def load_combined():
    frames = []
    for split in SPLITS:
        df = pd.read_csv(DAILY_CSV[split], usecols=[ID_COL, DATE_COL, TARGET_COL] + MEDIAN_COLS)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined[TARGET_COL].isin(SPECIES_ORDER)].copy()
    combined[DATE_COL] = pd.to_datetime(combined[DATE_COL])
    return combined


def assign_windows(df):
    day_of_year = df[DATE_COL].dt.dayofyear
    window_idx = (day_of_year - 1) // WINDOW_DAYS
    df = df.copy()
    df["window_start"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(window_idx * WINDOW_DAYS, unit="D")
    return df


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_combined()
    print(f"Combined rows (6 main species, all splits): {len(combined)}")
    print(f"Parcels: {combined[ID_COL].nunique()}")

    combined = assign_windows(combined)
    n_windows = combined["window_start"].nunique()
    print(f"10-day windows spanning the year: {n_windows}")

    # ---- per-parcel, per-window median of each band (robust to daily noise) ----
    per_parcel_window = combined.groupby([ID_COL, "window_start"])[MEDIAN_COLS].median().reset_index()
    labels = combined[[ID_COL, TARGET_COL]].drop_duplicates()
    per_parcel_window = per_parcel_window.merge(labels, on=ID_COL, how="left")
    print(f"Per-parcel-window rows: {len(per_parcel_window)}")

    # =====================================================
    # One-vs-rest Cohen's d, per (species, band, window)
    # =====================================================

    results = []
    for window in sorted(per_parcel_window["window_start"].unique()):
        window_df = per_parcel_window[per_parcel_window["window_start"] == window]
        for band_col in MEDIAN_COLS:
            band = band_col.replace("_median", "")
            for species in SPECIES_ORDER:
                vals_species = window_df.loc[window_df[TARGET_COL] == species, band_col].dropna()
                vals_rest = window_df.loc[window_df[TARGET_COL] != species, band_col].dropna()

                n_species = len(vals_species)
                n_rest = len(vals_rest)
                if n_species < MIN_REAL_N or n_rest < MIN_REAL_N:
                    continue

                mean_diff = vals_species.mean() - vals_rest.mean()
                pooled_std = np.sqrt((vals_species.var() + vals_rest.var()) / 2)
                if pooled_std == 0:
                    continue
                cohens_d = mean_diff / pooled_std

                results.append({
                    "species": species, "band": band, "window_start": window,
                    "cohens_d": cohens_d, "abs_cohens_d": abs(cohens_d),
                    "n_species": n_species, "n_rest": n_rest,
                })

    ovr = pd.DataFrame(results)
    out_full = TABLES_DIR / "subregion2_one_vs_rest_by_species_window_band.csv"
    ovr.to_csv(out_full, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_full} ({len(ovr)} rows)")

    # ---- top 10 per species ----
    top10_parts = []
    for species in SPECIES_ORDER:
        sub = ovr[ovr["species"] == species].sort_values("abs_cohens_d", ascending=False).head(10)
        top10_parts.append(sub)
    top10 = pd.concat(top10_parts, ignore_index=True)
    out_top10 = TABLES_DIR / "subregion2_one_vs_rest_top10_per_species.csv"
    top10.to_csv(out_top10, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_top10}")

    print("\nEach species' single best one-vs-rest signature (band, 10-day window, direction):")
    for species in SPECIES_ORDER:
        best = ovr[ovr["species"] == species].sort_values("abs_cohens_d", ascending=False).iloc[0]
        direction = "higher" if best["cohens_d"] > 0 else "lower"
        print(f"  {species}: {best['band']}, window starting {best['window_start'].strftime('%d.%m')} "
              f"({direction} than the rest, |d|={best['abs_cohens_d']:.2f})")

    # =====================================================
    # Figure A: heatmap grid, one panel per species (band x window)
    # =====================================================

    vmax = ovr["abs_cohens_d"].quantile(0.99)
    windows_sorted = sorted(ovr["window_start"].unique())
    month_ticks = []
    month_labels = []
    seen_months = set()
    for i, w in enumerate(windows_sorted):
        if w.month not in seen_months:
            month_ticks.append(i)
            month_labels.append(w.strftime("%b"))
            seen_months.add(w.month)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
    axes = axes.flatten()

    for ax, species in zip(axes, SPECIES_ORDER):
        sub = ovr[ovr["species"] == species]
        pivot = sub.pivot_table(index="band", columns="window_start", values="abs_cohens_d")
        pivot = pivot.reindex(index=ALL_BANDS, columns=windows_sorted)

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_yticks(range(len(ALL_BANDS)))
        ax.set_yticklabels(ALL_BANDS, fontsize=7, color=INK_MUTED)
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, fontsize=7, color=INK_MUTED, rotation=45, ha="right")
        ax.set_title(SPECIES_LABEL_ET[species], fontsize=11, color=INK_PRIMARY, loc="left")

    fig.suptitle("Iga puuliigi eristatavus (one-vs-rest |Cohen's d|) kanali ja 10-päevase akna kaupa",
                  fontsize=13, color=INK_PRIMARY, y=1.0)
    fig.colorbar(im, ax=axes, shrink=0.6, label="one-vs-rest |Cohen's d|", location="right", pad=0.02)
    out = FIGURES_DIR / "one_vs_rest_heatmap_grid.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # =====================================================
    # Figure B: best achievable one-vs-rest |d| per species
    # =====================================================

    best_per_species = ovr.loc[ovr.groupby("species")["abs_cohens_d"].idxmax()].sort_values("abs_cohens_d")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    labels = [SPECIES_LABEL_ET[s] for s in best_per_species["species"]]
    ax.barh(labels, best_per_species["abs_cohens_d"], color="#2a78d6", zorder=3)
    for i, (_, row) in enumerate(best_per_species.iterrows()):
        ax.annotate(f"{row['band']}, {row['window_start'].strftime('%d.%m')}",
                     (row["abs_cohens_d"], i), textcoords="offset points", xytext=(6, 0),
                     va="center", fontsize=9, color=INK_PRIMARY)

    ax.set_xlabel("Parim saavutatav one-vs-rest |Cohen's d|", color=INK_MUTED)
    ax.set_title("Iga liigi kõige selgem eristav tunnus (kanal, aken)", fontsize=12, color=INK_PRIMARY, pad=12)
    ax.set_xlim(0, best_per_species["abs_cohens_d"].max() * 1.35)
    fig.tight_layout()
    out = FIGURES_DIR / "one_vs_rest_best_score_per_species.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
