"""For every tree species and every 10-day window across the year, find
the top 5 most discriminating features - extending script 05's one-vs-
rest Cohen's d analysis (species vs. all others, per band, per window)
two ways: (1) reformats to explicitly rank and report the top 5 features
PER (species, window) pair rather than each species' top-10 windows/bands
overall, and (2) adds height, soil-derived TCD, and their own stats as
CONSTANT candidate features competing alongside the window's spectral
columns in every window's ranking - directly answering "does structural
information ever outrank spectral information for this species, in this
window, or is it always dominated by phenology."

Soil is deliberately excluded from this ranking: it is categorical (370
distinct codes), and Cohen's d doesn't apply to a categorical variable -
same reasoning as script 09, which used a separate descriptive method
(top-code overlap) for soil rather than force it into an effect-size
ranking built for continuous variables.

Method:
  1. Same 10-day windowing as script 05 (day-of-year // 10, per-parcel
     MEDIAN of daily values within a window - smooths single-date noise).
  2. Height (mean/median/std/min/max) and TCD (mean/median/std/min/max)
     are merged onto the SAME per-parcel-window rows used for the
     spectral columns, so every candidate feature in a given window's
     ranking is evaluated on an IDENTICAL parcel population for that
     window - a fair, apples-to-apples comparison (these 10 "structural"
     values are literally constant per parcel across all of a parcel's
     windows, but the pool of parcels differs window to window, since not
     every parcel has valid data in every window).
  3. One-vs-rest Cohen's d per (species, window, feature) - same
     MIN_REAL_N=20 reliability guard as scripts 04/05.
  4. Rank per (species, window), keep top 5.

Outputs:
  - tables/subregion2_top5_features_per_species_per_window.csv (long format,
    one row per species/window/rank)
  - tables/subregion2_top5_features_wide.csv (one row per species/window,
    columns rank1_feature..rank5_feature + their |d| values - the exact
    layout requested)
  - figures/top5_structural_feature_frequency.png (how often height/TCD
    make each species' top 5, across all windows)
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
HEIGHT_SOIL_CSV = TABLES_DIR / "subregion2_parcel_height_soil.csv"
TCD_CSV = TABLES_DIR / "subregion2_parcel_tcd.csv"

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

REFLECTANCE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
VEGETATION_INDICES = ["NDVI", "NDRE", "NDMI", "NBR", "EVI", "CIRE"]
ALL_BANDS = REFLECTANCE_BANDS + VEGETATION_INDICES
MEDIAN_COLS = [f"{b}_median" for b in ALL_BANDS]

HEIGHT_COLS = ["height_mean", "height_median", "height_std", "height_min", "height_max"]
TCD_COLS = ["tcd_mean", "tcd_median", "tcd_std", "tcd_min", "tcd_max"]
STRUCTURAL_COLS = HEIGHT_COLS + TCD_COLS

SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA"]  # MUU excluded - too few parcels (n=15) for a reliable Cohen's d, same as every prior EXP027/EXP028 analysis
SPECIES_LABEL_ET = {
    "HB": "haab (HB)", "KS": "kask (KS)", "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)", "LV": "lepp (LV)", "MA": "mänd (MA)",
}

MIN_REAL_N = 20
WINDOW_DAYS = 10
TOP_N = 5


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


def cohens_d(vals_a, vals_b):
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a < MIN_REAL_N or n_b < MIN_REAL_N:
        return None
    pooled_std = np.sqrt((vals_a.var() + vals_b.var()) / 2)
    if pooled_std == 0:
        return None
    return (vals_a.mean() - vals_b.mean()) / pooled_std


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_combined()
    combined = assign_windows(combined)
    print(f"Combined rows (6 species, all splits): {len(combined)}")
    print(f"10-day windows spanning the year: {combined['window_start'].nunique()}")

    # per-parcel, per-window median of each spectral column
    per_parcel_window = combined.groupby([ID_COL, "window_start"])[MEDIAN_COLS].median().reset_index()
    labels = combined[[ID_COL, TARGET_COL]].drop_duplicates()
    per_parcel_window = per_parcel_window.merge(labels, on=ID_COL, how="left")

    # merge in the constant structural features (height + TCD)
    height_soil = pd.read_csv(HEIGHT_SOIL_CSV)[[ID_COL] + HEIGHT_COLS]
    tcd = pd.read_csv(TCD_CSV)[[ID_COL] + TCD_COLS]
    per_parcel_window = per_parcel_window.merge(height_soil, on=ID_COL, how="left")
    per_parcel_window = per_parcel_window.merge(tcd, on=ID_COL, how="left")
    print(f"Per-parcel-window rows after structural merge: {len(per_parcel_window)}")

    all_candidate_cols = MEDIAN_COLS + STRUCTURAL_COLS

    # =====================================================
    # One-vs-rest Cohen's d per (species, window, feature)
    # =====================================================
    results = []
    windows_sorted = sorted(per_parcel_window["window_start"].unique())

    for window in windows_sorted:
        window_df = per_parcel_window[per_parcel_window["window_start"] == window]
        for col in all_candidate_cols:
            feature_name = col.replace("_median", "") if col in MEDIAN_COLS else col
            for species in SPECIES_ORDER:
                vals_species = window_df.loc[window_df[TARGET_COL] == species, col].dropna()
                vals_rest = window_df.loc[window_df[TARGET_COL] != species, col].dropna()
                d = cohens_d(vals_species, vals_rest)
                if d is None:
                    continue
                results.append({
                    "species": species, "feature": feature_name,
                    "feature_type": "structural" if col in STRUCTURAL_COLS else "spectral",
                    "window_start": window, "cohens_d": d, "abs_cohens_d": abs(d),
                    "n_species": len(vals_species), "n_rest": len(vals_rest),
                })

    ovr = pd.DataFrame(results)
    print(f"\nTotal (species, window, feature) rows computed: {len(ovr)}")

    # =====================================================
    # Top 5 per (species, window)
    # =====================================================
    top5_parts = []
    for species in SPECIES_ORDER:
        for window in windows_sorted:
            sub = ovr[(ovr["species"] == species) & (ovr["window_start"] == window)]
            if sub.empty:
                continue
            top5 = sub.sort_values("abs_cohens_d", ascending=False).head(TOP_N).reset_index(drop=True)
            top5["rank"] = top5.index + 1
            top5_parts.append(top5)

    top5_long = pd.concat(top5_parts, ignore_index=True)
    out_long = TABLES_DIR / "subregion2_top5_features_per_species_per_window.csv"
    top5_long.to_csv(out_long, index=False, encoding="utf-8-sig")
    print(f"Saved (long format): {out_long} ({len(top5_long)} rows)")

    # ---- wide format: one row per (species, window), columns rank1..rank5 ----
    wide_rows = []
    for (species, window), grp in top5_long.groupby(["species", "window_start"]):
        grp = grp.sort_values("rank")
        row = {
            "species": species,
            "window_start": window,
            "window_end": window + pd.Timedelta(days=WINDOW_DAYS - 1),
        }
        for _, r in grp.iterrows():
            row[f"rank{int(r['rank'])}_feature"] = r["feature"]
            row[f"rank{int(r['rank'])}_abs_cohens_d"] = round(r["abs_cohens_d"], 4)
            row[f"rank{int(r['rank'])}_type"] = r["feature_type"]
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows).sort_values(["species", "window_start"])
    out_wide = TABLES_DIR / "subregion2_top5_features_wide.csv"
    wide_df.to_csv(out_wide, index=False, encoding="utf-8-sig")
    print(f"Saved (wide format): {out_wide}")

    # =====================================================
    # Console summary: example windows + structural-feature frequency
    # =====================================================
    print("\n--- Example: first 3 windows for LV ---")
    for window in windows_sorted[:3]:
        row = wide_df[(wide_df["species"] == "LV") & (wide_df["window_start"] == window)]
        if row.empty:
            continue
        r = row.iloc[0]
        feats = [r.get(f"rank{i}_feature") for i in range(1, 6)]
        print(f"  LV {window.strftime('%d.%m.%Y')}-{r['window_end'].strftime('%d.%m.%Y')}: top 5 = {feats}")

    print("\n--- How often does a STRUCTURAL feature (height/TCD) make each species' top 5? ---")
    freq_rows = []
    for species in SPECIES_ORDER:
        sub = top5_long[top5_long["species"] == species]
        n_windows_total = sub["window_start"].nunique()
        n_windows_with_structural = sub[sub["feature_type"] == "structural"]["window_start"].nunique()
        pct = n_windows_with_structural / n_windows_total if n_windows_total else 0
        freq_rows.append({"species": species, "n_windows": n_windows_total,
                           "windows_with_structural_in_top5": n_windows_with_structural,
                           "pct": pct})
        print(f"  {species}: {n_windows_with_structural}/{n_windows_total} windows ({pct:.0%})")

    freq_df = pd.DataFrame(freq_rows)

    # which specific structural features appear at all, and how often (rank-1 only)
    print("\n--- Structural features that ever reach RANK 1 (most discriminating that window) ---")
    rank1_structural = top5_long[(top5_long["rank"] == 1) & (top5_long["feature_type"] == "structural")]
    if len(rank1_structural):
        print(rank1_structural.groupby(["species", "feature"]).size().sort_values(ascending=False).to_string())
    else:
        print("  (none - no structural feature ever ranked #1 in any species/window)")

    # =====================================================
    # Figure: structural-feature frequency per species
    # =====================================================
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#898781", length=0)
    ax.xaxis.grid(True, color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    freq_df_sorted = freq_df.sort_values("pct")
    labels = [SPECIES_LABEL_ET[s] for s in freq_df_sorted["species"]]
    ax.barh(labels, freq_df_sorted["pct"] * 100, color="#2a78d6", zorder=3)
    ax.set_xlabel("% aknaid, kus kõrgus- või TCD-tunnus top-5 hulgas", color="#898781")
    ax.set_title("Struktuursete tunnuste (kõrgus, TCD) sagedus liikide top-5 tunnuste hulgas",
                  fontsize=11, color="#0b0b0b", pad=12)
    fig.tight_layout()
    out_fig = FIGURES_DIR / "top5_structural_feature_frequency.png"
    fig.savefig(out_fig, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_fig}")

    print("\nDone.")


if __name__ == "__main__":
    main()
