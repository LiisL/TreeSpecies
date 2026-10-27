"""Class separability analysis: for every (band/index, date) column, how
well does it discriminate between the six main tree species - independent
of any trained model?

This is NOT feature engineering (constructing new derived variables -
that's experiments/EXP028_daily_full_features_height_soil/
signature_feature_engineering.py). It's a form of univariate feature
SELECTION / statistical class-separability analysis, using two classic,
model-free tools:

  1. ANOVA F-statistic (sklearn.feature_selection.f_classif) per column,
     across all 6 species groups at once - "how much does this one
     variable vary between species, relative to within a species?"
     Complements (doesn't duplicate) the RF feature importance already
     computed in EXP028 step 2: RF importance is entangled with tree-
     splitting behaviour and interactions between features; F-statistic
     is a clean, independent, per-variable statistic. Comparing the two
     is a useful cross-check.
  2. Pairwise Cohen's d (standardized mean difference) between every pair
     of species, per column - "which band/date best tells species A and
     species B apart, specifically?" This connects directly to the
     confusion-matrix patterns already seen throughout EXP028 (e.g. HB
     is repeatedly confused with KS/LV).

Uses all three splits combined (train+val+test) for maximum statistical
power - this is a descriptive/statistical analysis, not a train/test
evaluation, so there's no leakage concern in combining them (same
reasoning EXP027's signature curves already used).

Outputs:
  - tables/subregion2_separability_by_date_band.csv   (F-statistic per column, ranked)
  - tables/subregion2_pairwise_separability.csv         (best Cohen's d per species pair)
  - figures/separability_reflectance_bands.png           (F-statistic curves, 10 bands)
  - figures/separability_vegetation_indices.png           (F-statistic curves, 6 indices)
  - figures/separability_top30_datetime_band.png          (ranked bar chart)
  - figures/separability_pairwise_heatmap.png              (species x species max |Cohen's d|)
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.paths import PATHS
from src.preprocessing.timeseries import long_to_wide

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
TS_FEATURE_COLS = [f"{b}_mean" for b in ALL_BANDS] + [f"{b}_median" for b in ALL_BANDS]

SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA"]
SPECIES_LABEL_ET = {
    "HB": "haab (HB)", "KS": "kask (KS)", "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)", "LV": "lepp (LV)", "MA": "mänd (MA)",
}

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
DARK_BLUE = "#184f95"

EST_MONTHS = ["", "Jaan", "Veebr", "Märts", "Apr", "Mai", "Juuni",
              "Juuli", "Aug", "Sept", "Okt", "Nov", "Dets"]


def load_combined():
    frames = []
    for split in SPLITS:
        df = pd.read_csv(DAILY_CSV[split], usecols=[ID_COL, DATE_COL, TARGET_COL] + TS_FEATURE_COLS)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined[TARGET_COL].isin(SPECIES_ORDER)].copy()
    return combined


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=8)
    ax.yaxis.grid(True, color=GRID_HAIRLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def style_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: EST_MONTHS[mdates.num2date(x).month]))


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    combined = load_combined()
    print(f"Combined rows (6 main species, all splits): {len(combined)}")
    print(f"Parcels: {combined[ID_COL].nunique()}")

    # ---- pivot to one row per parcel, one column per (band/index, date) ----
    wide = long_to_wide(combined, "combined", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
    feature_cols = [c for c in wide.columns if c not in (ID_COL, TARGET_COL)]
    feature_cols = [c for c in feature_cols if not wide[c].isna().all()]
    print(f"(band, date) columns after pivoting: {len(feature_cols)}")

    X = wide[feature_cols]
    y = wide[TARGET_COL]

    # Real (non-imputed) observation count per species per column - needed to
    # guard against a real bug found while building this: for sparse dates, a
    # whole species group can have ZERO real observations, so after
    # median-imputation every one of its parcels gets the exact same value,
    # std becomes exactly 0, and Cohen's d divides by zero -> "inf". That's an
    # imputation artifact, not a real separability finding, so any column
    # where a relevant species has too few real observations is excluded
    # from being reported as a "best" separator (still gets imputed and used
    # for the overall F-statistic/ANOVA computation itself - only the
    # RANKING/reporting step excludes it).
    MIN_REAL_N = 20
    real_counts = X.notna().groupby(y.values).sum()  # species x column

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    X_imp_df = pd.DataFrame(X_imp, columns=feature_cols)

    # =====================================================
    # 1. ANOVA F-statistic per column (overall, all 6 species at once)
    # =====================================================

    f_stat, p_val = f_classif(X_imp_df, y)

    def parse_col(name):
        parts = name.split("_")
        month = int(parts[-2])
        day = int(parts[-1])
        year = int(parts[-3])
        stat = parts[-4]
        band = "_".join(parts[:-4])
        return band, stat, pd.Timestamp(year=year, month=month, day=day)

    parsed = [parse_col(c) for c in feature_cols]
    min_real_n_all_species = real_counts.min(axis=0)

    sep_table = pd.DataFrame({
        "feature": feature_cols,
        "band": [p[0] for p in parsed],
        "stat": [p[1] for p in parsed],
        "date": [p[2] for p in parsed],
        "f_statistic": f_stat,
        "p_value": p_val,
        "min_real_n_across_species": [min_real_n_all_species[c] for c in feature_cols],
    }).sort_values("f_statistic", ascending=False)

    n_unreliable = (sep_table["min_real_n_across_species"] < MIN_REAL_N).sum()
    print(f"\n{n_unreliable} of {len(sep_table)} columns have a species with fewer than "
          f"{MIN_REAL_N} real observations on that date - excluded from rankings/figures below "
          f"(kept in the saved CSV, flagged via min_real_n_across_species).")

    out_table = TABLES_DIR / "subregion2_separability_by_date_band.csv"
    sep_table.to_csv(out_table, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_table}")

    sep_reliable = sep_table[sep_table["min_real_n_across_species"] >= MIN_REAL_N].copy()

    print("\nTop 20 most separating (band, date) combinations (reliable only):")
    print(sep_reliable.head(20).to_string(index=False))

    print("\nMean F-statistic, mean vs median columns (reliable only):")
    print(sep_reliable.groupby("stat")["f_statistic"].mean())

    # =====================================================
    # 2. Pairwise Cohen's d per column, per species pair
    # =====================================================

    means = X_imp_df.groupby(y.values).mean()
    stds = X_imp_df.groupby(y.values).std()

    pair_results = []
    species_list = SPECIES_ORDER
    for i, sp_a in enumerate(species_list):
        for sp_b in species_list[i + 1:]:
            diff = means.loc[sp_a] - means.loc[sp_b]
            pooled_std = np.sqrt((stds.loc[sp_a] ** 2 + stds.loc[sp_b] ** 2) / 2)
            cohens_d = (diff / pooled_std).abs()

            reliable_mask = (real_counts.loc[sp_a] >= MIN_REAL_N) & (real_counts.loc[sp_b] >= MIN_REAL_N)
            cohens_d_reliable = cohens_d[reliable_mask]

            best_col = cohens_d_reliable.idxmax()
            band, stat, date = parse_col(best_col)
            pair_results.append({
                "species_a": sp_a, "species_b": sp_b,
                "max_abs_cohens_d": cohens_d_reliable.max(),
                "best_feature": best_col, "best_band": band, "best_stat": stat, "best_date": date,
                "n_reliable_columns": int(reliable_mask.sum()),
            })

    pair_df = pd.DataFrame(pair_results).sort_values("max_abs_cohens_d")
    out_pairs = TABLES_DIR / "subregion2_pairwise_separability.csv"
    pair_df.to_csv(out_pairs, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_pairs}")
    print("\nHardest-to-separate species pairs (lowest max achievable |Cohen's d|):")
    print(pair_df.head(6).to_string(index=False))

    # =====================================================
    # Figure A: F-statistic curves over the year, by band (mean + median)
    # =====================================================

    def plot_separability_grid(bands, title, out_name, ncols, nrows, figsize):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150, sharex=True)
        axes = axes.flatten()

        for ax, band in zip(axes, bands):
            style_axes(ax)
            style_date_axis(ax)
            for stat, color, ls in [("mean", BLUE, "-"), ("median", DARK_BLUE, "--")]:
                sub = sep_reliable[(sep_reliable["band"] == band) & (sep_reliable["stat"] == stat)].sort_values("date")
                ax.plot(sub["date"], sub["f_statistic"], color=color, linestyle=ls,
                         linewidth=1.3, marker="o", markersize=2.5, alpha=0.9, label=stat)
            ax.set_title(band, fontsize=10, color=INK_PRIMARY, loc="left")

        for ax in axes[len(bands):]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(title, fontsize=13, color=INK_PRIMARY, y=1.0)
        fig.text(0.0, 0.5, "ANOVA F-statistik (puuliikide eristatavus)", rotation=90, va="center", ha="left",
                  fontsize=9, color=INK_MUTED)
        fig.tight_layout(rect=[0.02, 0.03, 1, 0.98])

        out = FIGURES_DIR / out_name
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    plot_separability_grid(
        REFLECTANCE_BANDS,
        "Puuliikide eristatavus aja jooksul - Sentinel-2 peegelduskanalid",
        "separability_reflectance_bands.png",
        ncols=2, nrows=5, figsize=(10, 16),
    )
    plot_separability_grid(
        VEGETATION_INDICES,
        "Puuliikide eristatavus aja jooksul - vegetatsiooniindeksid",
        "separability_vegetation_indices.png",
        ncols=2, nrows=3, figsize=(10, 10),
    )

    # =====================================================
    # Figure B: top-30 (band, date) ranked bar chart
    # =====================================================

    top30 = sep_reliable.head(30).iloc[::-1].copy()
    top30["label"] = top30["band"] + " (" + top30["stat"] + ") " + top30["date"].dt.strftime("%d.%m")

    fig, ax = plt.subplots(figsize=(8, 9), dpi=150)
    style_axes(ax)
    ax.barh(top30["label"], top30["f_statistic"], color=BLUE, zorder=3)
    ax.set_xlabel("ANOVA F-statistik", color=INK_MUTED)
    ax.set_title("30 kõige rohkem puuliike eristavat (kanal, kuupäev) paari", fontsize=12, color=INK_PRIMARY, pad=12)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    out = FIGURES_DIR / "separability_top30_datetime_band.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # =====================================================
    # Figure C: pairwise max-separability heatmap
    # =====================================================

    heat = pd.DataFrame(np.nan, index=SPECIES_ORDER, columns=SPECIES_ORDER)
    for _, row in pair_df.iterrows():
        heat.loc[row["species_a"], row["species_b"]] = row["max_abs_cohens_d"]
        heat.loc[row["species_b"], row["species_a"]] = row["max_abs_cohens_d"]

    labels = [SPECIES_LABEL_ET[s] for s in SPECIES_ORDER]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    im = ax.imshow(heat.values.astype(float), cmap="Blues", vmin=0)

    ax.set_xticks(range(len(SPECIES_ORDER)))
    ax.set_yticks(range(len(SPECIES_ORDER)))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=INK_MUTED)
    ax.set_yticklabels(labels, color=INK_MUTED)

    for i in range(len(SPECIES_ORDER)):
        for j in range(len(SPECIES_ORDER)):
            val = heat.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmax(heat.values) * 0.6 else INK_PRIMARY
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    ax.set_title("Liigipaaride parim saavutatav eristatavus (max |Cohen's d|)", fontsize=12, color=INK_PRIMARY, pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, label="max |Cohen's d|")
    fig.tight_layout()
    out = FIGURES_DIR / "separability_pairwise_heatmap.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
