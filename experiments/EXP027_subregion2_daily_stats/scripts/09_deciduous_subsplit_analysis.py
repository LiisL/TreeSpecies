"""Deciduous-only sub-split analysis: within EXP031's stage 2b (HB, KS, LM,
LV - the four deciduous species), is there a data-driven way to further
split the group before final classification, rather than one flat 4-way
classifier? User's own hypothesis to check: soil type, height/size
differences, and statistically significant spectral signatures.

Three independent angles, all deciduous-only (not diluted by conifers):

  1. Spectral: (a) reuses EXP027 script 04's already-computed pairwise
     Cohen's d table (`subregion2_pairwise_separability.csv`), filtered to
     just the 6 deciduous-pair rows - answers "which PAIRS are hardest;
     does a natural 2+2 partition exist?" (b) NEW: one-vs-rest Cohen's d
     computed specifically within the 4-species deciduous subset (each
     species vs the OTHER THREE deciduous combined, not vs all 5 others
     like script 05 did) - answers "which SINGLE species is most cleanly
     peelable from the rest of the deciduous group?", a different question
     the existing 6-class one-vs-rest table can't answer directly since it
     dilutes each species' distinctiveness against 2 conifer classes too.
  2. Height: Kruskal-Wallis test (non-parametric - forestry height/age
     distributions are typically skewed, matches the median/IQR reporting
     style already used in script 07's forestry profile) on height_median
     across the 4 species, plus pairwise Mann-Whitney U for which specific
     pairs differ significantly.
  3. Soil: soil_majority has 370 distinct codes across ~9800 parcels (see
     EXP028 memory) - too sparse for a meaningful chi-square test with
     many near-empty cells. Reported descriptively instead: each species'
     top-5 modal soil codes and their overlap with the other 3 species'
     top-5 - answers "do these species occupy visibly different soil
     niches, or mostly the same ones?"

Same reliability guard as script 04 (MIN_REAL_N=20 real, non-imputed
observations required per species before a column is used in a "best
separator" ranking - avoids the imputation-zero-std Cohen's-d-divides-
by-zero bug caught while building that script).

Outputs:
  - tables/subregion2_deciduous_one_vs_rest_within_group.csv
  - tables/subregion2_deciduous_height_kruskal.csv
  - tables/subregion2_deciduous_soil_top_codes.csv
  - figures/deciduous_subsplit_height_boxplot.png
  - figures/deciduous_subsplit_one_vs_rest_bar.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
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
HEIGHT_SOIL_CSV = TABLES_DIR / "subregion2_parcel_height_soil.csv"

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

REFLECTANCE_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
VEGETATION_INDICES = ["NDVI", "NDRE", "NDMI", "NBR", "EVI", "CIRE"]
ALL_BANDS = REFLECTANCE_BANDS + VEGETATION_INDICES
TS_FEATURE_COLS = [f"{b}_mean" for b in ALL_BANDS] + [f"{b}_median" for b in ALL_BANDS]

DECIDUOUS = ["HB", "KS", "LM", "LV"]
SPECIES_LABEL_ET = {"HB": "haab (HB)", "KS": "kask (KS)", "LM": "lepp/must lepp (LM)", "LV": "hall lepp (LV)"}
MIN_REAL_N = 20

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
DARK_BLUE = "#184f95"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID_HAIRLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def load_combined_deciduous():
    frames = []
    for split in SPLITS:
        df = pd.read_csv(DAILY_CSV[split], usecols=[ID_COL, DATE_COL, TARGET_COL] + TS_FEATURE_COLS)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined[TARGET_COL].isin(DECIDUOUS)].copy()
    return combined


def parse_col(name):
    parts = name.split("_")
    month = int(parts[-2])
    day = int(parts[-1])
    year = int(parts[-3])
    stat = parts[-4]
    band = "_".join(parts[:-4])
    return band, stat, pd.Timestamp(year=year, month=month, day=day)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 1a. Reuse script 04's pairwise table - deciduous pairs only
    # =====================================================
    pairwise = pd.read_csv(TABLES_DIR / "subregion2_pairwise_separability.csv")
    pw_deciduous = pairwise[
        pairwise["species_a"].isin(DECIDUOUS) & pairwise["species_b"].isin(DECIDUOUS)
    ].sort_values("max_abs_cohens_d")
    print("=" * 70)
    print("1a. Pairwise separability among the 4 deciduous species (from script 04):")
    print("=" * 70)
    print(pw_deciduous.to_string(index=False))

    # =====================================================
    # 1b. NEW: one-vs-rest Cohen's d WITHIN deciduous only
    # =====================================================
    print("\n" + "=" * 70)
    print("1b. One-vs-rest Cohen's d, computed WITHIN the deciduous subset only")
    print("=" * 70)

    combined = load_combined_deciduous()
    print(f"Deciduous rows (all splits): {len(combined)}, parcels: {combined[ID_COL].nunique()}")

    wide = long_to_wide(combined, "deciduous_combined", ID_COL, TARGET_COL, DATE_COL, TS_FEATURE_COLS)
    feature_cols = [c for c in wide.columns if c not in (ID_COL, TARGET_COL)]
    feature_cols = [c for c in feature_cols if not wide[c].isna().all()]

    X = wide[feature_cols]
    y = wide[TARGET_COL]
    real_counts = X.notna().groupby(y.values).sum()

    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    ovr_results = []
    for sp in DECIDUOUS:
        is_sp = (y.values == sp)
        mean_sp = X_imp[is_sp].mean()
        std_sp = X_imp[is_sp].std()
        mean_rest = X_imp[~is_sp].mean()
        std_rest = X_imp[~is_sp].std()

        diff = mean_sp - mean_rest
        pooled_std = np.sqrt((std_sp ** 2 + std_rest ** 2) / 2)
        cohens_d = (diff / pooled_std).abs()

        real_n_sp = real_counts.loc[sp]
        real_n_rest = X.notna()[~is_sp].sum()  # "rest" pooled real-count per column
        reliable_mask = (real_n_sp >= MIN_REAL_N) & (real_n_rest >= MIN_REAL_N)
        cohens_d_reliable = cohens_d[reliable_mask]

        best_col = cohens_d_reliable.idxmax()
        band, stat, date = parse_col(best_col)
        ovr_results.append({
            "species": sp,
            "best_ovr_cohens_d": cohens_d_reliable.max(),
            "best_feature": best_col, "best_band": band, "best_stat": stat, "best_date": date,
            "n_reliable_columns": int(reliable_mask.sum()),
        })

    ovr_df = pd.DataFrame(ovr_results).sort_values("best_ovr_cohens_d", ascending=False)
    out_ovr = TABLES_DIR / "subregion2_deciduous_one_vs_rest_within_group.csv"
    ovr_df.to_csv(out_ovr, index=False, encoding="utf-8-sig")
    print(f"\n(higher = more cleanly 'peelable' as a single species vs the other 3 deciduous)")
    print(ovr_df.to_string(index=False))
    print(f"Saved: {out_ovr}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    style_axes(ax)
    order = ovr_df.sort_values("best_ovr_cohens_d")
    ax.barh([SPECIES_LABEL_ET[s] for s in order["species"]], order["best_ovr_cohens_d"], color=BLUE, zorder=3)
    ax.set_xlabel("Parim one-vs-rest |Cohen's d| (lehtpuurühma sees)", color=INK_MUTED)
    ax.set_title("Milline liik eristub kõige puhtamalt teistest lehtpuudest?", fontsize=11, color=INK_PRIMARY, pad=10)
    fig.tight_layout()
    out_fig1 = FIGURES_DIR / "deciduous_subsplit_one_vs_rest_bar.png"
    fig.savefig(out_fig1, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig1}")

    # =====================================================
    # 2. Height: Kruskal-Wallis + pairwise Mann-Whitney U
    # =====================================================
    print("\n" + "=" * 70)
    print("2. Height (height_median) across the 4 deciduous species")
    print("=" * 70)

    height_soil = pd.read_csv(HEIGHT_SOIL_CSV)
    # height_soil already carries its own 'target' column (built by EXP027
    # script 01) - no merge needed, just filter to the 4 deciduous species
    hs = height_soil[height_soil[TARGET_COL].isin(DECIDUOUS)].copy()
    print(f"Parcels with height+species (deciduous only): {len(hs)}")

    groups = {sp: hs.loc[hs[TARGET_COL] == sp, "height_median"].dropna() for sp in DECIDUOUS}
    for sp, g in groups.items():
        print(f"  {sp}: n={len(g)}, median={g.median():.2f}, IQR=[{g.quantile(.25):.2f}, {g.quantile(.75):.2f}]")

    h_stat, kw_p = kruskal(*groups.values())
    print(f"\nKruskal-Wallis across all 4 species: H={h_stat:.2f}, p={kw_p:.3e}")

    pairwise_height = []
    for i, sp_a in enumerate(DECIDUOUS):
        for sp_b in DECIDUOUS[i + 1:]:
            u_stat, p_val = mannwhitneyu(groups[sp_a], groups[sp_b], alternative="two-sided")
            median_diff = groups[sp_a].median() - groups[sp_b].median()
            pairwise_height.append({
                "species_a": sp_a, "species_b": sp_b,
                "median_a": groups[sp_a].median(), "median_b": groups[sp_b].median(),
                "median_diff": median_diff, "mannwhitney_u": u_stat, "p_value": p_val,
            })
    height_pair_df = pd.DataFrame(pairwise_height).sort_values("p_value")
    height_pair_df["significant_bonferroni"] = height_pair_df["p_value"] < (0.05 / len(height_pair_df))
    out_height = TABLES_DIR / "subregion2_deciduous_height_kruskal.csv"
    height_pair_df.to_csv(out_height, index=False, encoding="utf-8-sig")
    print("\nPairwise height comparisons (Bonferroni-corrected alpha = 0.05/6 = 0.0083):")
    print(height_pair_df.to_string(index=False))
    print(f"Saved: {out_height}")

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    style_axes(ax)
    box_data = [groups[sp] for sp in DECIDUOUS]
    bp = ax.boxplot(box_data, labels=[SPECIES_LABEL_ET[s] for s in DECIDUOUS], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.6)
    ax.set_ylabel("height_median (m)", color=INK_MUTED)
    ax.set_title(f"Kõrgus liikide kaupa (Kruskal-Wallis p={kw_p:.2e})", fontsize=11, color=INK_PRIMARY, pad=10)
    fig.tight_layout()
    out_fig2 = FIGURES_DIR / "deciduous_subsplit_height_boxplot.png"
    fig.savefig(out_fig2, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig2}")

    # =====================================================
    # 3. Soil: descriptive top-5 modal codes per species + overlap
    # =====================================================
    print("\n" + "=" * 70)
    print("3. Soil (soil_majority) top codes per species (descriptive - too sparse for chi-square)")
    print("=" * 70)

    soil_rows = []
    top_sets = {}
    for sp in DECIDUOUS:
        codes = hs.loc[hs[TARGET_COL] == sp, "soil_majority"].dropna()
        top5 = codes.value_counts(normalize=True).head(5)
        top_sets[sp] = set(top5.index)
        for code, frac in top5.items():
            soil_rows.append({"species": sp, "soil_code": code, "fraction_of_species": frac})
        print(f"\n{sp} top-5 soil codes (n={len(codes)}, {codes.nunique()} distinct codes):")
        print(top5.to_string())

    soil_df = pd.DataFrame(soil_rows)
    out_soil = TABLES_DIR / "subregion2_deciduous_soil_top_codes.csv"
    soil_df.to_csv(out_soil, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_soil}")

    print("\nTop-5 soil code overlap between species pairs (Jaccard on top-5 sets):")
    for i, sp_a in enumerate(DECIDUOUS):
        for sp_b in DECIDUOUS[i + 1:]:
            inter = len(top_sets[sp_a] & top_sets[sp_b])
            union = len(top_sets[sp_a] | top_sets[sp_b])
            jaccard = inter / union if union else 0.0
            print(f"  {sp_a}-{sp_b}: {inter}/{union} shared codes (Jaccard={jaccard:.2f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
