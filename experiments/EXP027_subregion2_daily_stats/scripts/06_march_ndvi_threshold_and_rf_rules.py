"""Domain-expert hypothesis check: "if NDVI is high in March, it's most
likely a conifer" - find the actual data threshold, and show the actual
threshold(s) the trained EXP028 Random Forest already uses.

A decision tree IS a sequence of threshold rules ("if NDVI > x, go left").
Random Forest is 400 of these voting. sklearn's RandomForestClassifier has
no API to hand it a fixed rule ("always split at 0.8") - but the actual
thresholds it already learned can be read directly out of the trained
model's internal tree structure, and compared against both the expert's
guess and the empirically optimal split.

Three things, in order:
  1. Per-parcel March NDVI (whole-month average, robust to single-date
     cloud gaps) vs. species - does the conifer/broadleaf split actually
     show up, and how large?
  2. A single-feature, depth-1 decision tree (the simplest possible
     "threshold rule") fit on March NDVI to separate conifer (KU, MA) from
     broadleaf (HB, KS, LM, LV) - this finds the empirically optimal
     threshold directly, comparable to the expert's ~0.8 guess.
  3. Every March-NDVI split threshold actually used across all 400 trees
     of the already-trained EXP028 "everything" model
     (models/trained/EXP028_rf_daily_full_features_height_soil.joblib,
     the only saved EXP028 model with individual March-NDVI columns in its
     feature set) - the real answer to "how does Random Forest decide".

Outputs:
  - tables/subregion2_march_ndvi_by_species.csv
  - figures/march_ndvi_by_species_boxplot.png
  - figures/march_ndvi_threshold_stump.png
  - figures/march_ndvi_rf_actual_thresholds.png
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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
FULL_MODEL_FILE = PATHS.models_trained / "EXP028_rf_daily_full_features_height_soil.joblib"

ID_COL = "id"
TARGET_COL = "target"
DATE_COL = "date_from"

SPECIES_ORDER = ["HB", "KS", "KU", "LM", "LV", "MA"]
CONIFERS = ["KU", "MA"]
BROADLEAF = ["HB", "KS", "LM", "LV"]
SPECIES_LABEL_ET = {
    "HB": "haab (HB)", "KS": "kask (KS)", "KU": "kuusk (KU)",
    "LM": "sanglepp (LM)", "LV": "lepp (LV)", "MA": "mänd (MA)",
}

INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID_HAIRLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
GREEN = "#008300"
SPECIES_COLOR = {"HB": "#2a78d6", "KS": "#eb6834", "KU": "#1baf7a", "LM": "#eda100", "LV": "#e87ba4", "MA": "#008300"}


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

    # =====================================================
    # 1. Per-parcel March NDVI
    # =====================================================

    frames = []
    for split in SPLITS:
        df = pd.read_csv(DAILY_CSV[split], usecols=[ID_COL, DATE_COL, TARGET_COL, "NDVI_median"])
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined[TARGET_COL].isin(SPECIES_ORDER)].copy()
    combined[DATE_COL] = pd.to_datetime(combined[DATE_COL])

    march = combined[combined[DATE_COL].dt.month == 3]
    march_ndvi = march.groupby(ID_COL)["NDVI_median"].median().rename("march_ndvi")
    labels = combined[[ID_COL, TARGET_COL]].drop_duplicates().set_index(ID_COL)
    per_parcel = labels.join(march_ndvi).dropna()

    print(f"Parcels with a March NDVI value: {len(per_parcel)}")
    print("\nMarch NDVI by species (median [IQR]):")
    summary = per_parcel.groupby(TARGET_COL)["march_ndvi"].describe()[["count", "25%", "50%", "75%"]]
    print(summary)

    out_table = TABLES_DIR / "subregion2_march_ndvi_by_species.csv"
    per_parcel.reset_index().to_csv(out_table, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_table}")

    # ---- boxplot by species ----
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    style_axes(ax)
    data = [per_parcel.loc[per_parcel[TARGET_COL] == s, "march_ndvi"] for s in SPECIES_ORDER]
    bp = ax.boxplot(data, labels=[SPECIES_LABEL_ET[s] for s in SPECIES_ORDER], patch_artist=True,
                     showfliers=False, medianprops={"color": INK_PRIMARY})
    for patch, s in zip(bp["boxes"], SPECIES_ORDER):
        patch.set_facecolor(SPECIES_COLOR[s])
        patch.set_alpha(0.75)
    ax.set_ylabel("Märtsi NDVI (kuu mediaan)", color=INK_MUTED)
    ax.set_title("Märtsikuu NDVI puuliigiti", fontsize=12, color=INK_PRIMARY, pad=12)
    fig.tight_layout()
    out = FIGURES_DIR / "march_ndvi_by_species_boxplot.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # =====================================================
    # 2. Empirically optimal threshold: depth-1 decision tree
    # =====================================================

    per_parcel = per_parcel.copy()
    per_parcel["is_conifer"] = per_parcel[TARGET_COL].isin(CONIFERS).astype(int)

    X = per_parcel[["march_ndvi"]].values
    y = per_parcel["is_conifer"].values

    stump = DecisionTreeClassifier(max_depth=1, random_state=42)
    stump.fit(X, y)

    threshold = stump.tree_.threshold[0]
    acc = stump.score(X, y)

    pred = stump.predict(X)
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\nEmpirically optimal single-threshold rule (conifer=KU+MA vs. broadleaf=HB+KS+LM+LV):")
    print(f"  March NDVI > {threshold:.3f} -> predict conifer")
    print(f"  Accuracy: {acc:.3f}, Sensitivity (conifer recall): {sensitivity:.3f}, Specificity (broadleaf recall): {specificity:.3f}")
    print(f"  (User's expert guess was ~0.8, for comparison)")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    style_axes(ax)
    ax.hist(per_parcel.loc[per_parcel["is_conifer"] == 1, "march_ndvi"], bins=40, alpha=0.65,
            color=GREEN, label="okaspuud (KU, MA)", density=True)
    ax.hist(per_parcel.loc[per_parcel["is_conifer"] == 0, "march_ndvi"], bins=40, alpha=0.65,
            color=BLUE, label="lehtpuud (HB, KS, LM, LV)", density=True)
    ax.axvline(threshold, color=INK_PRIMARY, linestyle="--", linewidth=1.5,
               label=f"leitud lävi: {threshold:.2f}")
    ax.set_xlabel("Märtsi NDVI", color=INK_MUTED)
    ax.set_ylabel("Tihedus", color=INK_MUTED)
    ax.set_title(f"Optimaalne lävi okaspuu vs. lehtpuu eristamiseks (täpsus {acc:.1%})",
                 fontsize=12, color=INK_PRIMARY, pad=12)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGURES_DIR / "march_ndvi_threshold_stump.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # =====================================================
    # 3. What thresholds does the ACTUAL trained RF use?
    # =====================================================

    bundle = joblib.load(FULL_MODEL_FILE)
    rf = bundle["model"]
    feature_cols = bundle["feature_cols"]

    march_ndvi_idx = [i for i, f in enumerate(feature_cols) if f.startswith("NDVI") and "_03_" in f]
    print(f"\nMarch-NDVI columns in the trained 'everything' model: {[feature_cols[i] for i in march_ndvi_idx]}")

    rf_thresholds = []
    for est in rf.estimators_:
        tree = est.tree_
        for node in range(tree.node_count):
            if tree.feature[node] in march_ndvi_idx:
                rf_thresholds.append(tree.threshold[node])

    rf_thresholds = np.array(rf_thresholds)
    print(f"\nSplits made on a March-NDVI column, across all {len(rf.estimators_)} trees: {len(rf_thresholds)}")
    if len(rf_thresholds) > 0:
        print(f"  Median threshold used: {np.median(rf_thresholds):.3f}")
        print(f"  25th-75th percentile: {np.percentile(rf_thresholds, 25):.3f} - {np.percentile(rf_thresholds, 75):.3f}")

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        style_axes(ax)
        ax.hist(rf_thresholds, bins=30, color=BLUE, alpha=0.85, zorder=3)
        ax.axvline(threshold, color=INK_PRIMARY, linestyle="--", linewidth=1.5,
                   label=f"lihtsa mudeli lävi: {threshold:.2f}")
        ax.axvline(np.median(rf_thresholds), color=GREEN, linestyle="-", linewidth=1.5,
                   label=f"RF mediaan-lävi: {np.median(rf_thresholds):.2f}")
        ax.set_xlabel("Lävend, mida RF puud märtsi-NDVI tunnusel kasutavad", color=INK_MUTED)
        ax.set_ylabel("Sagedus (üle kõigi puude)", color=INK_MUTED)
        ax.set_title("Milliseid läve väärtusi Random Forest tegelikult kasutab", fontsize=12, color=INK_PRIMARY, pad=12)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        out = FIGURES_DIR / "march_ndvi_rf_actual_thresholds.png"
        fig.savefig(out, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
    else:
        print("  No trees split on a March-NDVI column - the RF preferred other features for this split.")


if __name__ == "__main__":
    main()
