"""Clearcut-overlap filtering shared by the Family-A experiments that exclude
stands overlapping recent clearcuts by >20% (see
data/reference/clearcut_overlap/stands_overlapping_clearcuts_over20pct.csv).

Canonical version = the superset found in experiments/EXP011b_clean_no_clearcut/exp015.py:
validates that id_col exists in the overlap file and prints the per-split
class distribution after filtering. experiments/EXP019_auto_rf_search/autoRF.py
uses a structurally different per-dataframe/string-ID variant and is not
retrofitted to this shared version (only file with that shape).
"""

import pandas as pd


def remove_clearcut_ids(train_long, val_long, test_long, overlap_file, id_col, target_col):
    overlap = pd.read_csv(overlap_file)

    if id_col not in overlap.columns:
        raise ValueError(f"Overlap file is missing column '{id_col}'. Columns: {overlap.columns.tolist()}")

    remove_ids = set(
        overlap[id_col]
        .dropna()
        .astype(float)
        .tolist()
    )

    print("\nIDs to remove (>20% overlap with clearcuts):", len(remove_ids))

    def filter_df(df, name):
        before_rows = len(df)
        before_ids = df[id_col].nunique()

        df = df.copy()
        df[id_col] = df[id_col].astype(float)

        removed_ids = df[df[id_col].isin(remove_ids)][id_col].nunique()

        df = df[~df[id_col].isin(remove_ids)].copy()

        after_rows = len(df)
        after_ids = df[id_col].nunique()

        print(f"\n{name}:")
        print("IDs before:", before_ids)
        print("IDs removed:", removed_ids)
        print("IDs after:", after_ids)
        print("Rows before:", before_rows)
        print("Rows after:", after_rows)
        print("Class distribution after removal:")
        print(df[target_col].value_counts())

        return df

    train_long = filter_df(train_long, "TRAIN")
    val_long = filter_df(val_long, "VAL")
    test_long = filter_df(test_long, "TEST")

    return train_long, val_long, test_long
