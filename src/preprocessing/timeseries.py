"""Shared long-to-wide / AUX-merge scaffold used across the Family-A experiment scripts.

Extracted from experiments/EXP000_baseline/baasmudel1406.py (the most complete
instance of the scaffold) and cross-checked against the other 19 Family-A
experiments. Canonicalized to the safest/most-complete variant found in each
case (always validates required columns, always logs dropped columns) since
none of the historical experiments actually exercised the divergent error
paths - see docs/AUDIT_REPORT.md / EXPERIMENTS.md for context.
"""

import pandas as pd
import numpy as np


def get_timeseries_feature_cols(train_long, id_col, target_col, date_col, exclude_cols=None):
    exclude = {id_col, target_col, date_col, "date_from", "date_to", "split", "subregion"}
    if exclude_cols:
        exclude |= set(exclude_cols)

    return [
        col for col in train_long.columns
        if col not in exclude
        and pd.api.types.is_numeric_dtype(train_long[col])
    ]


def long_to_wide(df, name, id_col, target_col, date_col, ts_feature_cols):
    needed = [id_col, target_col, date_col] + ts_feature_cols
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y_%m_%d")

    dup_keys = df.duplicated(subset=[id_col, date_col], keep=False)
    if dup_keys.any():
        n_dup = df.loc[dup_keys, [id_col, date_col]].drop_duplicates().shape[0]
        raise ValueError(
            f"{name}: {n_dup} duplicate ({id_col}, {date_col}) combinations - "
            f"pivot_table(aggfunc='first') would silently drop rows"
        )

    labels = df[[id_col, target_col]].drop_duplicates()

    wide = df.pivot_table(
        index=id_col,
        columns=date_col,
        values=ts_feature_cols,
        aggfunc="first"
    )

    wide.columns = [
        f"{feature}_{date}"
        for feature, date in wide.columns
    ]

    wide = wide.reset_index()
    wide = wide.merge(labels, on=id_col, how="left")

    return wide


def aggregate_daily_to_windows(df, id_col, date_col, target_col, feature_cols, window_dates, window_days=10):
    """Aggregates long-format DAILY rows (one row per id+calendar-day, e.g.
    experiments/EXP047_composition_daily_statistical's download output)
    into the same fixed N-day windows used throughout the rest of the
    project (EXP029/032/038/043's DATES list) via a per-window mean, so
    daily-granularity downloads can reuse long_to_wide() and the
    species-signature meta-features unchanged - those hardcode specific
    window-start date strings (e.g. "NDVI_mean_2025_05_01") that a raw
    daily table would essentially never contain (satellite passes rarely
    land on an exact target date).

    window_dates need not be contiguous or evenly spaced - each window is
    handled independently as its own [start, start+window_days) slice, so
    a real gap in the DATES list (e.g. 2025-03-12 to 2025-04-01, present
    in the existing 26-date list) is simply a period with no output rows,
    not an error.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    labels_df = df[[id_col, target_col]].drop_duplicates(subset=[id_col])

    frames = []
    for w in window_dates:
        start = pd.Timestamp(w)
        end = start + pd.Timedelta(days=window_days)
        sub = df.loc[(df[date_col] >= start) & (df[date_col] < end)]
        if sub.empty:
            continue
        agg = sub.groupby(id_col)[feature_cols].mean().reset_index()
        agg[date_col] = w
        frames.append(agg)

    if not frames:
        raise ValueError("No rows fell into any window - check window_dates against the data's date range.")

    out = pd.concat(frames, ignore_index=True)
    out = out.merge(labels_df, on=id_col, how="left")
    return out


def merge_aux(wide_df, aux_df, name, id_col, numeric_cols, categorical_cols):
    keep_cols = [id_col] + numeric_cols + categorical_cols
    missing = [c for c in keep_cols if c not in aux_df.columns]

    if missing:
        raise ValueError(f"{name}: AUX file is missing columns {missing}")

    aux_sub = aux_df[keep_cols].drop_duplicates(subset=[id_col]).copy()

    merged = wide_df.merge(
        aux_sub,
        on=id_col,
        how="left"
    )

    print(f"\nAUX merge check ({name}):")
    for col in numeric_cols + categorical_cols:
        print(f"{col}: missing {merged[col].isna().sum()}")

    return merged


def encode_categoricals_consistently(train, val, test, categorical_cols):
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["__source__"] = "train"
    val["__source__"] = "val"
    test["__source__"] = "test"

    combined = pd.concat([train, val, test], ignore_index=True)

    for col in categorical_cols:
        combined[col] = combined[col].fillna("MISSING").astype(str)

    combined = pd.get_dummies(
        combined,
        columns=categorical_cols,
        prefix=categorical_cols
    )

    train_out = combined[combined["__source__"] == "train"].drop(columns="__source__").copy()
    val_out = combined[combined["__source__"] == "val"].drop(columns="__source__").copy()
    test_out = combined[combined["__source__"] == "test"].drop(columns="__source__").copy()

    return train_out, val_out, test_out


def align_columns(train, val, test, id_col, target_col):
    feature_cols = [
        c for c in train.columns
        if c not in [id_col, target_col]
    ]

    for df in [val, test]:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = np.nan

    train = train[[id_col] + feature_cols + [target_col]]
    val = val[[id_col] + feature_cols + [target_col]]
    test = test[[id_col] + feature_cols + [target_col]]

    return train, val, test, feature_cols


def drop_all_nan_train_columns(X_train, X_val, X_test):
    valid_cols = X_train.columns[~X_train.isna().all()].tolist()
    dropped_cols = [c for c in X_train.columns if c not in valid_cols]

    print("\nFully empty columns in train:", len(dropped_cols))

    if dropped_cols:
        for c in dropped_cols:
            print("Dropping:", c)

    return (
        X_train[valid_cols].copy(),
        X_val[valid_cols].copy(),
        X_test[valid_cols].copy(),
        valid_cols
    )
