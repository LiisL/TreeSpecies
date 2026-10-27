"""Manual class oversampling shared by the Family-A experiments.

Byte-identical across all 7 occurrences it was found in (EXP011, EXP011b,
EXP012, EXP013, EXP014b, EXP021, EXP022) - a straight move, zero behavioral risk.
"""

import pandas as pd


def manual_oversample(X, y, sampling_strategy, target_col, random_state=42):
    """
    Manual oversampling:
    - takes X and y as DataFrame/Series
    - only upsamples classes for which sampling_strategy gives a larger count
    - does not need the imblearn package
    """
    df = X.copy()
    df[target_col] = y.values

    parts = []

    for cls, group in df.groupby(target_col):
        current_n = len(group)
        target_n = sampling_strategy.get(cls, current_n)

        if target_n > current_n:
            sampled = group.sample(
                n=target_n,
                replace=True,
                random_state=random_state
            )
        else:
            sampled = group

        parts.append(sampled)

    out = pd.concat(parts, axis=0).sample(
        frac=1,
        random_state=random_state
    )

    y_out = out[target_col].copy()
    X_out = out.drop(columns=[target_col]).copy()

    return X_out, y_out
