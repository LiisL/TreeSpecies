"""Model evaluation helpers shared by the Family-A and Family-B experiments.

`evaluate_model` (multiclass) and `plot_confusion_matrix_pretty` were extracted
from experiments/EXP000_baseline/baasmudel1406.py - `plot_confusion_matrix_pretty`
was previously EXP000-only (no other experiment called it). `evaluate_binary` is
the print-only variant shared by the EXP017/EXP018/EXP018b HB-vs-X binary
experiments; `evaluate_binary_profile` is EXP020's distinct dict-returning
variant (guards roc_auc_score with try/except and returns a metrics dict) -
kept as a separate function since it is not a byte-identical duplicate of
`evaluate_binary`, just a similarly-shaped one.

`evaluate_model`'s `target_names` param (optional, default None - preserves
Family-A's original behavior exactly) was added for the Family-B (EXP005-
EXP009) extraction, which always passed `target_names=le.classes_` to
`classification_report` so the report shows species codes instead of
LabelEncoder's integer classes. `save_feature_importance` is Family-B's other
shared block (build/sort/print/save a feature-importance table).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)


def evaluate_model(title, y_true, y_pred, target_names=None):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)

    print(f"\n================ {title} ================")
    print("Accuracy:", acc)
    print("Balanced Accuracy:", bal_acc)
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)
    print("Cohen's Kappa:", kappa)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa
    }


def save_feature_importance(feature_names, importances, out_path, top_n_print=30):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n_print} feature importance:")
    print(importance_df.head(top_n_print))

    importance_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\nFeature importance saved:", out_path)

    return importance_df


def plot_confusion_matrix_pretty(y_true, y_pred, title, out_path, normalize=False):
    """
    Saves a clean confusion matrix as a PNG for inclusion in the report.
    normalize=False -> shows object counts
    normalize=True  -> shows percentages per true class
    """

    labels = sorted(pd.Series(y_true).dropna().unique())

    if normalize:
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            normalize="true"
        )
        values_format = ".2f"
    else:
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=labels
        )
        values_format = "d"

    fig, ax = plt.subplots(figsize=(9, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=values_format,
        colorbar=True
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Ennustatud puuliik", fontsize=12)
    ax.set_ylabel("Tegelik puuliik", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved: {out_path}")


def evaluate_binary(title, y_true, pred, proba, positive_label, negative_label):
    print(f"\n================ {title} ================")
    print("Accuracy:", accuracy_score(y_true, pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, pred))
    print("F1:", f1_score(y_true, pred))
    print("ROC AUC:", roc_auc_score(y_true, proba))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            pred,
            target_names=[negative_label, positive_label],
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, pred))


def evaluate_binary_profile(title, y_true, pred, proba):
    print(f"\n================ {title} ================")
    print("Accuracy:", accuracy_score(y_true, pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, pred))
    print("F1:", f1_score(y_true, pred, zero_division=0))

    try:
        auc = roc_auc_score(y_true, proba)
    except ValueError:
        auc = np.nan

    print("ROC AUC:", auc)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            pred,
            target_names=["OTHER", "SPECIES"],
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, pred))

    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": auc
    }
