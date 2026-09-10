"""Metrics and leakage-safe cross-validation splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


class LeakageError(AssertionError):
    """A split placed the same subject on both sides."""


@dataclass(frozen=True, slots=True)
class Fold:
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def subject_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 2,
    seed: int = 0,
) -> list[Fold]:
    """Repeated stratified group k-fold, grouped by subject.

    Stratification balances the class distribution across folds; grouping keeps
    every subject wholly on one side. Both are needed: grouping alone can produce
    folds missing a class entirely at this sample size.
    """
    folds: list[Fold] = []
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
            assert_no_subject_leakage(groups, tr, te)
            folds.append(Fold(repeat=r, fold=k, train_idx=tr, test_idx=te))
    return folds


def assert_no_subject_leakage(groups: np.ndarray, *index_sets: np.ndarray) -> None:
    """Raise if any subject appears in more than one of the given index sets."""
    seen: list[set] = [set(np.asarray(groups)[np.asarray(ix)]) for ix in index_sets]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            overlap = seen[i] & seen[j]
            if overlap:
                raise LeakageError(f"subjects present in two splits: {sorted(overlap)}")


def inner_folds(
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    n_splits: int = 3,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-aware inner folds *within* an outer training set, for model selection.

    Indices returned are positions into the full arrays, not into ``train_idx``.
    """
    y_tr, g_tr = y[train_idx], groups[train_idx]
    n_eff = min(n_splits, len(np.unique(g_tr)), int(np.min(np.bincount(y_tr)[np.bincount(y_tr) > 0])))
    n_eff = max(n_eff, 2)
    sgkf = StratifiedGroupKFold(n_splits=n_eff, shuffle=True, random_state=seed)
    out = []
    for tr, va in sgkf.split(np.zeros(len(y_tr)), y_tr, g_tr):
        a, b = train_idx[tr], train_idx[va]
        assert_no_subject_leakage(groups, a, b)
        out.append((a, b))
    return out


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, n_classes: int
) -> dict:
    """Full metric set. Binary tasks additionally get sensitivity/specificity."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n_classes))
        ).tolist(),
        "support": np.bincount(y_true, minlength=n_classes).tolist(),
    }
    # A fold can lack a class entirely at this sample size. sklearn returns NaN with
    # a warning rather than raising, so guard explicitly: the metric is omitted, not
    # recorded as a number that was never computed.
    try:
        if len(np.unique(y_true)) < 2:
            pass
        elif n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                              labels=list(range(n_classes)))
            )
    except ValueError:
        pass  # a fold can lack a class entirely; recorded as absent, never faked

    if n_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return out


def aggregate_folds(per_fold: list[dict]) -> dict:
    """Mean, standard deviation and per-fold spread of every scalar metric."""
    keys = [
        k for k in per_fold[0]
        if isinstance(per_fold[0][k], (int, float)) and not isinstance(per_fold[0][k], bool)
    ]
    out: dict = {}
    for k in keys:
        vals = np.array([f[k] for f in per_fold if k in f and f[k] == f[k]], dtype=float)
        if len(vals) == 0:
            continue
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "n_folds": int(len(vals)),
            "per_fold": [round(float(v), 5) for v in vals],
        }
    cms = [np.array(f["confusion_matrix"]) for f in per_fold if "confusion_matrix" in f]
    if cms:
        out["confusion_matrix_summed"] = np.sum(cms, axis=0).tolist()
    return out
