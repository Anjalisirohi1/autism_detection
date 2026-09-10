"""Age- and sex-matched subsampling, and demographic confound accounting.

In the Al-Jubouri corpus the ASD children are on average ~1.9 years older than the
TD children (8.58 vs 6.66) and far more likely to be male (83 % vs 52 %). Either
gap is large enough that a classifier could score well by detecting "older" or
"male" and never touch motor function at all.

This module supplies the two things needed to stop that being a caveat and make it
a measurement:

* :func:`match_on_age_sex` builds the largest exactly-sex-matched, closely
  age-matched subset, so the primary comparison can be repeated on a subsample
  where the confounds are removed by construction;
* :func:`demographic_features` supplies age and sex as a standalone feature block,
  so a baseline using *only* demographics can be run. If that baseline performs
  near the motor models, the headline result is confounded and must be reported
  that way.

Matching is a maximum-cardinality minimum-cost assignment: sex must match exactly,
age must be within a tolerance, and among feasible pairings the total absolute age
difference is minimised. Being optimal rather than greedy matters - a greedy pass
strands pairs that a global solution would keep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_INFEASIBLE = 1e6


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of an age/sex matching attempt."""

    index: np.ndarray            # positions into the original frame, both groups
    pairs: list[tuple[int, int]]  # (positive_pos, negative_pos)
    tolerance_years: float
    summary: dict

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_samples(self) -> int:
        return int(len(self.index))


def match_on_age_sex(
    meta: pd.DataFrame,
    tolerance_years: float = 1.0,
    group_col: str = "group",
    positive: str = "ASD",
    negative: str = "TD",
    age_col: str = "age_years",
    sex_col: str = "sex",
) -> MatchResult:
    """Largest sex-exact, age-close 1:1 matched subset of *meta*.

    Rows with a missing age or sex cannot be matched and are excluded.

    Raises:
        ValueError: if no feasible pair exists at this tolerance - the caller
            should report that rather than silently returning an empty corpus.
    """
    for col in (group_col, age_col, sex_col):
        if col not in meta.columns:
            raise ValueError(f"metadata has no column {col!r}")

    usable = meta[age_col].notna() & meta[sex_col].astype(str).str.strip().ne("")
    pos = np.flatnonzero(usable & meta[group_col].eq(positive).to_numpy())
    neg = np.flatnonzero(usable & meta[group_col].eq(negative).to_numpy())
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(f"need members of both {positive!r} and {negative!r} to match")

    age = meta[age_col].to_numpy(dtype=float)
    sex = meta[sex_col].astype(str).str.upper().str.strip().to_numpy()

    cost = np.full((len(pos), len(neg)), _INFEASIBLE)
    for i, pi in enumerate(pos):
        gap = np.abs(age[neg] - age[pi])
        ok = (sex[neg] == sex[pi]) & (gap <= tolerance_years)
        cost[i, ok] = gap[ok]

    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(pos[r]), int(neg[c])) for r, c in zip(rows, cols) if cost[r, c] < _INFEASIBLE]
    if not pairs:
        raise ValueError(
            f"no age/sex-matched pair exists within {tolerance_years} years; "
            f"matched analysis is not possible on this cohort"
        )

    index = np.array(sorted([i for p in pairs for i in p]), dtype=int)
    sub = meta.iloc[index]
    summary = {
        "tolerance_years": tolerance_years,
        "n_pairs": len(pairs),
        "n_samples": int(len(index)),
        "excluded_missing_demographics": int((~usable).sum()),
        "age_mean_by_group": {
            g: round(float(x[age_col].mean()), 3) for g, x in sub.groupby(group_col)
        },
        "age_gap_years": round(
            abs(float(sub[sub[group_col] == positive][age_col].mean())
                - float(sub[sub[group_col] == negative][age_col].mean())), 4
        ),
        "age_gap_years_before_matching": round(
            abs(float(meta[meta[group_col] == positive][age_col].mean())
                - float(meta[meta[group_col] == negative][age_col].mean())), 4
        ),
        "sex_counts_by_group": {
            g: {k: int(v) for k, v in x[sex_col].str.upper().value_counts().items()}
            for g, x in sub.groupby(group_col)
        },
        "max_pair_age_difference": round(
            float(max(abs(age[a] - age[b]) for a, b in pairs)), 3
        ),
    }
    return MatchResult(index=index, pairs=pairs, tolerance_years=tolerance_years,
                       summary=summary)


def demographic_features(
    meta: pd.DataFrame, age_col: str = "age_years", sex_col: str = "sex"
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Age and sex only, as an ``(N, 2)`` block for the demographic baseline.

    Missing ages are filled with the corpus median so the baseline still runs; a
    baseline is only meaningful if it never fails.
    """
    age = pd.to_numeric(meta[age_col], errors="coerce")
    age = age.fillna(age.median() if age.notna().any() else 0.0).to_numpy(dtype=float)
    male = meta[sex_col].astype(str).str.upper().str.strip().eq("M").to_numpy(dtype=float)
    return np.column_stack([age, male]), ("age_years", "is_male")


def fold_composition(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    group_col: str = "group",
    positive: str = "ASD",
    negative: str = "TD",
    age_col: str = "age_years",
    sex_col: str = "sex",
) -> dict:
    """Per-fold demographic composition, so imbalance is visible per split.

    Returns scalars only, so the values survive fold aggregation instead of being
    averaged away or dropped.
    """
    out: dict[str, float] = {}
    for tag, ix in (("train", train_idx), ("test", test_idx)):
        part = meta.iloc[ix]
        pos = part[part[group_col] == positive]
        neg = part[part[group_col] == negative]
        age = pd.to_numeric(part[age_col], errors="coerce")
        male = part[sex_col].astype(str).str.upper().str.strip().eq("M")
        out[f"{tag}_age_mean"] = float(age.mean()) if age.notna().any() else float("nan")
        out[f"{tag}_frac_male"] = float(male.mean()) if len(part) else float("nan")
        for name, g in ((positive.lower(), pos), (negative.lower(), neg)):
            a = pd.to_numeric(g[age_col], errors="coerce")
            m = g[sex_col].astype(str).str.upper().str.strip().eq("M")
            out[f"{tag}_age_mean_{name}"] = float(a.mean()) if a.notna().any() else float("nan")
            out[f"{tag}_frac_male_{name}"] = float(m.mean()) if len(g) else float("nan")
            out[f"{tag}_n_{name}"] = int(len(g))
        gap = out.get(f"{tag}_age_mean_{positive.lower()}", float("nan")) - out.get(
            f"{tag}_age_mean_{negative.lower()}", float("nan"))
        out[f"{tag}_age_gap_years"] = float(gap)
    return out
