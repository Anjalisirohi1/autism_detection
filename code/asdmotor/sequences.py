"""Phase 3: fixed-length masked sequences, and leakage-safe normalisation.

Design decisions, and why.

**One sequence per sample; no sliding windows.** With 334 samples over 32 subjects,
overlapping windows would multiply `n` without adding information, and windows drawn
from one source clip are near-duplicates of each other. Under subject-wise grouping
they cannot cross a split, so they inflate apparent precision rather than real
precision, and they are the most likely route to a leakage artefact. If windowing is
ever adopted, :func:`window_sequences` exists so that window-level and sample-level
results can be compared directly under the same grouping, and both must be reported.

**Explicit masking, never synthetic frames.** The dataset authors padded every file
to 179 rows by cyclically repeating its own leading rows. Phase 1 recovers the true
length exactly; those synthetic frames are stripped, and the sequence is re-padded
here with zeros **plus a mask channel**. Models consume the mask. A model must never
see a cyclically repeated frame as real motion.

**Normalisation is fitted per fold, on training subjects only.**
:class:`SequenceStandardiser` takes the training row indices explicitly and computes
statistics over unmasked frames of those rows alone. There is no global-fit code path
in this module, so a leaking pipeline cannot be written by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SequenceSet:
    """Padded, masked sequences with aligned metadata.

    Attributes:
        values: ``(N, T, D)`` float32, zero-padded past each sequence's true length.
        mask: ``(N, T)`` bool, True for real frames.
        lengths: ``(N,)`` int, true frame counts.
        sample_ids: ``(N,)`` sample identifiers, aligned with the metadata table.
        feature_names: ``(D,)`` column names.
    """

    values: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    sample_ids: np.ndarray
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n, t, d = self.values.shape
        if self.mask.shape != (n, t):
            raise ValueError(f"mask {self.mask.shape} != values {(n, t)}")
        if len(self.lengths) != n or len(self.sample_ids) != n:
            raise ValueError("lengths/sample_ids not aligned with values")
        if len(self.feature_names) != d:
            raise ValueError(f"{len(self.feature_names)} names for {d} features")
        if not np.array_equal(self.mask.sum(axis=1), self.lengths):
            raise ValueError("mask does not agree with lengths")
        if np.any(self.values[~self.mask] != 0):
            raise ValueError("padded positions must be exactly zero")

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[2])


def pad_sequences(
    chunks: list[np.ndarray],
    sample_ids: list[str],
    feature_names: tuple[str, ...],
    max_length: int | None = None,
) -> SequenceSet:
    """Right-pad ragged ``(T_i, D)`` blocks to a common length with a mask."""
    if not chunks:
        raise ValueError("no sequences given")
    lengths = np.array([len(c) for c in chunks], dtype=np.int64)
    T = int(max_length if max_length is not None else lengths.max())
    D = chunks[0].shape[1]

    values = np.zeros((len(chunks), T, D), dtype=np.float32)
    mask = np.zeros((len(chunks), T), dtype=bool)
    for i, c in enumerate(chunks):
        if c.shape[1] != D:
            raise ValueError(f"sequence {i} has {c.shape[1]} features, expected {D}")
        k = min(len(c), T)
        values[i, :k] = c[:k]
        mask[i, :k] = True
    return SequenceSet(values, mask, np.minimum(lengths, T), np.array(sample_ids, dtype=object),
                       feature_names)


class SequenceStandardiser:
    """Per-feature z-scoring fitted on training rows only.

    The training row indices are a required argument of :meth:`fit`; there is no
    way to call it without naming them, which is the point.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_fit_rows_: int | None = None

    def fit(self, values: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> "SequenceStandardiser":
        """Fit on unmasked frames of ``values[train_idx]`` and nothing else."""
        train_idx = np.asarray(train_idx)
        if train_idx.size == 0:
            raise ValueError("train_idx is empty")
        v = values[train_idx]
        m = mask[train_idx]
        flat = v[m]                              # (n_real_train_frames, D)
        if len(flat) == 0:
            raise ValueError("no unmasked training frames to fit on")
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.n_fit_rows_ = int(len(train_idx))
        return self

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("SequenceStandardiser.transform called before fit")
        out = (values - self.mean_) / (self.std_ + self.eps)
        out = out.astype(np.float32)
        out[~mask] = 0.0                         # keep padding at exactly zero
        return out

    def fit_transform(self, values, mask, train_idx):  # pragma: no cover - convenience
        return self.fit(values, mask, train_idx).transform(values, mask)


def aggregate_statistics(
    values: np.ndarray, mask: np.ndarray, feature_names: tuple[str, ...]
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Collapse each sequence to per-feature summary statistics.

    This is the non-temporal representation used by the aggregate-statistics
    baseline in Phase 4: if a sequence model cannot beat it, temporal structure is
    not being exploited.

    Returns ``(N, 6D)`` with mean, std, min, max, median and mean absolute
    frame-to-frame change, computed over real frames only.
    """
    n, _, d = values.shape
    out = np.zeros((n, 6 * d), dtype=np.float32)
    for i in range(n):
        v = values[i, mask[i]]
        if len(v) == 0:
            continue
        diff = np.abs(np.diff(v, axis=0)).mean(axis=0) if len(v) > 1 else np.zeros(d)
        out[i] = np.concatenate([v.mean(0), v.std(0), v.min(0), v.max(0),
                                 np.median(v, 0), diff])
    names = tuple(
        f"{stat}__{f}"
        for stat in ("mean", "std", "min", "max", "median", "absdiff")
        for f in feature_names
    )
    return out, names


def window_sequences(
    seq: SequenceSet, window: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut fixed windows out of real (unmasked) frames.

    Not used by the default pipeline - see the module docstring. Provided so that a
    window-level analysis can be run and compared against the sample-level one under
    identical subject grouping.

    Returns ``(windows, window_lengths, source_row_index)``.
    """
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    out, src = [], []
    for i in range(seq.n_samples):
        v = seq.values[i, seq.mask[i]]
        for start in range(0, max(len(v) - window + 1, 0), stride):
            out.append(v[start:start + window])
            src.append(i)
    if not out:
        raise ValueError(f"no windows of length {window} fit any sequence")
    w = np.stack(out).astype(np.float32)
    return w, np.full(len(w), window, dtype=np.int64), np.array(src, dtype=np.int64)
