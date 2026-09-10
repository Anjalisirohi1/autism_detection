"""Per-file data-quality metrics for skeleton samples.

Two problems motivated these checks in the MMASD+ release, and both are measured
rather than assumed. They are applied unchanged to any other dataset so that a
claim like "sensor skeletons are cleaner than inferred ones" is tested, not asserted.

1. **Repeat padding.** MMASD+ pads short clips to a fixed length by cyclically
   repeating the file's own leading rows. :func:`detect_repeat_padding` recovers the
   true length exactly.

2. **Geometric incoherence.** In a genuine 3D pose sequence, rigid bone lengths are
   near-constant within a clip. :func:`geometry_metrics` measures the coefficient of
   variation of those lengths, the recovery of the anatomical skeleton by a
   rigidity-weighted minimum spanning tree, and axis degeneracy (x == y == z, which
   real coordinates essentially never satisfy).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from .skeletons import MMASD_PLUS, Skeleton


def detect_repeat_padding(coords: np.ndarray) -> int:
    """Return the true (pre-padding) frame count of ``coords`` ``(T, J, 3)``.

    Cyclic repeat padding of true length ``k`` satisfies ``coords[k:] ==
    coords[:T - k]`` exactly. The smallest such ``k`` is the true length; ``T`` is
    returned when no repeat structure is present.
    """
    flat = coords.reshape(len(coords), -1)
    n = len(flat)
    for k in range(1, n):
        if np.array_equal(flat[k:], flat[: n - k]):
            return k
    return n


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    """Label-free measures of whether a file encodes a coherent 3D skeleton."""

    bone_cv_median: float
    bone_cv_max: float
    mst_anatomical_edges: int
    mst_possible_edges: int
    axis_degenerate_fraction: float
    vertical_chain_score: float
    hip_centre_norm: float
    torso_length_median: float
    coord_min: float
    coord_max: float
    coord_abs_p99: float
    static_fraction: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _pairwise_distance_cv(coords: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = d.std(axis=0) / (d.mean(axis=0) + 1e-12)
    return np.nan_to_num(cv, nan=0.0, posinf=0.0)


def geometry_metrics(coords: np.ndarray, skeleton: Skeleton = MMASD_PLUS) -> GeometryMetrics:
    """Compute geometric-integrity metrics for one ``(T, J, 3)`` sample."""
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"expected (T, J, 3) array, got {coords.shape}")
    if coords.shape[1] != skeleton.n_joints:
        raise ValueError(
            f"{skeleton.name} expects {skeleton.n_joints} joints, got {coords.shape[1]}"
        )
    if len(coords) < 2:
        raise ValueError("need at least two frames")

    rigid = skeleton.edge_indices(skeleton.rigid_bones)
    lengths = np.stack(
        [np.linalg.norm(coords[:, a] - coords[:, b], axis=1) for a, b in rigid], axis=1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        bone_cv = lengths.std(axis=0) / (lengths.mean(axis=0) + 1e-12)
    bone_cv = np.nan_to_num(bone_cv, nan=0.0, posinf=0.0)

    cv_matrix = _pairwise_distance_cv(coords)
    np.fill_diagonal(cv_matrix, 0.0)
    mst = minimum_spanning_tree(cv_matrix).toarray()
    mst_edges = {frozenset((int(i), int(j))) for i, j in zip(*np.nonzero(mst))}
    true_edges = {frozenset(e) for e in skeleton.edge_indices(skeleton.edges)}
    n_anatomical = len(mst_edges & true_edges)

    med = np.median(coords, axis=0)
    chain = skeleton.idx(*skeleton.vertical_chain)
    chain_y = med[list(chain), 1] * skeleton.vertical_sign
    chain_score = float(np.mean(np.diff(chain_y) > 0)) if len(chain_y) > 1 else 0.0

    hips = coords[:, skeleton.idx(*skeleton.hip_joints), :].mean(axis=1)
    shoulders = coords[:, skeleton.idx(*skeleton.shoulder_joints), :].mean(axis=1)
    torso = np.linalg.norm(shoulders - hips, axis=1)

    spread = np.abs(coords - coords.mean(axis=2, keepdims=True)).max(axis=2)
    degenerate = float((spread < 0.02).mean())

    frame_motion = np.abs(np.diff(coords.reshape(len(coords), -1), axis=0)).max(axis=1)
    static = float((frame_motion < 1e-9).mean())

    return GeometryMetrics(
        bone_cv_median=float(np.median(bone_cv)),
        bone_cv_max=float(bone_cv.max()),
        mst_anatomical_edges=int(n_anatomical),
        mst_possible_edges=int(skeleton.n_joints - 1),
        axis_degenerate_fraction=degenerate,
        vertical_chain_score=chain_score,
        hip_centre_norm=float(np.linalg.norm(np.median(hips, axis=0))),
        torso_length_median=float(np.median(torso)),
        coord_min=float(coords.min()),
        coord_max=float(coords.max()),
        coord_abs_p99=float(np.percentile(np.abs(coords), 99)),
        static_fraction=static,
    )


@dataclass(frozen=True, slots=True)
class StructuralCriterion:
    """Sensor-agnostic validity test, as opposed to a noise-sensitive one.

    ``bone_cv_median`` conflates two different things: whether the data encodes a
    body at all, and how noisy the sensor is. That is fine when comparing files
    from one source, but it does not transfer across sensors - a threshold
    calibrated on MediaPipe world landmarks rejects perfectly valid Kinect
    recordings, whose depth noise sets a higher CV floor on short clips.

    These four checks test *structure* instead, and each fails only for data that
    is not a human body:

    * the head-to-foot chain is ordered correctly in every file;
    * x, y and z are not degenerate (real 3D points are not on the diagonal);
    * the reconstructed torso is a physically possible length;
    * rigid bone lengths are stable to well within an order of magnitude.

    ``bone_cv_max`` is therefore set generously - it is a backstop against gross
    incoherence, not a noise threshold.
    """

    min_vertical_chain_score: float = 1.0
    max_axis_degenerate_fraction: float = 0.01
    torso_length_range_m: tuple[float, float] = (0.20, 0.70)
    max_bone_cv_median: float = 0.15

    def failures(self, m: GeometryMetrics) -> tuple[str, ...]:
        out: list[str] = []
        if m.vertical_chain_score < self.min_vertical_chain_score:
            out.append("vertical_chain_out_of_order")
        if m.axis_degenerate_fraction > self.max_axis_degenerate_fraction:
            out.append("axis_degenerate")
        lo, hi = self.torso_length_range_m
        if not (lo <= m.torso_length_median <= hi):
            out.append("implausible_torso_length")
        if m.bone_cv_median > self.max_bone_cv_median:
            out.append("unstable_bone_lengths")
        return tuple(out)

    def passes(self, m: GeometryMetrics) -> bool:
        return not self.failures(m)


def integrity_grade(
    metrics: GeometryMetrics, *, clean_threshold: float, suspect_threshold: float
) -> str:
    """Grade a file as ``coherent`` / ``suspect`` / ``incoherent``.

    Driven by rigid-bone stability, the least assumption-laden signal available: it
    needs no knowledge of units, axis order or origin.
    """
    if metrics.bone_cv_median <= clean_threshold:
        return "coherent"
    if metrics.bone_cv_median <= suspect_threshold:
        return "suspect"
    return "incoherent"
