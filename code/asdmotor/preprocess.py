"""Normalisation of skeleton sequences.

Three optional normalisations, each removing a specific nuisance factor:

``centre``
    Subtract the per-frame hip midpoint. Removes whole-body translation - which is
    crop-dependent in MMASD+ and distance-from-camera-dependent in Kinect data, and
    carries no motor information either way.

``scale``
    Divide by a per-sequence body-size scalar (median torso length). Children aged
    2-12 differ in stature by more than a factor of two; without this, body size is
    a free proxy for age and, in an ASD-vs-TD comparison, potentially for group.
    Scaling is per-sequence rather than per-frame so that genuine depth-related size
    change is not silently removed.

``rotate``
    Rotate into a body-fixed frame (hip-to-hip, hip-to-shoulder, their cross
    product). Removes camera azimuth, so the same movement recorded from a
    different angle yields the same features. For Kinect gait recordings, where the
    child walks toward the camera, this also removes the systematic yaw change
    across the walk.

None of these fit anything across sequences, so they cannot leak information
between subjects or splits. Transforms that *are* estimated from data (feature
standardisation) are fitted on training subjects only, at the modelling stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skeletons import MMASD_PLUS, Skeleton

#: Smallest body-scale accepted before a sequence is rejected as degenerate.
MIN_BODY_SCALE = 1e-3


@dataclass(frozen=True, slots=True)
class NormalisationSpec:
    centre: bool = True
    scale: bool = True
    rotate: bool = True

    @property
    def name(self) -> str:
        parts = [k for k in ("centre", "scale", "rotate") if getattr(self, k)]
        return "+".join(parts) if parts else "raw"


def hip_centre(coords: np.ndarray, skeleton: Skeleton = MMASD_PLUS) -> np.ndarray:
    """Per-frame pelvis midpoint, ``(T, 3)``."""
    return coords[:, skeleton.idx(*skeleton.hip_joints), :].mean(axis=1)


def shoulder_centre(coords: np.ndarray, skeleton: Skeleton = MMASD_PLUS) -> np.ndarray:
    """Per-frame shoulder midpoint, ``(T, 3)``."""
    return coords[:, skeleton.idx(*skeleton.shoulder_joints), :].mean(axis=1)


def body_scale(coords: np.ndarray, skeleton: Skeleton = MMASD_PLUS) -> float:
    """Per-sequence body size: the median pelvis-to-shoulder distance."""
    return float(
        np.median(
            np.linalg.norm(
                shoulder_centre(coords, skeleton) - hip_centre(coords, skeleton), axis=1
            )
        )
    )


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-12)


def body_frame(coords: np.ndarray, skeleton: Skeleton = MMASD_PLUS) -> np.ndarray:
    """Per-frame body-fixed orthonormal basis, ``(T, 3, 3)`` with rows = axes.

    ``e0`` medio-lateral (right hip to left hip), ``e1`` longitudinal (pelvis to
    shoulder centre, orthogonalised against ``e0``), ``e2 = e0 x e1``. Degenerate
    frames fall back to the identity, so a bad frame perturbs only itself.
    """
    left_hip, right_hip = skeleton.idx(*skeleton.hip_joints)
    lateral = coords[:, left_hip, :] - coords[:, right_hip, :]
    longitudinal = shoulder_centre(coords, skeleton) - hip_centre(coords, skeleton)

    e0 = _unit(lateral)
    proj = (longitudinal * e0).sum(axis=1, keepdims=True) * e0
    e1 = _unit(longitudinal - proj)
    e2 = _unit(np.cross(e0, e1))

    basis = np.stack([e0, e1, e2], axis=1)
    bad = ~np.isfinite(basis).all(axis=(1, 2)) | (np.linalg.norm(basis, axis=2) < 0.5).any(axis=1)
    basis[bad] = np.eye(3)
    return basis


def normalise(
    coords: np.ndarray, spec: NormalisationSpec, skeleton: Skeleton = MMASD_PLUS
) -> np.ndarray:
    """Apply *spec* to a ``(T, J, 3)`` sequence and return a new array.

    Raises:
        ValueError: if scaling is requested and the body scale is degenerate; the
            caller should drop the sample rather than divide by ~0.
    """
    out = np.asarray(coords, dtype=float).copy()
    if spec.centre:
        out = out - hip_centre(out, skeleton)[:, None, :]
    if spec.rotate:
        basis = body_frame(coords, skeleton)
        out = np.einsum("tab,tjb->tja", basis, out)
    if spec.scale:
        s = body_scale(coords, skeleton)
        if not np.isfinite(s) or s < MIN_BODY_SCALE:
            raise ValueError(f"degenerate body scale ({s!r}); sample cannot be normalised")
        out = out / s
    return out


def unpad(coords: np.ndarray, true_length: int) -> np.ndarray:
    """Drop repeat padding. See ``quality.detect_repeat_padding``."""
    if not 0 < true_length <= len(coords):
        raise ValueError(f"true_length {true_length} outside 1..{len(coords)}")
    return coords[:true_length]
