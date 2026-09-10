"""Frame-level motor-function features, for any skeleton specification.

Two feature families are kept strictly separate so experiments can compare them:

``pose``
    Normalised joint coordinates only - the static-pose baseline.

``motor``
    Engineered kinematics: speed, acceleration, jerk, signed displacement, joint
    angles and angular velocity, trunk orientation and postural sway, and left/right
    symmetry.

Both are produced per body region, so region ablations use the same code path as the
whole-body run. The joint set, regions, angles and symmetry pairs come from a
:class:`~asdmotor.skeletons.Skeleton`, so the same code serves MediaPipe landmarks
and Kinect skeletons without modification.

Feature choice is grounded in the autism motor literature rather than "compute
everything"; rationale and mathematics are in ``docs/FEATURE_DEFINITIONS.md``.

Units and time. Derivatives are **per frame**. MMASD+ reports a variable 25-30 fps
and no per-clip rate, so converting there would inject a silent scale error; the
Al-Jubouri recordings do carry timestamps (30.3 fps median), and their sampling
interval is recorded in the sample metadata so per-second conversion is available
downstream when wanted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import hip_centre, shoulder_centre
from .skeletons import MMASD_PLUS, Skeleton


@dataclass(frozen=True, slots=True)
class FeatureBlock:
    """A ``(T, D)`` frame-level feature matrix with named columns."""

    values: np.ndarray
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.names):
            raise ValueError(
                f"values {self.values.shape} inconsistent with {len(self.names)} names"
            )

    @property
    def n_frames(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at ``b`` of the chain a-b-c, in radians, ``(T,)``."""
    u, v = a - b, c - b
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    denom = nu * nv
    cos = np.divide((u * v).sum(axis=1), denom, out=np.zeros(len(u)), where=denom > 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _diff(x: np.ndarray) -> np.ndarray:
    """First difference with the first frame repeated, preserving length."""
    d = np.diff(x, axis=0)
    return np.concatenate([d[:1], d], axis=0) if len(d) else np.zeros_like(x)


def pose_features(
    coords: np.ndarray, region: str = "whole_body", skeleton: Skeleton = MMASD_PLUS
) -> FeatureBlock:
    """Normalised joint coordinates for *region*: the static-pose baseline."""
    joints = skeleton.regions[region]
    idx = list(skeleton.idx(*joints))
    values = coords[:, idx, :].reshape(len(coords), -1)
    names = tuple(f"pos_{j}_{ax}" for j in joints for ax in ("x", "y", "z"))
    return FeatureBlock(values, names)


def motor_features(
    coords: np.ndarray,
    region: str = "whole_body",
    skeleton: Skeleton = MMASD_PLUS,
    *,
    world_coords: np.ndarray | None = None,
) -> FeatureBlock:
    """Engineered frame-level motor-function features for *region*.

    Args:
        coords: ``(T, J, 3)`` normalised sequence, padding already removed.
        region: key of ``skeleton.regions``.
        skeleton: joint-set specification.
        world_coords: optional ``(T, J, 3)`` sequence in a **world-aligned** frame
            (centred and scaled, but not rotated into the body frame). Trunk
            orientation is only defined there: the body-frame rotation pins the
            trunk to its own axis by construction, which would make the feature a
            constant. Defaults to *coords*.
    """
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(f"expected (T, J, 3), got {coords.shape}")
    if coords.shape[1] != skeleton.n_joints:
        raise ValueError(
            f"{skeleton.name} expects {skeleton.n_joints} joints, got {coords.shape[1]}"
        )
    if region not in skeleton.regions:
        raise KeyError(f"unknown region {region!r} for {skeleton.name}")
    world = coords if world_coords is None else np.asarray(world_coords, dtype=float)
    if world.shape != coords.shape:
        raise ValueError(f"world_coords {world.shape} != coords {coords.shape}")

    joints = skeleton.regions[region]
    idx = list(skeleton.idx(*joints))
    T = len(coords)
    ix = skeleton.index

    cols: list[np.ndarray] = []
    names: list[str] = []

    # --- per-joint kinematics -------------------------------------------------
    vel = _diff(coords)
    acc = _diff(vel)
    jerk = _diff(acc)
    for arr, tag in ((vel, "speed"), (acc, "accel"), (jerk, "jerk")):
        cols.append(np.linalg.norm(arr[:, idx, :], axis=2))
        names.extend(f"{tag}_{j}" for j in joints)

    cols.append(vel[:, idx, :].reshape(T, -1))
    names.extend(f"disp_{j}_{ax}" for j in joints for ax in ("x", "y", "z"))

    # --- joint angles and angular velocity -----------------------------------
    for angle_name, (a, b, c) in skeleton.angle_triplets:
        if region != "whole_body" and skeleton.angle_region[angle_name] != region:
            continue
        theta = _angle(coords[:, ix[a]], coords[:, ix[b]], coords[:, ix[c]])
        cols.append(theta[:, None]); names.append(f"angle_{angle_name}")
        cols.append(_diff(theta)[:, None]); names.append(f"angvel_{angle_name}")

    # --- trunk orientation / postural control ---------------------------------
    # Torso properties, withheld from other regions so the region ablation compares
    # regions rather than leaking torso information into every block. Computed in
    # the world-aligned frame: the body-frame rotation would pin the trunk to its
    # own axis by construction.
    if region in ("torso", "whole_body"):
        lean = shoulder_centre(world, skeleton) - hip_centre(world, skeleton)
        n = np.linalg.norm(lean, axis=1)
        # inclination from vertical; vertical_sign makes one formula serve both
        # y-down (MediaPipe world) and y-up (Kinect camera space) conventions
        up = -skeleton.vertical_sign * lean[:, 1]
        incl = np.arccos(np.clip(np.divide(up, n, out=np.zeros(T), where=n > 1e-12), -1.0, 1.0))
        cols.append(incl[:, None]); names.append("angle_trunk_inclination")
        cols.append(_diff(incl)[:, None]); names.append("angvel_trunk_inclination")

        sway = lean - np.median(lean, axis=0)
        cols.append(sway); names.extend(("sway_trunk_x", "sway_trunk_y", "sway_trunk_z"))
        horiz = [a for a in (0, 1, 2) if a != 1]
        cols.append(np.linalg.norm(sway[:, horiz], axis=1)[:, None])
        names.append("sway_trunk_horizontal")

    # --- left/right symmetry --------------------------------------------------
    region_joints = set(joints)
    speed_all = np.linalg.norm(vel, axis=2)
    lat = skeleton.lateral_axis
    for left, right in skeleton.symmetric_pairs:
        if region != "whole_body" and not ({left, right} & region_joints):
            continue
        sl, sr = speed_all[:, ix[left]], speed_all[:, ix[right]]
        denom = sl + sr
        si = np.divide(sl - sr, denom, out=np.zeros(T), where=denom > 1e-12)
        cols.append(si[:, None]); names.append(f"symmetry_speed_{left}_{right}")

        dl = np.abs(coords[:, ix[left], lat])
        dr = np.abs(coords[:, ix[right], lat])
        denom = dl + dr
        cols.append(np.divide(dl - dr, denom, out=np.zeros(T), where=denom > 1e-12)[:, None])
        names.append(f"symmetry_lateral_{left}_{right}")

    values = np.concatenate([c if c.ndim == 2 else c[:, None] for c in cols], axis=1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return FeatureBlock(values, tuple(names))


FEATURE_SETS = ("pose", "motor")


def build_features(
    coords: np.ndarray,
    feature_set: str,
    region: str,
    skeleton: Skeleton = MMASD_PLUS,
    *,
    world_coords: np.ndarray | None = None,
) -> FeatureBlock:
    """Dispatch to a named feature set."""
    if feature_set == "pose":
        return pose_features(coords, region, skeleton)
    if feature_set == "motor":
        return motor_features(coords, region, skeleton, world_coords=world_coords)
    raise KeyError(f"unknown feature set {feature_set!r}; known: {sorted(FEATURE_SETS)}")
