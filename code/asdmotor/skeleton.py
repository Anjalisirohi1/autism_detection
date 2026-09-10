"""Backwards-compatible MMASD+ skeleton constants.

The canonical definition now lives in :mod:`asdmotor.skeletons` as
``MMASD_PLUS``; this module re-exports it as module-level names so existing
imports keep working.
"""

from __future__ import annotations

from typing import Final

from .skeletons import MMASD_PLUS

JOINT_NAMES: Final = MMASD_PLUS.joints
AXES: Final = ("x", "y", "z")
COORD_COLUMNS: Final = MMASD_PLUS.coord_columns
LABEL_COLUMNS: Final = ("Action_Label", "ASD_Label")
N_JOINTS: Final = MMASD_PLUS.n_joints
N_COORDS: Final = N_JOINTS * 3
JOINT_INDEX: Final = MMASD_PLUS.index
BODY_REGIONS: Final = MMASD_PLUS.regions
SYMMETRIC_PAIRS: Final = MMASD_PLUS.symmetric_pairs
SKELETON_EDGES: Final = MMASD_PLUS.edges
RIGID_BONES: Final = MMASD_PLUS.rigid_bones
VERTICAL_CHAIN: Final = MMASD_PLUS.vertical_chain


def region_indices(region: str) -> tuple[int, ...]:
    """Column-block indices (joint indices) of a region."""
    return MMASD_PLUS.region_indices(region)


def edge_indices(edges: tuple[tuple[str, str], ...]) -> tuple[tuple[int, int], ...]:
    """Translate (name, name) edges into (index, index) pairs."""
    return MMASD_PLUS.edge_indices(edges)
