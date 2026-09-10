"""Skeleton specifications, so the feature pipeline is representation-agnostic.

Phase 2/3 code was written against the MMASD+ MediaPipe landmark set. Moving the
project's primary dataset to Al-Jubouri's Kinect v2 recordings means a different
joint set, a different vertical convention and a different origin - but the same
motor features. Rather than duplicating the pipeline, the joint set is lifted into
a :class:`Skeleton` value object that every stage takes as an argument.

Nothing here is invented: each joint list is the source dataset's own column names,
and each topology is the sensor vendor's published skeleton restricted to those
joints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Skeleton:
    """Everything the feature pipeline needs to know about a joint set.

    Attributes:
        name: identifier used in configs and manifests.
        joints: joint names in coordinate-column order.
        regions: body region -> member joints. Must include ``whole_body``.
        symmetric_pairs: (left, right) pairs for symmetry features.
        edges: skeleton connectivity, for topology recovery.
        rigid_bones: edges whose length is genuinely constant in a human body.
        angle_triplets: (name, (proximal, vertex, distal)) three-point angles.
        angle_region: angle name -> the region it belongs to.
        vertical_chain: head-to-foot joints, for the anatomical ordering check.
        vertical_sign: +1 if the vertical axis increases downward (image-style,
            MediaPipe world landmarks), -1 if it increases upward (Kinect camera
            space). Used so one ordering check serves both conventions.
        hip_joints / shoulder_joints: the two joints whose midpoint defines the
            pelvis and shoulder centres.
        lateral_axis: index of the medio-lateral axis in the body frame.
        units: free text, recorded in manifests.
    """

    name: str
    joints: tuple[str, ...]
    regions: dict[str, tuple[str, ...]]
    symmetric_pairs: tuple[tuple[str, str], ...]
    edges: tuple[tuple[str, str], ...]
    rigid_bones: tuple[tuple[str, str], ...]
    angle_triplets: tuple[tuple[str, tuple[str, str, str]], ...]
    angle_region: dict[str, str]
    vertical_chain: tuple[str, ...]
    vertical_sign: int
    hip_joints: tuple[str, str]
    shoulder_joints: tuple[str, str]
    units: str
    lateral_axis: int = 0

    @property
    def n_joints(self) -> int:
        return len(self.joints)

    @property
    def index(self) -> dict[str, int]:
        return {j: i for i, j in enumerate(self.joints)}

    @property
    def coord_columns(self) -> tuple[str, ...]:
        return tuple(f"{j}_{a}" for j in self.joints for a in ("x", "y", "z"))

    def idx(self, *names: str) -> tuple[int, ...]:
        ix = self.index
        return tuple(ix[n] for n in names)

    def region_indices(self, region: str) -> tuple[int, ...]:
        try:
            joints = self.regions[region]
        except KeyError as exc:
            raise KeyError(
                f"unknown body region {region!r}; known: {sorted(self.regions)}"
            ) from exc
        return self.idx(*joints)

    def edge_indices(self, edges: tuple[tuple[str, str], ...]) -> tuple[tuple[int, int], ...]:
        ix = self.index
        return tuple((ix[a], ix[b]) for a, b in edges)

    def validate(self) -> None:
        """Fail loudly on a malformed specification."""
        known = set(self.joints)
        if len(known) != len(self.joints):
            raise ValueError(f"{self.name}: duplicate joint names")
        if "whole_body" not in self.regions:
            raise ValueError(f"{self.name}: regions must include 'whole_body'")
        for r, js in self.regions.items():
            missing = set(js) - known
            if missing:
                raise ValueError(f"{self.name}: region {r} references unknown joints {missing}")
        for group in (self.edges, self.rigid_bones, self.symmetric_pairs):
            for a, b in group:
                if a not in known or b not in known:
                    raise ValueError(f"{self.name}: edge ({a}, {b}) references unknown joints")
        for aname, (a, b, c) in self.angle_triplets:
            if {a, b, c} - known:
                raise ValueError(f"{self.name}: angle {aname} references unknown joints")
            if aname not in self.angle_region:
                raise ValueError(f"{self.name}: angle {aname} has no region assignment")
        if set(self.vertical_chain) - known:
            raise ValueError(f"{self.name}: vertical_chain references unknown joints")
        if self.vertical_sign not in (1, -1):
            raise ValueError(f"{self.name}: vertical_sign must be +1 or -1")
        for j in self.hip_joints + self.shoulder_joints:
            if j not in known:
                raise ValueError(f"{self.name}: hip/shoulder joint {j} unknown")


# ---------------------------------------------------------------------------
# MMASD+ - MediaPipe BlazePose subset, 25 landmarks, world landmarks (y down)
# ---------------------------------------------------------------------------
_MMASD_JOINTS: Final = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index",
    "Left_hip", "Right_hip", "Left_knee", "Right_knee",
    "Left_ankle", "Right_ankle", "Left_heel", "Right_heel",
    "Left_foot", "Right_foot",
)

MMASD_PLUS = Skeleton(
    name="mmasd_plus_mediapipe",
    joints=_MMASD_JOINTS,
    regions={
        "head": ("nose", "left_eye", "right_eye", "left_ear", "right_ear"),
        "torso": ("left_shoulder", "right_shoulder", "Left_hip", "Right_hip"),
        "left_arm": ("left_shoulder", "left_elbow", "left_wrist", "left_pinky", "left_index"),
        "right_arm": ("right_shoulder", "right_elbow", "right_wrist", "right_pinky", "right_index"),
        "left_leg": ("Left_hip", "Left_knee", "Left_ankle", "Left_heel", "Left_foot"),
        "right_leg": ("Right_hip", "Right_knee", "Right_ankle", "Right_heel", "Right_foot"),
        "whole_body": _MMASD_JOINTS,
    },
    symmetric_pairs=(
        ("left_eye", "right_eye"), ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"), ("left_pinky", "right_pinky"),
        ("left_index", "right_index"), ("Left_hip", "Right_hip"),
        ("Left_knee", "Right_knee"), ("Left_ankle", "Right_ankle"),
        ("Left_heel", "Right_heel"), ("Left_foot", "Right_foot"),
    ),
    edges=(
        ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"),
        ("right_eye", "right_ear"), ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_wrist", "left_pinky"), ("left_wrist", "left_index"),
        ("left_pinky", "left_index"), ("right_wrist", "right_pinky"),
        ("right_wrist", "right_index"), ("right_pinky", "right_index"),
        ("left_shoulder", "Left_hip"), ("right_shoulder", "Right_hip"),
        ("Left_hip", "Right_hip"), ("Left_hip", "Left_knee"),
        ("Left_knee", "Left_ankle"), ("Left_ankle", "Left_heel"),
        ("Left_ankle", "Left_foot"), ("Left_heel", "Left_foot"),
        ("Right_hip", "Right_knee"), ("Right_knee", "Right_ankle"),
        ("Right_ankle", "Right_heel"), ("Right_ankle", "Right_foot"),
        ("Right_heel", "Right_foot"),
    ),
    rigid_bones=(
        ("left_shoulder", "right_shoulder"), ("Left_hip", "Right_hip"),
        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
        ("Left_hip", "Left_knee"), ("Right_hip", "Right_knee"),
        ("Left_knee", "Left_ankle"), ("Right_knee", "Right_ankle"),
        ("left_shoulder", "Left_hip"), ("right_shoulder", "Right_hip"),
    ),
    angle_triplets=(
        ("left_elbow", ("left_shoulder", "left_elbow", "left_wrist")),
        ("right_elbow", ("right_shoulder", "right_elbow", "right_wrist")),
        ("left_shoulder", ("left_elbow", "left_shoulder", "Left_hip")),
        ("right_shoulder", ("right_elbow", "right_shoulder", "Right_hip")),
        ("left_hip", ("left_shoulder", "Left_hip", "Left_knee")),
        ("right_hip", ("right_shoulder", "Right_hip", "Right_knee")),
        ("left_knee", ("Left_hip", "Left_knee", "Left_ankle")),
        ("right_knee", ("Right_hip", "Right_knee", "Right_ankle")),
        ("left_ankle", ("Left_knee", "Left_ankle", "Left_foot")),
        ("right_ankle", ("Right_knee", "Right_ankle", "Right_foot")),
    ),
    angle_region={
        "left_elbow": "left_arm", "left_shoulder": "left_arm",
        "right_elbow": "right_arm", "right_shoulder": "right_arm",
        "left_hip": "left_leg", "left_knee": "left_leg", "left_ankle": "left_leg",
        "right_hip": "right_leg", "right_knee": "right_leg", "right_ankle": "right_leg",
    },
    vertical_chain=("nose", "left_shoulder", "Left_hip", "Left_knee", "Left_ankle"),
    vertical_sign=+1,   # MediaPipe world landmarks: y increases downward
    hip_joints=("Left_hip", "Right_hip"),
    shoulder_joints=("left_shoulder", "right_shoulder"),
    units="metres (MediaPipe pose_world_landmarks, hip-centred)",
)


# ---------------------------------------------------------------------------
# Al-Jubouri - Kinect v2 body skeleton, 25 joints, camera space (y up, z depth)
# ---------------------------------------------------------------------------
#: Column order exactly as the dataset's own spreadsheets write it. "Midspain" is
#: the authors' spelling of Kinect's SpineMid joint; it is preserved verbatim so
#: column matching stays exact.
_KINECT_JOINTS: Final = (
    "Midspain", "AnkleLeft", "AnkleRight", "ElbowLeft", "ElbowRight",
    "FootLeft", "FootRight", "HandLeft", "HandRight", "HandTipLeft",
    "HandTipRight", "Head", "HipLeft", "HipRight", "KneeLeft", "KneeRight",
    "Neck", "ShoulderLeft", "ShoulderRight", "SpineBase", "SpineShoulder",
    "ThumbLeft", "ThumbRight", "WristLeft", "WristRight",
)

ALJUBOURI_KINECT = Skeleton(
    name="aljubouri_kinect_v2",
    joints=_KINECT_JOINTS,
    regions={
        "head": ("Head", "Neck"),
        "torso": ("SpineBase", "Midspain", "SpineShoulder", "Neck",
                  "ShoulderLeft", "ShoulderRight", "HipLeft", "HipRight"),
        "left_arm": ("ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
                     "HandTipLeft", "ThumbLeft"),
        "right_arm": ("ShoulderRight", "ElbowRight", "WristRight", "HandRight",
                      "HandTipRight", "ThumbRight"),
        "left_leg": ("HipLeft", "KneeLeft", "AnkleLeft", "FootLeft"),
        "right_leg": ("HipRight", "KneeRight", "AnkleRight", "FootRight"),
        "whole_body": _KINECT_JOINTS,
    },
    symmetric_pairs=(
        ("ShoulderLeft", "ShoulderRight"), ("ElbowLeft", "ElbowRight"),
        ("WristLeft", "WristRight"), ("HandLeft", "HandRight"),
        ("HandTipLeft", "HandTipRight"), ("ThumbLeft", "ThumbRight"),
        ("HipLeft", "HipRight"), ("KneeLeft", "KneeRight"),
        ("AnkleLeft", "AnkleRight"), ("FootLeft", "FootRight"),
    ),
    edges=(
        ("Head", "Neck"), ("Neck", "SpineShoulder"),
        ("SpineShoulder", "ShoulderLeft"), ("SpineShoulder", "ShoulderRight"),
        ("SpineShoulder", "Midspain"), ("Midspain", "SpineBase"),
        ("SpineBase", "HipLeft"), ("SpineBase", "HipRight"),
        ("ShoulderLeft", "ElbowLeft"), ("ElbowLeft", "WristLeft"),
        ("WristLeft", "HandLeft"), ("HandLeft", "HandTipLeft"),
        ("WristLeft", "ThumbLeft"),
        ("ShoulderRight", "ElbowRight"), ("ElbowRight", "WristRight"),
        ("WristRight", "HandRight"), ("HandRight", "HandTipRight"),
        ("WristRight", "ThumbRight"),
        ("HipLeft", "KneeLeft"), ("KneeLeft", "AnkleLeft"), ("AnkleLeft", "FootLeft"),
        ("HipRight", "KneeRight"), ("KneeRight", "AnkleRight"), ("AnkleRight", "FootRight"),
    ),
    rigid_bones=(
        ("ShoulderLeft", "ShoulderRight"), ("HipLeft", "HipRight"),
        ("ShoulderLeft", "ElbowLeft"), ("ShoulderRight", "ElbowRight"),
        ("ElbowLeft", "WristLeft"), ("ElbowRight", "WristRight"),
        ("HipLeft", "KneeLeft"), ("HipRight", "KneeRight"),
        ("KneeLeft", "AnkleLeft"), ("KneeRight", "AnkleRight"),
        ("SpineBase", "SpineShoulder"), ("Neck", "Head"),
    ),
    angle_triplets=(
        ("left_elbow", ("ShoulderLeft", "ElbowLeft", "WristLeft")),
        ("right_elbow", ("ShoulderRight", "ElbowRight", "WristRight")),
        ("left_shoulder", ("ElbowLeft", "ShoulderLeft", "HipLeft")),
        ("right_shoulder", ("ElbowRight", "ShoulderRight", "HipRight")),
        ("left_hip", ("ShoulderLeft", "HipLeft", "KneeLeft")),
        ("right_hip", ("ShoulderRight", "HipRight", "KneeRight")),
        ("left_knee", ("HipLeft", "KneeLeft", "AnkleLeft")),
        ("right_knee", ("HipRight", "KneeRight", "AnkleRight")),
        ("left_ankle", ("KneeLeft", "AnkleLeft", "FootLeft")),
        ("right_ankle", ("KneeRight", "AnkleRight", "FootRight")),
        ("trunk_flexion", ("SpineShoulder", "SpineBase", "KneeLeft")),
    ),
    angle_region={
        "left_elbow": "left_arm", "left_shoulder": "left_arm",
        "right_elbow": "right_arm", "right_shoulder": "right_arm",
        "left_hip": "left_leg", "left_knee": "left_leg", "left_ankle": "left_leg",
        "right_hip": "right_leg", "right_knee": "right_leg", "right_ankle": "right_leg",
        "trunk_flexion": "torso",
    },
    vertical_chain=("Head", "ShoulderLeft", "HipLeft", "KneeLeft", "AnkleLeft"),
    vertical_sign=-1,   # Kinect camera space: y increases upward
    hip_joints=("HipLeft", "HipRight"),
    shoulder_joints=("ShoulderLeft", "ShoulderRight"),
    units="metres (Kinect v2 camera space; z is depth from the sensor)",
)


SKELETONS: Final[dict[str, Skeleton]] = {
    MMASD_PLUS.name: MMASD_PLUS,
    ALJUBOURI_KINECT.name: ALJUBOURI_KINECT,
    "mmasd_plus": MMASD_PLUS,
    "aljubouri": ALJUBOURI_KINECT,
}

for _s in (MMASD_PLUS, ALJUBOURI_KINECT):
    _s.validate()


def get_skeleton(name: str) -> Skeleton:
    try:
        return SKELETONS[name]
    except KeyError as exc:
        raise KeyError(f"unknown skeleton {name!r}; known: {sorted(set(SKELETONS))}") from exc
