"""
Head-pose gates per enrollment step (yaw / pitch in degrees).

Ranges must match your solvePnP convention; tune if left/right feel swapped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HeadPoseGate:
    yaw_min: Optional[float] = None
    yaw_max: Optional[float] = None
    pitch_min: Optional[float] = None
    pitch_max: Optional[float] = None

    def satisfied(self, yaw: float, pitch: float) -> bool:
        if self.yaw_min is not None and yaw < self.yaw_min:
            return False
        if self.yaw_max is not None and yaw > self.yaw_max:
            return False
        if self.pitch_min is not None and pitch < self.pitch_min:
            return False
        if self.pitch_max is not None and pitch > self.pitch_max:
            return False
        return True


# Order matches DEFAULT_POSE_PROMPTS in main.py:
# straight, left, right, up, down
POSE_GATES: List[HeadPoseGate] = [
    HeadPoseGate(yaw_min=-20, yaw_max=20, pitch_min=-22, pitch_max=22),
    HeadPoseGate(yaw_min=14, yaw_max=72, pitch_min=-35, pitch_max=35),
    HeadPoseGate(yaw_min=-72, yaw_max=-14, pitch_min=-35, pitch_max=35),
    HeadPoseGate(yaw_min=-32, yaw_max=32, pitch_min=-65, pitch_max=-6),
    HeadPoseGate(yaw_min=-32, yaw_max=32, pitch_min=6, pitch_max=65),
]


def gate_for_step(step: int) -> HeadPoseGate:
    if step < 0 or step >= len(POSE_GATES):
        return POSE_GATES[0]
    return POSE_GATES[step]
