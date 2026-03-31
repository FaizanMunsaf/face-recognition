"""
MediaPipe Face Landmarker (Tasks API) + OpenCV solvePnP head pose (yaw, pitch, roll).

Model is downloaded once to data/models/face_landmarker.task
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import time

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# 3D reference points (generic face, same units as common OpenCV head-pose tutorials).
_FACE_3D = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)

# MediaPipe Face Mesh indices (compatible with Face Landmarker topology).
_LM_NOSE_TIP = 1
_LM_CHIN = 152
_LM_LEFT_EYE_OUTER = 33
_LM_RIGHT_EYE_OUTER = 263
_LM_MOUTH_LEFT = 61
_LM_MOUTH_RIGHT = 291
_LM_IDS = [
    _LM_NOSE_TIP,
    _LM_CHIN,
    _LM_LEFT_EYE_OUTER,
    _LM_RIGHT_EYE_OUTER,
    _LM_MOUTH_LEFT,
    _LM_MOUTH_RIGHT,
]


def _landmarks_xyxy(lm, w: int, h: int) -> np.ndarray:
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float64)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 0 else 0.0


def _pick_landmarks_for_bbox(
    candidates: list, w: int, h: int, prefer_xyxy: np.ndarray
) -> Optional[list]:
    """Choose MediaPipe face whose landmark hull best overlaps InsightFace bbox."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    pb = prefer_xyxy.astype(np.float64)
    best_i, best = 0, -1.0
    for i, lm in enumerate(candidates):
        bb = _landmarks_xyxy(lm, w, h)
        score = _iou_xyxy(pb, bb)
        if score <= 0:
            # centroid distance fallback
            cx, cy = (pb[0] + pb[2]) * 0.5, (pb[1] + pb[3]) * 0.5
            mx, my = (bb[0] + bb[2]) * 0.5, (bb[1] + bb[3]) * 0.5
            score = -float((cx - mx) ** 2 + (cy - my) ** 2) * 1e-6
        if score > best:
            best, best_i = score, i
    return candidates[best_i]


def default_model_path() -> Path:
    root = Path(__file__).resolve().parent / "data" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root / "face_landmarker.task"


def ensure_face_landmarker_model(path: Optional[Path] = None) -> Path:
    path = path or default_model_path()
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe face landmarker model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """Pitch, yaw, roll in degrees (approximate)."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        yaw = math.degrees(math.atan2(-R[2, 0], sy))
        roll = 0.0
    return pitch, yaw, roll


class MediaPipeHeadPose:
    """VIDEO running mode; call ``process_frame`` in order with monotonic timestamps."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python.core import base_options as bo
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        mp_path = ensure_face_landmarker_model(model_path)
        opts = FaceLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=str(mp_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = FaceLandmarker.create_from_options(opts)
        self._mp_Image = mp.Image
        self._ImageFormat = mp.ImageFormat

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        prefer_xyxy: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Returns (pitch, yaw, roll) in degrees, or None if no face / landmarks.

        When ``prefer_xyxy`` is set (InsightFace xyxy), picks the MediaPipe face that
        best overlaps that box so pose matches the face we embed.
        """
        if self._landmarker is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_Image(
            image_format=self._ImageFormat.SRGB, data=rgb
        )
        ts_ms = int(time.monotonic() * 1000)
        res = self._landmarker.detect_for_video(mp_image, ts_ms)
        if not res.face_landmarks:
            return None
        faces: List = list(res.face_landmarks)
        if prefer_xyxy is not None and len(faces) > 0:
            picked = _pick_landmarks_for_bbox(faces, w, h, prefer_xyxy)
            lm = picked if picked is not None else faces[0]
        else:
            lm = faces[0]
        max_idx = max(_LM_IDS)
        if len(lm) <= max_idx:
            return None

        pts_2d = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in _LM_IDS],
            dtype=np.float64,
        )
        focal = float(max(w, h))
        cam = np.array(
            [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]],
            dtype=np.float64,
        )
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            _FACE_3D, pts_2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        rot, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = _rotation_matrix_to_euler_angles(rot)
        return pitch, yaw, roll
