"""Heuristic checks on face ROI: brightness, blur, detector score (OpenCV + NumPy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class FaceQualityReport:
    ok_for_capture: bool
    messages: List[str]
    mean_gray: float
    laplacian_var: float


def assess_face_quality(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    det_score: float,
    *,
    dark_below: float = 48.0,
    bright_above: float = 238.0,
    blur_below: float = 40.0,
    det_below: float = 0.52,
) -> FaceQualityReport:
    """
    ``bbox`` is xyxy in frame coordinates. Used for enrollment guidance only.
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    msgs: List[str] = []
    if x2 <= x1 or y2 <= y1:
        return FaceQualityReport(
            False, ["Face box invalid — center your face."], 0.0, 0.0
        )

    roi = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mean_gray = float(np.mean(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if mean_gray < dark_below:
        msgs.append(
            "Too dark — turn toward a lamp or window (light in front of you, not behind)."
        )
    if mean_gray > bright_above:
        msgs.append("Too bright / glare — soften light or step back.")
    if lap_var < blur_below:
        msgs.append("Blurry — hold still, move a bit closer, or clean the camera.")
    if det_score < det_below:
        msgs.append("Face not clear to detector — improve light and face the camera.")

    return FaceQualityReport(
        ok_for_capture=len(msgs) == 0,
        messages=msgs,
        mean_gray=mean_gray,
        laplacian_var=lap_var,
    )


def draw_quality_banner(bgr: np.ndarray, messages: List[str], line_h: int = 26) -> None:
    """Draw semi-transparent banner at bottom with tips (mutates image)."""
    if not messages:
        return
    h, w = bgr.shape[:2]
    n = min(len(messages), 3)
    bar_h = 8 + n * line_h
    y0 = h - bar_h
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (20, 20, 80), -1)
    cv2.addWeighted(overlay, 0.65, bgr, 0.35, 0, bgr)
    for i, msg in enumerate(messages[:n]):
        y = y0 + 18 + i * line_h
        cv2.putText(
            bgr,
            msg[:90],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 255),
            1,
            cv2.LINE_AA,
        )
