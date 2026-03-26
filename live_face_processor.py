"""WebRTC video: enrollment preview + wide multi-panel strip (fits HTML video player).

When several faces appear, the **largest** box (closest / dominant in frame) is used for crop, alignment, and capture.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from face_model import FaceHit, FacePipeline, draw_arcface_recognition_landmarks

_pipeline: Optional[FacePipeline] = None
_pipeline_lock = threading.Lock()

# Survives Streamlit reruns: webrtc_streamer may stop the worker when `playing` flickers false,
# which clears ctx.video_processor before the Capture button handler runs. The worker thread
# still updates this buffer every frame so capture can read the last live frame.
_live_lock = threading.Lock()
_live_raw_bgr: Optional[np.ndarray] = None
_live_face_count: int = 0
_live_subject_ok: bool = False


def get_live_capture_metrics() -> tuple[int, bool, bool]:
    """(face_count, subject_ok, has_frame) — subject_ok means ≥1 face (largest is used). Cheap, no image copy."""
    with _live_lock:
        return _live_face_count, _live_subject_ok, _live_raw_bgr is not None


def take_live_capture_frame() -> Optional[np.ndarray]:
    """Copy of the latest raw camera frame from the WebRTC processor, or None."""
    with _live_lock:
        if _live_raw_bgr is None:
            return None
        return _live_raw_bgr.copy()


def get_shared_face_pipeline() -> FacePipeline:
    """Single shared model (safe for WebRTC worker thread)."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = FacePipeline()
        return _pipeline


def _padded_square_crop(frame_bgr: np.ndarray, bbox: np.ndarray, pad: float = 0.15) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(float)
    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(w, h) * (1.0 + 2.0 * pad)
    nx1 = int(cx - side * 0.5)
    ny1 = int(cy - side * 0.5)
    nx2 = int(cx + side * 0.5)
    ny2 = int(cy + side * 0.5)
    fh, fw = frame_bgr.shape[:2]
    nx1, ny1 = max(0, nx1), max(0, ny1)
    nx2, ny2 = min(fw, nx2), min(fh, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return frame_bgr[ny1:ny2, nx1:nx2]


def compose_enrollment_multiview(
    pl: FacePipeline,
    img: np.ndarray,
    hits: List[FaceHit],
    *,
    main_h: int = 160,
    gap: int = 6,
    label_h: int = 22,
) -> Tuple[np.ndarray, int, bool]:
    """
    One **wide, short** frame so browser video players show everything without vertical crop:

    [labels bar]
    [ main camera | crop RGB | gray | aligned | net ]

    Returns ``(composed_bgr, face_count, subject_ok)`` where subject_ok is True if ``n >= 1``.
    """
    n = len(hits)
    subject_ok = n >= 1

    main = img.copy()
    if n == 0:
        cv2.putText(
            main,
            "NO FACE",
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        largest = FacePipeline._sort_by_area(hits)[0]
        if n > 1:
            cv2.putText(
                main,
                f"{n} faces - CLOSEST (largest box)",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        x1, y1, x2, y2 = largest.bbox.astype(int)
        col = (0, 255, 0)
        cv2.rectangle(main, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        if largest.kps is not None:
            draw_arcface_recognition_landmarks(
                main,
                largest.kps,
                radius=3,
                font_scale=0.32,
                line_color=(200, 200, 200),
            )
        cv2.putText(
            main,
            "OK: 1 face" if n == 1 else "OK: closest face",
            (8, main.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            col,
            1,
            cv2.LINE_AA,
        )

    mh, mw = main.shape[:2]
    scale = main_h / max(mh, 1)
    main_w = max(int(mw * scale), 80)
    main_s = cv2.resize(main, (main_w, main_h), interpolation=cv2.INTER_AREA)

    panel = main_h

    def sq(x: np.ndarray) -> np.ndarray:
        return cv2.resize(x, (panel, panel), interpolation=cv2.INTER_AREA)

    if n == 0:
        dead = np.zeros((panel, panel, 3), dtype=np.uint8)
        p0 = p1 = p2 = p3 = dead
    else:
        largest = FacePipeline._sort_by_area(hits)[0]
        crop = _padded_square_crop(img, largest.bbox)
        gray = cv2.cvtColor(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR
        )
        aligned = pl.aligned_face_chip(img, largest)
        if aligned is None:
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        net = pl.network_preprocess_preview_bgr(aligned)
        if net is None:
            net = aligned.copy()
        p0, p1, p2, p3 = sq(crop), sq(gray), sq(aligned), sq(net)

    g = np.full((main_h, gap, 3), 32, dtype=np.uint8)
    row = np.hstack([main_s, g, p0, g, p1, g, p2, g, p3])
    total_w = row.shape[1]

    bar = np.zeros((label_h, total_w, 3), dtype=np.uint8)
    parts: List[Tuple[int, Optional[str]]] = [
        (main_s.shape[1], "Live"),
        (gap, None),
        (panel, "BBox RGB"),
        (gap, None),
        (panel, "Gray"),
        (gap, None),
        (panel, "Aligned"),
        (gap, None),
        (panel, "Net"),
    ]
    cursor = 0
    for w, lab in parts:
        if lab:
            cv2.putText(
                bar,
                lab[:14],
                (cursor + 3, label_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        cursor += w

    composed = np.vstack([bar, row])
    return composed, n, subject_ok


class LiveFaceBoxProcessor(VideoProcessorBase):
    """
    Wide strip: main camera + four processing views (fits typical video element).
    """

    def __init__(self) -> None:
        super().__init__()
        self.latest_bgr: Optional[np.ndarray] = None
        self.last_face_count: int = 0
        self.last_subject_ok: bool = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global _live_raw_bgr, _live_face_count, _live_subject_ok
        img = frame.to_ndarray(format="bgr24")
        raw = img.copy()
        self.latest_bgr = raw
        pl = get_shared_face_pipeline()
        hits = pl.all_faces(img)
        self.last_face_count = len(hits)
        composed, _, self.last_subject_ok = compose_enrollment_multiview(pl, img, hits)
        with _live_lock:
            _live_raw_bgr = raw.copy()
            _live_face_count = self.last_face_count
            _live_subject_ok = self.last_subject_ok
        return av.VideoFrame.from_ndarray(composed, format="bgr24")
