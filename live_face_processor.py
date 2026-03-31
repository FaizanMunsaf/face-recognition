"""WebRTC video: enrollment preview + wide multi-panel strip (fits HTML video player).

When several faces appear, the **largest** box (closest / dominant in frame) is used for crop, alignment, and capture.
"""

from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from enroll_pose_gates import gate_for_step
from face_model import FaceHit, FacePipeline, draw_arcface_recognition_landmarks
from face_quality import assess_face_quality

_pipeline: Optional[FacePipeline] = None
_pipeline_lock = threading.Lock()

# Match ``main.AUTO_POSE_QUALITY_KW`` (avoid importing main from this module).
AUTO_POSE_QUALITY_KW = dict(
    det_below=0.40,
    blur_below=24.0,
    dark_below=38.0,
    bright_above=248.0,
)
# 0 = queue auto-capture on first frame where pose + quality pass (no consecutive-frame wait).
LIVE_AUTO_STABLE_FRAMES = 0

_auto_cfg_lock = threading.Lock()
_live_auto_enabled: bool = False
_live_auto_step: int = 0

_auto_pending_lock = threading.Lock()
_auto_pending_bgr: Optional[np.ndarray] = None

_auto_metrics_lock = threading.Lock()
_live_auto_yaw: Optional[float] = None
_live_auto_pitch: Optional[float] = None
_live_auto_pose_ok: bool = False
_live_auto_stable: int = 0
_live_auto_quality_bad: bool = False

_mp_lock = threading.Lock()
_mp_pose: object = None  # None | False (failed) | MediaPipeHeadPose


def configure_live_auto_pose(enabled: bool, step_index: int) -> None:
    """Called from Streamlit each run; worker thread reads before processing each frame."""
    global _live_auto_enabled, _live_auto_step
    with _auto_cfg_lock:
        _live_auto_enabled = bool(enabled)
        _live_auto_step = int(step_index)


def _get_mediapipe_head_pose():
    global _mp_pose
    with _mp_lock:
        if _mp_pose is False:
            return None
        if _mp_pose is None:
            try:
                from head_pose_mediapipe import MediaPipeHeadPose

                _mp_pose = MediaPipeHeadPose()
            except Exception:
                _mp_pose = False
        return _mp_pose if _mp_pose is not False else None


def consume_pending_auto_capture() -> Optional[np.ndarray]:
    """Thread-safe: grab one frame queued by the worker for Streamlit to embed, or None."""
    with _auto_pending_lock:
        global _auto_pending_bgr
        b = _auto_pending_bgr
        _auto_pending_bgr = None
        return b.copy() if b is not None else None


def get_live_auto_status() -> dict:
    """Lightweight status for the Streamlit auto-pose poll fragment."""
    with _auto_cfg_lock:
        en = _live_auto_enabled
        stp = _live_auto_step
    with _auto_metrics_lock:
        return {
            "enabled": en,
            "step": stp,
            "yaw": _live_auto_yaw,
            "pitch": _live_auto_pitch,
            "pose_ok": _live_auto_pose_ok,
            "stable": _live_auto_stable,
            "quality_bad": _live_auto_quality_bad,
        }

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
        self._auto_stable: int = 0
        self._auto_seen_step: int = -999

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global _live_raw_bgr, _live_face_count, _live_subject_ok, _auto_pending_bgr
        global _live_auto_yaw, _live_auto_pitch, _live_auto_pose_ok, _live_auto_stable
        global _live_auto_quality_bad
        img = frame.to_ndarray(format="bgr24")
        raw = img.copy()
        self.latest_bgr = raw
        pl = get_shared_face_pipeline()
        hits = pl.all_faces(img)
        self.last_face_count = len(hits)

        with _auto_cfg_lock:
            auto_on = _live_auto_enabled
            auto_step = _live_auto_step

        if auto_on and auto_step != self._auto_seen_step:
            self._auto_stable = 0
            self._auto_seen_step = auto_step

        yaw_d: Optional[float] = None
        pitch_d: Optional[float] = None
        pose_ok = False
        quality_bad = False

        if auto_on:
            if not hits:
                self._auto_stable = 0
                with _auto_metrics_lock:
                    _live_auto_yaw = None
                    _live_auto_pitch = None
                    _live_auto_pose_ok = False
                    _live_auto_quality_bad = False
                    _live_auto_stable = 0
            else:
                h0 = FacePipeline._sort_by_area(hits)[0]
                rep = assess_face_quality(
                    img, h0.bbox, h0.det_score, **AUTO_POSE_QUALITY_KW
                )
                quality_bad = len(rep.messages) > 0
                mp = _get_mediapipe_head_pose()
                ypr = (
                    mp.process_frame(img, prefer_xyxy=h0.bbox)
                    if mp is not None
                    else None
                )
                gate = gate_for_step(auto_step)
                if ypr is None:
                    pose_ok = False
                else:
                    pitch, yaw, _roll = ypr
                    yaw_d, pitch_d = yaw, pitch
                    pose_ok = gate.satisfied(yaw, pitch)

                if quality_bad or not pose_ok:
                    self._auto_stable = 0
                else:
                    self._auto_stable += 1

                # 0 = require one good frame only; N>0 = N consecutive good frames.
                _need = 1 if LIVE_AUTO_STABLE_FRAMES <= 0 else LIVE_AUTO_STABLE_FRAMES
                if self._auto_stable >= _need:
                    with _auto_pending_lock:
                        if _auto_pending_bgr is None:
                            _auto_pending_bgr = raw.copy()
                    self._auto_stable = 0

                with _auto_metrics_lock:
                    _live_auto_yaw = yaw_d
                    _live_auto_pitch = pitch_d
                    _live_auto_pose_ok = pose_ok
                    _live_auto_quality_bad = quality_bad
                    _live_auto_stable = self._auto_stable
        else:
            self._auto_stable = 0
            with _auto_metrics_lock:
                _live_auto_yaw = None
                _live_auto_pitch = None
                _live_auto_pose_ok = False
                _live_auto_quality_bad = False
                _live_auto_stable = 0

        composed, _, self.last_subject_ok = compose_enrollment_multiview(pl, img, hits)

        if auto_on:
            if not hits:
                line = f"AUTO: show face | step {auto_step + 1}"
            else:
                ys = f"{yaw_d:+.0f}" if yaw_d is not None else "—"
                ps = f"{pitch_d:+.0f}" if pitch_d is not None else "—"
                q = "Q!" if quality_bad else "ok"
                if LIVE_AUTO_STABLE_FRAMES <= 0:
                    stab_txt = "instant"
                else:
                    stab_txt = f"{self._auto_stable}/{LIVE_AUTO_STABLE_FRAMES}"
                line = (
                    f"AUTO y={ys} p={ps} pose={'OK' if pose_ok else '--'} {q} "
                    f"stable={stab_txt}"
                )
            col = (80, 255, 180) if pose_ok and not quality_bad and hits else (180, 180, 255)
            cv2.putText(
                composed,
                line[:95],
                (2, composed.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                col,
                1,
                cv2.LINE_AA,
            )

        with _live_lock:
            _live_raw_bgr = raw.copy()
            _live_face_count = self.last_face_count
            _live_subject_ok = self.last_subject_ok
        return av.VideoFrame.from_ndarray(composed, format="bgr24")


# =====================================================================
# 7-Step Face Capture (head pose + eye close + mouth open)
# =====================================================================

SEVEN_STEPS = [
    {"id": "front",       "instruction": "Look STRAIGHT at the camera", "detect": "front"},
    {"id": "right",       "instruction": "Turn head to YOUR RIGHT -->", "detect": "right"},
    {"id": "left",        "instruction": "Turn head to YOUR LEFT <--",  "detect": "left"},
    {"id": "up",          "instruction": "Tilt head UP",                "detect": "up"},
    {"id": "down",        "instruction": "Tilt head DOWN",              "detect": "down"},
    # {"id": "eyes_closed", "instruction": "CLOSE both eyes",             "detect": "eyes_closed"},  # not approved by client yet
    {"id": "mouth_open",  "instruction": "OPEN mouth wide",             "detect": "mouth_open"},
]

_SS_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)

_SS_EYE_CLOSE_THRESH = 6
_SS_MOUTH_OPEN_THRESH = 18
_SS_HOLD_DURATION = 1.5

_SS_YAW_STRAIGHT_MAX = 10
_SS_PITCH_STRAIGHT_MAX = 12
_SS_YAW_TURN_MIN = 18
_SS_PITCH_UP_MIN = 15
_SS_PITCH_DOWN_MIN = 12

_C_GREEN = (0, 210, 90)
_C_BLUE = (220, 160, 0)
_C_WHITE = (255, 255, 255)
_C_DARK = (18, 18, 18)
_C_GRAY = (150, 150, 150)
_C_RED = (50, 50, 230)
_C_CYAN = (200, 200, 0)

# Shared state between processor thread and Streamlit main thread.
_ss_lock = threading.Lock()
_ss_target_step: int = 0
_ss_generation: int = 0
_ss_pending_bgr: Optional[np.ndarray] = None


def configure_seven_step(step: int) -> None:
    """Called by Streamlit to tell the processor which step to detect."""
    global _ss_target_step
    with _ss_lock:
        _ss_target_step = step


def reset_seven_step() -> None:
    """Reset shared 7-step state (called when user clicks Start / Reset)."""
    global _ss_target_step, _ss_pending_bgr, _ss_generation
    with _ss_lock:
        _ss_target_step = 0
        _ss_pending_bgr = None
        _ss_generation += 1


def consume_seven_step_capture() -> Optional[np.ndarray]:
    """Thread-safe: grab one captured frame queued by the processor, or None."""
    global _ss_pending_bgr
    with _ss_lock:
        if _ss_pending_bgr is None:
            return None
        bgr = _ss_pending_bgr.copy()
        _ss_pending_bgr = None
        return bgr


def _ss_px_dist(p1, p2, w: int, h: int) -> float:
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)


def _ss_get_head_pose(lm, w: int, h: int):
    img_pts = np.array([
        (lm[1].x * w, lm[1].y * h),
        (lm[152].x * w, lm[152].y * h),
        (lm[33].x * w, lm[33].y * h),
        (lm[263].x * w, lm[263].y * h),
        (lm[61].x * w, lm[61].y * h),
        (lm[291].x * w, lm[291].y * h),
    ], dtype=np.float64)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _SS_MODEL_3D, img_pts, cam, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    if yaw > 90:
        yaw -= 180
    elif yaw < -90:
        yaw += 180
    return pitch, yaw


def _ss_check_step(detect: str, lm, w: int, h: int) -> bool:
    if detect == "eyes_closed":
        left = _ss_px_dist(lm[159], lm[145], w, h)
        right = _ss_px_dist(lm[386], lm[374], w, h)
        return (left + right) / 2 < _SS_EYE_CLOSE_THRESH
    if detect == "mouth_open":
        return _ss_px_dist(lm[13], lm[14], w, h) > _SS_MOUTH_OPEN_THRESH
    pitch, yaw = _ss_get_head_pose(lm, w, h)
    if pitch is None:
        return False
    if detect == "front":
        return abs(yaw) < _SS_YAW_STRAIGHT_MAX and abs(pitch) < _SS_PITCH_STRAIGHT_MAX
    if detect == "right":
        return yaw < -_SS_YAW_TURN_MIN
    if detect == "left":
        return yaw > _SS_YAW_TURN_MIN
    if detect == "up":
        return pitch < -_SS_PITCH_UP_MIN
    if detect == "down":
        return pitch > _SS_PITCH_DOWN_MIN
    return False


def _ss_draw_overlay(frame, step_idx, hold_start, pose_ok, pitch, yaw, captured=False):
    h, w = frame.shape[:2]
    step = SEVEN_STEPS[step_idx]

    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 85), _C_DARK, -1)
    cv2.addWeighted(bar, 0.7, frame, 0.3, 0, frame)
    cv2.putText(
        frame, f"Step {step_idx + 1} / {len(SEVEN_STEPS)}",
        (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _C_GRAY, 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, step["instruction"],
        (18, 64), cv2.FONT_HERSHEY_DUPLEX, 0.85, _C_WHITE, 2, cv2.LINE_AA,
    )

    if pitch is not None and yaw is not None:
        cv2.putText(
            frame, f"Yaw={yaw:+.1f}  Pitch={pitch:+.1f}",
            (18, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _C_CYAN, 1, cv2.LINE_AA,
        )

    if captured:
        badge, col = "CAPTURED!", _C_GREEN
    elif pose_ok:
        badge, col = "POSE OK", _C_GREEN
    else:
        badge, col = "Adjust pose...", _C_RED
    cv2.putText(frame, badge, (18, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Hold arc (right side)
    cx, cy, r = w - 65, h // 2, 38
    cv2.circle(frame, (cx, cy), r, (55, 55, 55), 4)
    if captured:
        frac = 1.0
    elif pose_ok and hold_start is not None:
        frac = min((time.time() - hold_start) / _SS_HOLD_DURATION, 1.0)
    else:
        frac = 0.0

    if frac > 0:
        pts = []
        end_deg = int(-90 + frac * 360)
        for deg in range(-90, end_deg + 1, 3):
            rad = math.radians(deg)
            pts.append((int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))))
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, _C_GREEN, 4)
        cv2.putText(
            frame, f"{int(frac * 100)}%", (cx - 17, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, _C_GREEN, 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, "HOLD", (cx - 20, cy + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _C_GRAY, 1, cv2.LINE_AA,
        )

    # Progress dots
    dot_y = h - 22
    total = len(SEVEN_STEPS)
    sp = 30
    start_x = w // 2 - (total - 1) * sp // 2
    for i in range(total):
        cx2 = start_x + i * sp
        if i < step_idx:
            cv2.circle(frame, (cx2, dot_y), 8, _C_GREEN, -1)
        elif i == step_idx:
            cv2.circle(frame, (cx2, dot_y), 10, _C_BLUE, 2)
            cv2.circle(frame, (cx2, dot_y), 5, _C_BLUE, -1)
        else:
            cv2.circle(frame, (cx2, dot_y), 7, _C_GRAY, 1)


def _ss_draw_done(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(
        frame, "All 7 poses captured!",
        (w // 2 - 195, h // 2 - 25),
        cv2.FONT_HERSHEY_DUPLEX, 1.05, _C_GREEN, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Check Streamlit for results",
        (w // 2 - 190, h // 2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, _C_WHITE, 1, cv2.LINE_AA,
    )


class LiveSevenStepProcessor(VideoProcessorBase):
    """7-step face capture: 5 head poses + eye close + mouth open, with visual overlays."""

    def __init__(self) -> None:
        super().__init__()
        self._landmarker = None
        self._mp_Image = None
        self._ImageFormat = None
        self._t0 = time.monotonic()
        self._hold_start: Optional[float] = None
        self._current_step: int = -1
        self._capture_queued: bool = False
        self._gen: int = -1

    def _ensure_landmarker(self) -> bool:
        if self._landmarker is not None:
            return True
        try:
            import mediapipe as mp
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
            from head_pose_mediapipe import ensure_face_landmarker_model

            model_path = ensure_face_landmarker_model()
            opts = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
            )
            self._landmarker = FaceLandmarker.create_from_options(opts)
            self._mp_Image = mp.Image
            self._ImageFormat = mp.ImageFormat
            return True
        except Exception:
            return False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global _ss_pending_bgr

        with _ss_lock:
            target = _ss_target_step
            gen = _ss_generation

        if gen != self._gen:
            self._gen = gen
            self._current_step = -1
            self._hold_start = None
            self._capture_queued = False

        if target != self._current_step:
            self._current_step = target
            self._hold_start = None
            self._capture_queued = False

        img = frame.to_ndarray(format="bgr24")
        raw = img.copy()
        fh, fw = img.shape[:2]

        if target >= len(SEVEN_STEPS):
            _ss_draw_done(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if not self._ensure_landmarker():
            cv2.putText(
                img, "MediaPipe init failed", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = self._mp_Image(image_format=self._ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(mp_img, ts_ms)

        pose_ok = False
        pitch, yaw = None, None

        if result.face_landmarks and not self._capture_queued:
            lm = result.face_landmarks[0]
            for p in lm:
                cv2.circle(img, (int(p.x * fw), int(p.y * fh)), 1, (0, 190, 70), -1)

            pitch, yaw = _ss_get_head_pose(lm, fw, fh)
            pose_ok = _ss_check_step(SEVEN_STEPS[target]["detect"], lm, fw, fh)

            if pose_ok:
                if self._hold_start is None:
                    self._hold_start = time.time()
                elif time.time() - self._hold_start >= _SS_HOLD_DURATION:
                    with _ss_lock:
                        if _ss_pending_bgr is None:
                            _ss_pending_bgr = raw.copy()
                            self._capture_queued = True
            else:
                self._hold_start = None

        elif not result.face_landmarks and not self._capture_queued:
            self._hold_start = None
            cv2.putText(
                img, "No face detected - move closer",
                (18, fh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, _C_RED, 2, cv2.LINE_AA,
            )

        _ss_draw_overlay(
            img, target, self._hold_start,
            pose_ok or self._capture_queued, pitch, yaw,
            captured=self._capture_queued,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")
