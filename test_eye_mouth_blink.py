import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/faizan/Desktop/take-talker/data/models/face_landmarker.task"

# Eye / mouth thresholds (pixels at 640x480)
EYE_CLOSE_THRESHOLD  = 6
MOUTH_OPEN_THRESHOLD = 18

# Show live Yaw/Pitch on screen so you can verify directions
DEBUG_ANGLES = True

# Head pose thresholds (degrees)
YAW_STRAIGHT_MAX   = 10
PITCH_STRAIGHT_MAX = 12
YAW_RIGHT_MIN      = 18   # person turns RIGHT  -> yaw goes NEGATIVE
YAW_LEFT_MIN       = 18   # person turns LEFT   -> yaw goes POSITIVE
PITCH_UP_MIN       = 15   # person looks UP     -> pitch goes NEGATIVE
PITCH_DOWN_MIN     = 12   # person looks DOWN   -> pitch goes POSITIVE

HOLD_DURATION = 1.5       # seconds to hold pose before capture
SAVE_DIR      = "captured_faces"

# =========================
# Steps
# =========================
STEPS = [
    {"id": "front",       "instruction": "Look STRAIGHT at the camera", "detect": "front"},
    {"id": "right",       "instruction": "Turn head to YOUR RIGHT -->",  "detect": "right"},
    {"id": "left",        "instruction": "Turn head to YOUR LEFT <--",   "detect": "left"},
    {"id": "up",          "instruction": "Tilt head UP",                 "detect": "up"},
    {"id": "down",        "instruction": "Tilt head DOWN",               "detect": "down"},
    # {"id": "eyes_closed", "instruction": "CLOSE both eyes",              "detect": "eyes_closed"},  # not approved by client yet
    {"id": "mouth_open",  "instruction": "OPEN mouth wide",              "detect": "mouth_open"},
]

# =========================
# 3-D model for solvePnP
# =========================
MODEL_POINTS = np.array([
    ( 0.0,   0.0,   0.0),
    ( 0.0, -63.6, -12.5),
    (-43.3,  32.7, -26.0),
    ( 43.3,  32.7, -26.0),
    (-28.9, -28.9, -24.1),
    ( 28.9, -28.9, -24.1),
], dtype=np.float64)

# =========================
# Helpers
# =========================
def px_dist(p1, p2, w, h):
    return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)


def get_head_pose(lm, w, h):
    img_pts = np.array([
        (lm[1].x * w,   lm[1].y * h),
        (lm[152].x * w, lm[152].y * h),
        (lm[33].x * w,  lm[33].y * h),
        (lm[263].x * w, lm[263].y * h),
        (lm[61].x * w,  lm[61].y * h),
        (lm[291].x * w, lm[291].y * h),
    ], dtype=np.float64)

    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, img_pts, cam,
                                np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None

    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)

    if sy > 1e-6:
        pitch = math.degrees(math.atan2( rmat[2, 1], rmat[2, 2]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))
    else:
        pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = math.degrees(math.atan2(-rmat[2, 0], sy))

    if pitch >  90: pitch -= 180
    elif pitch < -90: pitch += 180
    if yaw >  90: yaw -= 180
    elif yaw < -90: yaw += 180

    return pitch, yaw


def check_step(detect, lm, w, h):
    # ── Eyes closed ───────────────────────────────────────────────────
    if detect == "eyes_closed":
        left_eye  = px_dist(lm[159], lm[145], w, h)
        right_eye = px_dist(lm[386], lm[374], w, h)
        avg = (left_eye + right_eye) / 2
        return avg < EYE_CLOSE_THRESHOLD

    # ── Mouth open ────────────────────────────────────────────────────
    if detect == "mouth_open":
        return px_dist(lm[13], lm[14], w, h) > MOUTH_OPEN_THRESHOLD

    # ── Head pose steps ───────────────────────────────────────────────
    pitch, yaw = get_head_pose(lm, w, h)
    if pitch is None:
        return False

    if detect == "front":
        return abs(yaw) < YAW_STRAIGHT_MAX and abs(pitch) < PITCH_STRAIGHT_MAX

    # Standard front-facing webcam (mirror image):
    #   Person turns RIGHT  -->  yaw becomes NEGATIVE
    #   Person turns LEFT   -->  yaw becomes POSITIVE
    #   Person looks UP     -->  pitch becomes NEGATIVE
    #   Person looks DOWN   -->  pitch becomes POSITIVE
    if detect == "right":
        return yaw < -YAW_RIGHT_MIN

    if detect == "left":
        return yaw >  YAW_LEFT_MIN

    if detect == "up":
        return pitch < -PITCH_UP_MIN

    if detect == "down":
        return pitch >  PITCH_DOWN_MIN

    return False


# =========================
# Colors (BGR)
# =========================
C_GREEN = (0, 210, 90)
C_BLUE  = (220, 160, 0)
C_WHITE = (255, 255, 255)
C_DARK  = (18, 18, 18)
C_GRAY  = (150, 150, 150)
C_RED   = (50, 50, 230)
C_CYAN  = (200, 200, 0)


# =========================
# UI drawing
# =========================
def draw_overlay(frame, step_idx, hold_start, pose_ok, pitch, yaw):
    h, w = frame.shape[:2]
    step = STEPS[step_idx]

    # Top bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, 85), C_DARK, -1)
    cv2.addWeighted(bar, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"Step {step_idx+1} / {len(STEPS)}",
                (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, step["instruction"],
                (18, 64), cv2.FONT_HERSHEY_DUPLEX, 0.85, C_WHITE, 2, cv2.LINE_AA)

    # Live angle debug
    if DEBUG_ANGLES and pitch is not None and yaw is not None:
        cv2.putText(frame, f"Yaw={yaw:+.1f}  Pitch={pitch:+.1f}",
                    (18, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 1, cv2.LINE_AA)

    # Pose badge
    badge = "POSE OK  v" if pose_ok else "Adjust pose..."
    col   = C_GREEN if pose_ok else C_RED
    cv2.putText(frame, badge, (18, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Hold arc (right side)
    cx, cy, r = w - 65, h // 2, 38
    cv2.circle(frame, (cx, cy), r, (55, 55, 55), 4)
    if pose_ok and hold_start is not None:
        frac = min((time.time() - hold_start) / HOLD_DURATION, 1.0)
        pts  = []
        end_deg = int(-90 + frac * 360)
        for deg in range(-90, end_deg + 1, 3):
            rad = math.radians(deg)
            pts.append((int(cx + r * math.cos(rad)),
                        int(cy + r * math.sin(rad))))
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts)], False, C_GREEN, 4)
        cv2.putText(frame, f"{int(frac*100)}%",
                    (cx - 17, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_GREEN, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "HOLD",
                    (cx - 20, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_GRAY, 1, cv2.LINE_AA)

    # Progress dots
    dot_y   = h - 22
    total   = len(STEPS)
    sp      = 30
    start_x = w // 2 - (total - 1) * sp // 2
    for i in range(total):
        cx2 = start_x + i * sp
        if i < step_idx:
            cv2.circle(frame, (cx2, dot_y), 8, C_GREEN, -1)
        elif i == step_idx:
            cv2.circle(frame, (cx2, dot_y), 10, C_BLUE, 2)
            cv2.circle(frame, (cx2, dot_y),  5, C_BLUE, -1)
        else:
            cv2.circle(frame, (cx2, dot_y),  7, C_GRAY, 1)


def draw_done(frame):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "All 7 poses captured!",
                (w//2 - 195, h//2 - 25),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, C_GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Saved to: {SAVE_DIR}/",
                (w//2 - 165, h//2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Press ESC to exit",
                (w//2 - 120, h//2 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_GRAY, 1, cv2.LINE_AA)


# =========================
# Open camera — tries indices 0,1,2 with V4L2 then ANY
# =========================
def open_camera():
    for idx in [0, 1, 2, 4]:
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[OK] Camera opened: index={idx}  backend={'V4L2' if backend == cv2.CAP_V4L2 else 'ANY'}")
                return cap
            cap.release()
    return None


cap = open_camera()
if cap is None:
    print("[ERROR] Could not open any camera.")
    exit(1)

# =========================
# MediaPipe
# =========================
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(options)

# =========================
# Session output folder
# =========================
session_dir = os.path.join(SAVE_DIR, f"session_{int(time.time())}")
os.makedirs(session_dir, exist_ok=True)
print(f"[i] Saving to: {session_dir}")

# =========================
# State
# =========================
step_idx   = 0
hold_start = None
captured   = []
done       = False
t0         = time.time()

# =========================
# Main loop
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.05)
        continue

    h, w = frame.shape[:2]

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms  = int((time.time() - t0) * 1000)
    result = landmarker.detect_for_video(mp_img, ts_ms)

    # Completion screen
    if done:
        draw_done(frame)
        cv2.imshow("Face Capture - 7 Steps", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    pose_ok   = False
    dbg_pitch = None
    dbg_yaw   = None

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # Draw landmark dots
        for p in lm:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 190, 70), -1)

        # Get angles for debug display
        dbg_pitch, dbg_yaw = get_head_pose(lm, w, h)

        step    = STEPS[step_idx]
        pose_ok = check_step(step["detect"], lm, w, h)

        if pose_ok:
            if hold_start is None:
                hold_start = time.time()
            elif time.time() - hold_start >= HOLD_DURATION:
                fname = os.path.join(session_dir,
                                     f"{step_idx+1:02d}_{step['id']}.jpg")
                cv2.imwrite(fname, frame)
                print(f"[OK] Step {step_idx+1}/7  [{step['id']}]  -> {fname}")
                captured.append(step["id"])
                step_idx  += 1
                hold_start = None
                if step_idx >= len(STEPS):
                    done = True
                    print(f"\n[DONE] All 7 poses saved -> {session_dir}")
        else:
            hold_start = None

    else:
        hold_start = None
        cv2.putText(frame, "No face detected - move closer",
                    (18, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, C_RED, 2, cv2.LINE_AA)

    draw_overlay(frame, min(step_idx, len(STEPS) - 1),
                 hold_start, pose_ok, dbg_pitch, dbg_yaw)

    cv2.imshow("Face Capture - 7 Steps", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
