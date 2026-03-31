# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import os
# import time

# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
# from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

# # =========================
# # CONFIG
# # =========================
# MODEL_PATH = "/home/faizan/Desktop/take-talker/data/models/face_landmarker.task"

# EYE_CLOSE_THRESHOLD = 6
# MOUTH_OPEN_THRESHOLD = 20
# BLINK_FRAMES = 2
# MOUTH_FRAMES = 2

# # Tolerances for head pose (degrees)
# YAW_LIMIT = 8
# PITCH_LIMIT = 10
# CENTER_THRESHOLD_RATIO = 0.05  # 5% of frame width for centering

# # Save directory
# SAVE_DIR = "captured_faces"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # =========================
# # Initialize MediaPipe FaceLandmarker
# # =========================
# options = FaceLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=vision.RunningMode.VIDEO,
#     num_faces=1
# )
# landmarker = FaceLandmarker.create_from_options(options)

# # =========================
# # Camera setup
# # =========================
# cap = cv2.VideoCapture(0)
# frame_timestamp = 0

# # =========================
# # State variables
# # =========================
# blink_counter = 0
# eye_closed_flag = False
# eye_blinked_once = False

# mouth_counter = 0
# mouth_open_flag = False
# mouth_opened_once = False

# image_saved = False

# # 3D model points for solvePnP
# model_points = np.array([
#     (0.0, 0.0, 0.0),          # Nose tip
#     (0.0, -63.6, -12.5),      # Chin
#     (-43.3, 32.7, -26.0),     # Left eye left corner
#     (43.3, 32.7, -26.0),      # Right eye right corner
#     (-28.9, -28.9, -24.1),    # Left Mouth corner
#     (28.9, -28.9, -24.1)      # Right mouth corner
# ], dtype=np.float64)

# def get_distance(p1, p2, w, h):
#     x1, y1 = int(p1.x * w), int(p1.y * h)
#     x2, y2 = int(p2.x * w), int(p2.y * h)
#     return math.hypot(x2 - x1, y2 - y1)

# def rotation_vector_to_euler_angles(rot_vec):
#     rmat, _ = cv2.Rodrigues(rot_vec)

#     sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])

#     singular = sy < 1e-6

#     if not singular:
#         x = math.atan2(rmat[2, 1], rmat[2, 2])
#         y = math.atan2(-rmat[2, 0], sy)
#         z = math.atan2(rmat[1, 0], rmat[0, 0])
#     else:
#         x = math.atan2(-rmat[1, 2], rmat[1, 1])
#         y = math.atan2(-rmat[2, 0], sy)
#         z = 0

#     pitch = np.degrees(x)
#     yaw = np.degrees(y)
#     roll = np.degrees(z)

#     # Normalize pitch to [-90, 90]
#     if pitch > 90:
#         pitch -= 180
#     elif pitch < -90:
#         pitch += 180

#     # Normalize yaw to [-90, 90]
#     if yaw > 90:
#         yaw -= 180
#     elif yaw < -90:
#         yaw += 180

#     return pitch, yaw, roll

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

#     result = landmarker.detect_for_video(mp_image, frame_timestamp)
#     frame_timestamp += 1

#     liveness_verified = False
#     face_state = []

#     if result.face_landmarks:
#         lm = result.face_landmarks[0]

#         # Draw landmarks
#         for p in lm:
#             x, y = int(p.x * w), int(p.y * h)
#             cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

#         # Eye blink detection
#         eye_dist = get_distance(lm[159], lm[145], w, h)
#         if eye_dist < EYE_CLOSE_THRESHOLD:
#             blink_counter += 1
#             if blink_counter >= BLINK_FRAMES:
#                 eye_closed_flag = True
#         else:
#             if eye_closed_flag:
#                 eye_blinked_once = True
#             eye_closed_flag = False
#             blink_counter = 0
#         face_state.append("Eyes Closed" if eye_closed_flag else "Eyes Open")

#         # Mouth open detection
#         mouth_dist = get_distance(lm[13], lm[14], w, h)
#         if mouth_dist > MOUTH_OPEN_THRESHOLD:
#             mouth_counter += 1
#             if mouth_counter >= MOUTH_FRAMES:
#                 mouth_open_flag = True
#         else:
#             if mouth_open_flag:
#                 mouth_opened_once = True
#             mouth_open_flag = False
#             mouth_counter = 0
#         face_state.append("Mouth Open" if mouth_open_flag else "Mouth Closed")

#         # Head pose estimation
#         image_points = np.array([
#             (lm[1].x * w, lm[1].y * h),      # Nose tip
#             (lm[152].x * w, lm[152].y * h),  # Chin
#             (lm[33].x * w, lm[33].y * h),    # Left eye left corner
#             (lm[263].x * w, lm[263].y * h),  # Right eye right corner
#             (lm[61].x * w, lm[61].y * h),    # Left mouth corner
#             (lm[291].x * w, lm[291].y * h)   # Right mouth corner
#         ], dtype=np.float64)

#         focal_length = w
#         center = (w / 2, h / 2)
#         camera_matrix = np.array(
#             [[focal_length, 0, center[0]],
#              [0, focal_length, center[1]],
#              [0, 0, 1]], dtype=np.float64)
#         dist_coeffs = np.zeros((4, 1))

#         success, rotation_vector, _ = cv2.solvePnP(
#             model_points, image_points, camera_matrix, dist_coeffs
#         )

#         pitch, yaw, roll = rotation_vector_to_euler_angles(rotation_vector)

#         face_state.append(f"Yaw:{yaw:.1f} Pitch:{pitch:.1f}")

#         # Face centering using nose tip
#         nose_x = lm[1].x * w
#         offset = abs(nose_x - w / 2)
#         centered = offset < (w * CENTER_THRESHOLD_RATIO)
#         face_state.append("Centered" if centered else "Not Centered")

#         # Liveness check
#         head_straight = abs(yaw) < YAW_LIMIT and abs(pitch) < PITCH_LIMIT

#         if eye_blinked_once and mouth_opened_once and head_straight and centered:
#             liveness_verified = True

#     # Display face state text
#     y0 = 30
#     for s in face_state:
#         cv2.putText(frame, s, (30, y0),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         y0 += 30

#     if liveness_verified:
#         cv2.putText(frame, "✅ Liveness Verified", (30, y0 + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
#         if not image_saved:
#             filename = os.path.join(SAVE_DIR, f"face_{int(time.time())}.jpg")
#             cv2.imwrite(filename, frame)
#             print("Saved:", filename)
#             image_saved = True
#         # Reset for next detection
#         eye_blinked_once = False
#         mouth_opened_once = False
#         image_saved = False
#     else:
#         cv2.putText(frame, "❌ Liveness Not Verified", (30, y0 + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

#     cv2.imshow("Liveness Detection", frame)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
#         break

# cap.release()
# cv2.destroyAllWindows()
