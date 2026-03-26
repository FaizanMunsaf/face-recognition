#!/usr/bin/env python3
"""
Guided enrollment (pose prompts + frame capture) and photo recognition
using ArcFace-style embeddings (InsightFace buffalo_l).

Examples:
  python main.py enroll
  python main.py enroll --name alice
  python main.py enroll --name alice --auto-pose
  python main.py enroll --name bob --video ./clip.mp4
  python main.py recognize --image ./group.jpg
  python main.py recognize --image ./group.jpg --out ./out.jpg --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from enroll_pose_gates import gate_for_step
from face_model import (
    FaceHit,
    FacePipeline,
    cosine_best_match,
    draw_arcface_recognition_landmarks,
    read_image_bgr,
)
from face_quality import assess_face_quality, draw_quality_banner
from gallery_store import (
    default_gallery_dir,
    load_gallery,
    merge_person,
    resolve_person_id,
    save_gallery,
)

# Consecutive frames with good pose + quality before auto-capture (~0.5s at 30 FPS).
STABLE_FRAMES_AUTO = 15

# One prompt per capture; extend or reorder as you like.
DEFAULT_POSE_PROMPTS: List[str] = [
    "Look straight at the camera",
    "Turn your face LEFT (show your right cheek)",
    "Turn your face RIGHT (show your left cheek)",
    "Tilt your face UP slightly",
    "Tilt your face DOWN slightly",
]


def _print_enrollment_capture(
    pose_idx: int,
    n_poses: int,
    hit: FaceHit,
    prev_embedding: Optional[np.ndarray],
    n_faces_in_frame: int,
) -> None:
    emb = hit.embedding
    dim = int(emb.size)
    norm = float(np.linalg.norm(emb))
    x1, y1, x2, y2 = hit.bbox.astype(int)
    print(
        f"  [pose {pose_idx}/{n_poses}] pipeline output stored as one {dim}-D embedding:"
    )
    print(f"    L2 norm={norm:.4f} (buffalo_l is L2-normalized; expect ~1.0)")
    print(f"    detector score={hit.det_score:.3f}  bbox=({x1},{y1})-({x2},{y2})")
    if n_faces_in_frame > 1:
        print(
            f"    note: {n_faces_in_frame} faces in frame — using the largest box for this pose."
        )
    if prev_embedding is not None:
        sim = float(np.dot(emb.ravel(), prev_embedding.ravel()))
        print(f"    cosine vs previous pose in this session: {sim:.3f} (same person usually >0.3–0.7)")
    preview = emb.ravel()[:8]
    print(f"    first 8 dims (preview): {np.array2string(preview, precision=4, floatmode='fixed')}")


def _print_pairwise_cosine_matrix(embeddings: List[np.ndarray]) -> None:
    if len(embeddings) < 2:
        return
    m = len(embeddings)
    print("\n  Pairwise cosine similarity between poses stored for this person:")
    header = "        " + "".join(f"  P{j + 1:>2}" for j in range(m))
    print(header)
    for i in range(m):
        row = f"    P{i + 1:>2}"
        for j in range(m):
            s = float(np.dot(embeddings[i].ravel(), embeddings[j].ravel()))
            row += f"  {s:5.2f}"
        print(row)


def _open_capture(video_path: Optional[str]) -> cv2.VideoCapture:
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video_path}")
        return cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam (device 0). Try --video path/to/file.mp4")
    return cap


def run_enroll(
    name: Optional[str],
    video_path: Optional[str],
    prompts: List[str],
    gallery_root: Path,
    auto_pose: bool = False,
) -> None:
    person_id = resolve_person_id(name)
    if not (name or "").strip():
        print(f"Using auto-generated person id (UUID): {person_id}")

    pipeline = FacePipeline()
    gallery = load_gallery(gallery_root)
    cap = _open_capture(video_path)
    captured: List[np.ndarray] = []
    step = 0

    if auto_pose and video_path:
        print(
            "Note: --auto-pose is intended for webcam; disabling auto-pose for file video "
            "(use SPACE to capture each frame)."
        )
        auto_pose = False

    mp_pose = None
    if auto_pose:
        from head_pose_mediapipe import MediaPipeHeadPose

        mp_pose = MediaPipeHeadPose()

    win = (
        "Enrollment — AUTO pose (hold steady) | Q quit"
        if auto_pose
        else "Enrollment — SPACE capture | . next frame (video) | Q quit"
    )
    pipe_win = "Enrollment — pipeline: raw crop | aligned chip | net colors"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(pipe_win, cv2.WINDOW_NORMAL)

    print(f"\nEnrolling: {person_id!r}")
    if auto_pose:
        print(
            f"Auto-pose: hold each instructed pose ~{STABLE_FRAMES_AUTO} stable frames; "
            "SPACE still captures manually. Poor light/blur blocks auto until fixed."
        )
    else:
        print("Controls: SPACE = capture this pose | Q = abort enrollment")
    print(
        "5 recognition landmarks per face: R-eye, L-eye, nose, L-mouth, R-mouth "
        "(detector points → ArcFace warp). Main view + strip panel 1–2 show dots, lines, labels."
    )
    print(
        "Second window = 3 panels: bbox crop (raw) -> aligned chip (same 5 pts, canonical grid) -> "
        "network preprocessing (RGB swap + mean/std)."
    )
    print(
        "Tips: orange banner = improve lighting or hold still (OpenCV checks). "
        "MediaPipe estimates head yaw/pitch for auto-capture when enabled."
    )
    if not auto_pose:
        print("Live green box = largest face used when you press SPACE; terminal shows embedding stats.")
    if video_path:
        print("Video: . = next frame , = previous frame")

    stable_pose_frames = 0

    def draw_overlay(frame, text: str) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame,
            text[:80],
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    try:
        while step < len(prompts):
            ok, frame = cap.read()
            if not ok or frame is None:
                if video_path:
                    print("End of video or read error; rewinding.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("Failed to read from camera.")
                break

            hits = pipeline.all_faces(frame)
            hits = FacePipeline._sort_by_area(hits)

            show = frame.copy()
            fh, fw = show.shape[:2]
            quality_msgs: List[str] = []
            ypr = None
            pose_ok = False
            gate = gate_for_step(step)

            if hits:
                h0 = hits[0]
                rep = assess_face_quality(frame, h0.bbox, h0.det_score)
                quality_msgs = rep.messages

                if auto_pose and mp_pose is not None:
                    ypr = mp_pose.process_frame(frame)
                    if ypr is None:
                        pose_ok = False
                    else:
                        pitch, yaw, roll = ypr
                        pose_ok = gate.satisfied(yaw, pitch)
                    col = (0, 255, 0) if pose_ok else (0, 140, 255)
                    ytxt = fh - 52
                    if ypr is not None:
                        pitch, yaw, roll = ypr
                        hud = (
                            f"MediaPipe yaw={yaw:+.0f} pitch={pitch:+.0f}  "
                            f"pose={'OK' if pose_ok else 'adjust'}  "
                            f"stable={stable_pose_frames}/{STABLE_FRAMES_AUTO}"
                        )
                    else:
                        hud = "MediaPipe: no head pose — show your face"
                    cv2.putText(
                        show,
                        hud[:85],
                        (10, ytxt),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        col,
                        2,
                        cv2.LINE_AA,
                    )

                x1, y1, x2, y2 = h0.bbox.astype(int)
                cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
                hint = (
                    "largest face — hold pose (auto)"
                    if auto_pose
                    else "largest face -> used on SPACE"
                )
                cv2.putText(
                    show,
                    hint,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if h0.kps is not None:
                    draw_arcface_recognition_landmarks(
                        show, h0.kps, radius=6, font_scale=0.52, line_color=(210, 210, 210)
                    )
                if len(hits) > 1:
                    cv2.putText(
                        show,
                        f"{len(hits)} faces — using CLOSEST (largest box)",
                        (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
                strip = pipeline.enrollment_pipeline_strip(frame, h0)
                cv2.imshow(pipe_win, strip)
            else:
                stable_pose_frames = 0
                blank = np.zeros((240, 620, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "No face — pipeline preview idle",
                    (24, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(pipe_win, blank)

            draw_quality_banner(show, quality_msgs)
            prompt = prompts[step]
            draw_overlay(show, f"{step + 1}/{len(prompts)}: {prompt}")
            cv2.imshow(win, show)
            key = cv2.waitKey(30) & 0xFF

            if hits and auto_pose and mp_pose is not None:
                if quality_msgs:
                    stable_pose_frames = 0
                elif pose_ok:
                    stable_pose_frames += 1
                else:
                    stable_pose_frames = 0
            elif auto_pose:
                stable_pose_frames = 0

            if key == ord("q"):
                print("Aborted.")
                return
            if video_path:
                if key == ord("."):
                    continue
                if key == ord(","):
                    cur = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, cur - 2))
                    continue

            auto_fire = (
                auto_pose
                and hits
                and stable_pose_frames >= STABLE_FRAMES_AUTO
                and not quality_msgs
                and pose_ok
            )

            if key == ord(" ") or auto_fire:
                if not hits:
                    print("  No face detected — adjust and press SPACE again.")
                    continue
                hit = hits[0]
                if len(hits) > 1:
                    print(
                        f"  {len(hits)} faces in frame — using largest (closest) face for enrollment."
                    )
                rep = assess_face_quality(frame, hit.bbox, hit.det_score)
                if auto_fire and rep.messages:
                    stable_pose_frames = 0
                    continue
                if key == ord(" ") and rep.messages:
                    print("  Warning (manual capture): " + " | ".join(rep.messages))
                emb = hit.embedding.copy()
                prev = captured[-1] if captured else None
                tag = "auto-pose" if auto_fire else "manual SPACE"
                print(f"  [{tag}]")
                _print_enrollment_capture(
                    step + 1,
                    len(prompts),
                    hit,
                    prev,
                    len(hits),
                )
                print(
                    "    window columns: (1) raw bbox crop  (2) landmark-aligned chip  "
                    "(3) same pixels after RGB swap + (x-127.5)/127.5 (shown re-scaled to 0-255)"
                )
                strip = pipeline.enrollment_pipeline_strip(frame, hit)
                cv2.imshow(pipe_win, strip)
                captured.append(emb)
                print(f"  -> stored pose {len(captured)}/{len(prompts)} in session buffer\n")
                step += 1
                stable_pose_frames = 0
    finally:
        if mp_pose is not None:
            mp_pose.close()
        cap.release()
        cv2.destroyWindow(win)
        cv2.destroyWindow(pipe_win)

    if len(captured) != len(prompts):
        print("Enrollment incomplete; gallery not updated.")
        return

    _print_pairwise_cosine_matrix(captured)
    gallery = merge_person(gallery, person_id, captured)
    save_gallery(gallery, gallery_root)
    print(f"\nSaved {len(captured)} embeddings for {person_id!r} under {gallery_root}")


def _annotate_recognition(
    img: np.ndarray,
    hits: List[FaceHit],
    gallery: dict,
    threshold: float,
) -> np.ndarray:
    """Draw bounding boxes and name / similarity labels (BGR image)."""
    vis = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    for hit in hits:
        who, score = cosine_best_match(hit.embedding, gallery, threshold)
        label = who if who else "unknown"
        x1, y1, x2, y2 = hit.bbox.astype(int)
        color = (0, 200, 0) if who else (0, 140, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        text = f"{label} {score:.2f}"
        (tw, th), bln = cv2.getTextSize(text, font, scale, thickness)
        pad = 6
        # putText origin y is the text baseline; keep label above box when possible
        if y1 - pad - th >= 0:
            baseline_y = y1 - pad
            bg_y1 = baseline_y - th - bln
            bg_y2 = baseline_y + bln
        else:
            baseline_y = y2 + th + pad
            bg_y1 = y2 + pad
            bg_y2 = baseline_y + bln
        cv2.rectangle(
            vis,
            (x1, bg_y1),
            (x1 + tw + 2 * pad, bg_y2),
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            text,
            (x1 + pad, baseline_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        if hit.kps is not None:
            draw_arcface_recognition_landmarks(
                vis, hit.kps, radius=5, font_scale=0.45, line_color=(200, 200, 200)
            )
    return vis


def run_recognize(
    image_path: str,
    gallery_root: Path,
    threshold: float,
    out_path: Optional[Path],
    show: bool,
    no_save: bool,
) -> None:
    img = read_image_bgr(image_path)
    gallery = load_gallery(gallery_root)
    if not gallery:
        print("Gallery is empty. Run: python main.py enroll  (optional: --name <id>)")
        return

    pipeline = FacePipeline()
    hits = pipeline.all_faces(img)
    print(f"Image: {image_path}")
    print(f"Faces detected: {len(hits)}")
    if not hits:
        return

    for i, hit in enumerate(hits, start=1):
        who, score = cosine_best_match(hit.embedding, gallery, threshold)
        label = who if who else "unknown"
        x1, y1, x2, y2 = hit.bbox.astype(int)
        print(f"  Face {i}: {label} (best similarity {score:.3f}) bbox=({x1},{y1})-({x2},{y2})")

    annotated = _annotate_recognition(img, hits, gallery, threshold)
    src = Path(image_path)
    default_out = src.with_name(f"{src.stem}_recognized.jpg")

    if out_path is not None:
        save_to = out_path
    elif not no_save:
        save_to = default_out
    else:
        save_to = None

    if save_to is not None:
        out_str = str(save_to)
        ok = cv2.imwrite(out_str, annotated)
        if ok:
            print(f"Saved annotated image: {out_str}")
        else:
            print(f"Failed to write image: {out_str}", file=sys.stderr)

    if show:
        win = "Recognition (any key to close)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, annotated)
        cv2.waitKey(0)
        cv2.destroyWindow(win)

    if save_to is None and not show:
        print(
            "No annotated output: pass --show or allow default save (omit --no-save).",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided face enrollment + recognition")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gdir = default_gallery_dir()

    p_en = sub.add_parser("enroll", help="Pose prompts and capture embeddings for one person")
    p_en.add_argument(
        "--name",
        default=None,
        help="Person id in the gallery (default: random UUID if omitted)",
    )
    p_en.add_argument("--video", default=None, help="Use video file instead of webcam")
    p_en.add_argument("--gallery-dir", type=Path, default=gdir, help="Gallery storage directory")
    p_en.add_argument(
        "--auto-pose",
        action="store_true",
        help="Webcam only: auto-capture each pose when head angle + lighting look good (SPACE still works)",
    )

    p_re = sub.add_parser("recognize", help="Detect faces in an image and match to gallery")
    p_re.add_argument("--image", required=True, help="Path to query image")
    p_re.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Cosine similarity threshold (typical 0.4–0.55 for buffalo_l)",
    )
    p_re.add_argument("--gallery-dir", type=Path, default=gdir, help="Gallery storage directory")
    p_re.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write annotated image with boxes (default: <image_stem>_recognized.jpg next to input)",
    )
    p_re.add_argument(
        "--show",
        action="store_true",
        help="Open a window with the annotated image until a key is pressed",
    )
    p_re.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write an annotated file (use with --show only if you only want a window)",
    )

    args = parser.parse_args()
    if args.cmd == "enroll":
        run_enroll(
            args.name,
            args.video,
            DEFAULT_POSE_PROMPTS,
            args.gallery_dir,
            auto_pose=args.auto_pose,
        )
    elif args.cmd == "recognize":
        run_recognize(
            args.image,
            args.gallery_dir,
            args.threshold,
            args.out,
            args.show,
            args.no_save,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()




