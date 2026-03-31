"""
Streamlit UI for face enrollment (pose snapshots) and recognition.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import shlex
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
from face_model import FacePipeline, cosine_best_match
from face_quality import assess_face_quality
from live_face_processor import (
    LIVE_AUTO_STABLE_FRAMES,
    LiveFaceBoxProcessor,
    LiveSevenStepProcessor,
    SEVEN_STEPS,
    configure_live_auto_pose,
    configure_seven_step,
    consume_pending_auto_capture,
    consume_seven_step_capture,
    get_live_auto_status,
    get_live_capture_metrics,
    get_shared_face_pipeline,
    reset_seven_step,
    take_live_capture_frame,
)
from gallery_store import (
    default_gallery_dir,
    load_gallery,
    merge_person,
    resolve_person_id,
    save_gallery,
)
from main import AUTO_POSE_QUALITY_KW, DEFAULT_POSE_PROMPTS, _annotate_recognition


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _live_apply_pose_capture(bgr: np.ndarray, *, auto: bool) -> bool:
    """
    Append embedding for current ``live_enroll_step`` and advance step.
    Returns True if a face was stored.
    """
    step = st.session_state.live_enroll_step
    n = len(DEFAULT_POSE_PROMPTS)
    embs: List[np.ndarray] = list(st.session_state.live_enroll_embs)
    pipeline = get_shared_face_pipeline()
    hits = pipeline.all_faces(bgr)
    hits = FacePipeline._sort_by_area(hits)
    if not hits:
        return False
    h0 = hits[0]
    rep = assess_face_quality(bgr, h0.bbox, h0.det_score, **AUTO_POSE_QUALITY_KW)
    for msg in rep.messages:
        st.warning(msg)
    emb = h0.embedding.copy()
    embs.append(emb)
    st.session_state.live_enroll_embs = embs
    st.session_state.live_enroll_step = step + 1
    tag = "Auto-pose" if auto else "Manual"
    st.success(f"{tag}: stored pose {step + 1}/{n}.")
    if len(hits) > 1:
        st.caption(f"Closest of **{len(hits)}** faces (largest box) was used.")
    with st.expander("Pipeline preview (last capture)", expanded=False):
        try:
            strip = pipeline.enrollment_pipeline_strip(bgr, h0)
            st.image(_bgr_to_rgb(strip), use_container_width=True)
        except Exception:
            st.caption("Pipeline strip unavailable.")
    return True


@st.fragment(run_every=timedelta(seconds=0.2))
def _live_auto_pose_poll_fragment() -> None:
    """Poll worker-thread auto-capture queue (Enroll live tab only; no-op if session not started)."""
    if "live_enroll_step" not in st.session_state:
        return
    if not st.session_state.get("live_auto_pose", True):
        return
    cur_step = st.session_state.live_enroll_step
    nposes = len(DEFAULT_POSE_PROMPTS)
    if cur_step >= nposes:
        return
    pending = consume_pending_auto_capture()
    if pending is not None:
        if _live_apply_pose_capture(pending, auto=True):
            st.rerun()
        return
    stc = get_live_auto_status()
    if not stc["enabled"]:
        return
    yy, pp = stc["yaw"], stc["pitch"]
    ys = f"{yy:+.0f}" if yy is not None else "—"
    ps = f"{pp:+.0f}" if pp is not None else "—"
    stab_goal = "instant" if LIVE_AUTO_STABLE_FRAMES <= 0 else str(LIVE_AUTO_STABLE_FRAMES)
    st.caption(
        f"Auto-pose: yaw={ys}° pitch={ps}° • "
        f"angle={'OK' if stc['pose_ok'] else 'adjust to match prompt'} • "
        f"stable **{stc['stable']}**/{stab_goal} • "
        f"quality: {'improve light/focus' if stc['quality_bad'] else 'OK'}"
    )


def _show_cli_enroll_equivalent(person_id: str) -> None:
    """Same gallery / poses as `python main.py enroll --name …`."""
    root = Path(__file__).resolve().parent
    qid = shlex.quote(person_id)
    qroot = shlex.quote(str(root))
    base = f"cd {qroot} && python main.py enroll --name {qid}"
    no_auto = f"cd {qroot} && python main.py enroll --name {qid} --no-auto-pose"
    with st.expander("Same as terminal (CLI) — same `data/gallery`", expanded=True):
        st.markdown(
            "This enrollment uses the **same person id**, **same poses**, and **same embeddings file** "
            "as running from the project folder:"
        )
        st.code(base, language="bash")
        st.caption(
            "Webcam: **auto-pose is ON by default** (MediaPipe + head-angle gates). "
            "SPACE-only (no auto) matches:"
        )
        st.code(no_auto, language="bash")


def _bytes_or_upload_to_bgr(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr


def _reset_enrollment() -> None:
    for k in ("enroll_step", "enroll_embs", "enroll_name_active", "enroll_id_was_uuid"):
        st.session_state.pop(k, None)


def _reset_live_enrollment() -> None:
    for k in (
        "live_enroll_step",
        "live_enroll_embs",
        "live_enroll_name_active",
        "live_enroll_id_was_uuid",
    ):
        st.session_state.pop(k, None)


def _enrollment_ui(gallery_root: Path) -> None:
    st.subheader("Enroll a person")
    st.caption(
        "Complete each pose using the webcam snapshot or an uploaded photo, then click **Save this pose**. "
        "If several people are visible, the **closest face** (largest box in the frame) is enrolled."
    )

    name = st.text_input(
        "Person id (optional)",
        key="enroll_name_input",
        placeholder="Leave blank for auto UUID, or e.g. alice",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start / reset enrollment", type="primary"):
            st.session_state.enroll_step = 0
            st.session_state.enroll_embs = []
            st.session_state.enroll_name_active = resolve_person_id(name)
            st.session_state.enroll_id_was_uuid = not (name or "").strip()
            st.rerun()
    with col_b:
        if st.button("Cancel enrollment"):
            _reset_enrollment()
            st.rerun()

    if "enroll_step" not in st.session_state:
        st.info(
            "Click **Start / reset enrollment**. If the id field is empty, a **UUID** is assigned — "
            "equivalent to `python main.py enroll` with no `--name` (see expander after you start)."
        )
        return

    step = st.session_state.enroll_step
    embs: List[np.ndarray] = st.session_state.enroll_embs
    active_name = st.session_state.get("enroll_name_active", "")
    if st.session_state.get("enroll_id_was_uuid"):
        st.info(f"**Person id (UUID):** `{active_name}` — same id the CLI would print for `python main.py enroll` (no `--name`).")
    _show_cli_enroll_equivalent(active_name)

    n = len(DEFAULT_POSE_PROMPTS)
    st.progress(min(float(step) / float(n), 1.0))
    if step < n:
        st.markdown(f"**Pose {step + 1} / {n}** — {DEFAULT_POSE_PROMPTS[step]}")
    else:
        st.success(f"All {n} poses captured for **{active_name}**.")
        if st.button("Write to gallery"):
            gallery = load_gallery(gallery_root)
            gallery = merge_person(gallery, active_name, embs)
            save_gallery(gallery, gallery_root)
            st.success(f"Saved {len(embs)} embeddings to `{gallery_root}`.")
            _reset_enrollment()
            st.rerun()
        return

    source = st.radio("Image source", ("Webcam", "Upload file"), horizontal=True)
    img_bgr: Optional[np.ndarray] = None

    if source == "Webcam":
        shot = st.camera_input("Take a snapshot when ready", key=f"cam_{step}")
        if shot is not None:
            img_bgr = _bytes_or_upload_to_bgr(shot.getvalue())
    else:
        up = st.file_uploader("Photo for this pose", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            img_bgr = _bytes_or_upload_to_bgr(up.getvalue())

    pipeline = get_shared_face_pipeline()

    if img_bgr is None:
        st.warning("Provide an image to continue.")
        return

    hits = pipeline.all_faces(img_bgr)
    hits = FacePipeline._sort_by_area(hits)

    if not hits:
        st.error("No face detected. Try again with better light and face the camera.")
        return

    h0 = hits[0]
    if len(hits) > 1:
        st.info(
            f"**{len(hits)}** faces detected — saving the **closest** one (largest face box). "
            "Step closer to the camera than others if you want to be the subject."
        )
    rep = assess_face_quality(img_bgr, h0.bbox, h0.det_score)
    for msg in rep.messages:
        st.warning(msg)

    c1, c2 = st.columns(2)
    with c1:
        preview = _bgr_to_rgb(img_bgr)
        st.image(preview, caption="Full frame — subject is the largest face if several appear", use_container_width=True)
    with c2:
        try:
            strip = pipeline.enrollment_pipeline_strip(img_bgr, h0)
            st.image(_bgr_to_rgb(strip), caption="Pipeline: crop → aligned → net preview", use_container_width=True)
        except Exception:
            st.caption("Pipeline preview unavailable.")

    if st.button("Save this pose", type="primary", key=f"save_pose_{step}"):
        emb = h0.embedding.copy()
        embs.append(emb)
        st.session_state.enroll_embs = embs
        st.session_state.enroll_step = step + 1
        st.success(f"Stored pose {step + 1}/{n}.")
        st.rerun()


def _enrollment_live_ui(gallery_root: Path) -> None:
    """Webcam enrollment with live overlay of all face bounding boxes (streamlit-webrtc)."""
    try:
        from streamlit_webrtc import (
            RTCConfiguration,
            VideoHTMLAttributes,
            WebRtcMode,
            webrtc_streamer,
        )
    except ImportError:  # pragma: no cover
        st.error("Install **streamlit-webrtc**: `pip install streamlit-webrtc`")
        return

    st.subheader("Enroll (live camera)")
    st.caption(
        "Uses **this computer’s camera** in the browser (allow access; pick the built-in webcam if asked). "
        "CLI: `python main.py enroll` uses laptop camera **index 0** by default (`--camera 1` if needed)."
    )
    st.caption(
        "Same **6 poses** and **gallery** as `python main.py enroll --name <id>`. "
        "Stream shows **one main view** plus **four panels**: bbox crop (RGB), **grayscale** crop, **aligned** chip, "
        "and **network RGB** preview. With multiple faces, **Capture** uses the **closest** face (largest box)."
    )

    name = st.text_input(
        "Person id (optional)",
        key="live_enroll_name_input",
        placeholder="Leave blank for auto UUID, or e.g. alice",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start / reset live enrollment", type="primary", key="live_start"):
            st.session_state.live_enroll_step = 0
            st.session_state.live_enroll_embs = []
            st.session_state.live_enroll_name_active = resolve_person_id(name)
            st.session_state.live_enroll_id_was_uuid = not (name or "").strip()
            st.rerun()
    with col_b:
        if st.button("Cancel live enrollment", key="live_cancel"):
            _reset_live_enrollment()
            st.rerun()

    if "live_enroll_step" not in st.session_state:
        st.info(
            "Click **Start / reset live enrollment**, then **START** the camera below. "
            "A UUID is used if the id field is empty — **same gallery** as "
            "`python main.py enroll` (no `--name` needed for auto-UUID)."
        )
        return

    step = st.session_state.live_enroll_step
    embs: List[np.ndarray] = st.session_state.live_enroll_embs
    active_name = st.session_state.get("live_enroll_name_active", "")
    if st.session_state.get("live_enroll_id_was_uuid"):
        st.info(
            f"**Person id (UUID):** `{active_name}` — use this exact value in "
            f"`python main.py enroll --name ...` if you repeat enrollment in the terminal."
        )
    _show_cli_enroll_equivalent(active_name)

    n = len(DEFAULT_POSE_PROMPTS)
    st.progress(min(float(step) / float(n), 1.0))

    if step >= n:
        st.success(f"All {n} poses captured for **{active_name}**.")
        if st.button("Write to gallery", key="live_write_gallery"):
            gallery = load_gallery(gallery_root)
            gallery = merge_person(gallery, active_name, embs)
            save_gallery(gallery, gallery_root)
            st.success(f"Saved {len(embs)} embeddings to `{gallery_root}`.")
            _reset_live_enrollment()
            st.rerun()
        return

    st.markdown(f"**Pose {step + 1} / {n}** — {DEFAULT_POSE_PROMPTS[step]}")

    if "live_auto_pose" not in st.session_state:
        st.session_state.live_auto_pose = True
    st.checkbox(
        "Automatically save this pose when your head matches the instruction (MediaPipe — same logic as CLI webcam enroll)",
        key="live_auto_pose",
    )
    configure_live_auto_pose(
        bool(st.session_state.live_auto_pose) and step < n,
        step if step < n else 0,
    )

    st.markdown("##### Capture")
    st.caption(
        "With auto-save on, hold the pose until the video overlay **stable** counter fills; "
        "or click **Capture** anytime. Row = **Live** + **BBox RGB** + **Gray** + **Aligned** + **Net**."
    )
    # WebRTC must be instantiated every run; the button is drawn into this slot so it stays above the player.
    capture_btn_placeholder = st.empty()

    ctx = webrtc_streamer(
        key="live_enroll_webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=LiveFaceBoxProcessor,
        async_processing=True,
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=True,
            style={"width": "100%", "maxWidth": "100%", "height": "auto"},
        ),
    )

    vp = ctx.video_processor
    playing = bool(ctx.state.playing)
    nfc_buf, _subject_ok_buf, has_frame_buf = get_live_capture_metrics()
    # Do not gate on ctx.state.playing: reruns can flip it false and tear down the worker
    # before this handler runs; the shared buffer still holds the last frame from the worker.
    cap_disabled = not has_frame_buf

    if vp is not None and isinstance(vp, LiveFaceBoxProcessor):
        nfc = vp.last_face_count
        if nfc == 1:
            st.success("**1** face — capture will use it.")
        elif nfc == 0:
            st.info("No face in frame yet — step into view.")
        else:
            st.info(
                f"**{nfc}** faces — **Capture** will use the **closest** (largest box). "
                "Move closer to the camera to be that face."
            )
    elif has_frame_buf:
        if nfc_buf == 1:
            st.success("**1** face — capture will use it.")
        elif nfc_buf == 0:
            st.info("No face in frame yet — step into view.")
        else:
            st.info(
                f"**{nfc_buf}** faces — **Capture** uses the **closest** (largest box)."
            )
    elif not playing:
        st.warning("Press **START** on the video below and allow the camera.")
    else:
        st.info("Waiting for the first video frame…")

    if capture_btn_placeholder.button(
        "Capture this pose",
        type="primary",
        key=f"live_cap_{step}",
        disabled=cap_disabled,
    ):
        bgr = take_live_capture_frame()
        if bgr is None:
            st.error("No video frame — start the camera and wait a moment.")
        elif not _live_apply_pose_capture(bgr, auto=False):
            st.error("No face in frame — stay in view and try again.")
        st.rerun()

    _live_auto_pose_poll_fragment()

    if len(embs) > 0:
        st.caption(f"Poses stored this session: **{len(embs)}** / {n}")


@st.fragment(run_every=timedelta(seconds=0.3))
def _seven_step_poll_fragment() -> None:
    """Poll the 7-step processor for auto-captured frames."""
    if "ss_step" not in st.session_state:
        return
    step = st.session_state.ss_step
    n = len(SEVEN_STEPS)
    if step >= n:
        return
    bgr = consume_seven_step_capture()
    if bgr is None:
        return
    pipeline = get_shared_face_pipeline()
    hits = pipeline.all_faces(bgr)
    embs: List[np.ndarray] = list(st.session_state.get("ss_embs", []))
    images: list = list(st.session_state.get("ss_images", []))
    strips: list = list(st.session_state.get("ss_strips", []))
    if hits:
        hits = FacePipeline._sort_by_area(hits)
        h0 = hits[0]
        embs.append(h0.embedding.copy())
        try:
            strips.append(pipeline.enrollment_pipeline_strip(bgr, h0))
        except Exception:
            strips.append(None)
    else:
        strips.append(None)
    images.append(bgr)
    st.session_state.ss_embs = embs
    st.session_state.ss_images = images
    st.session_state.ss_strips = strips
    st.session_state.ss_step = step + 1
    configure_seven_step(step + 1)
    st.rerun()


def _reset_seven_step_session() -> None:
    for k in ("ss_step", "ss_embs", "ss_images", "ss_strips", "ss_name", "ss_id_was_uuid"):
        st.session_state.pop(k, None)


def _seven_step_capture_ui(gallery_root: Path) -> None:
    """7-step guided face capture with live overlays in the browser."""
    try:
        from streamlit_webrtc import (
            RTCConfiguration,
            VideoHTMLAttributes,
            WebRtcMode,
            webrtc_streamer,
        )
    except ImportError:
        st.error("Install **streamlit-webrtc**: `pip install streamlit-webrtc`")
        return

    st.subheader("7-Step Face Capture")
    st.caption(
        "Guided capture with visual overlays: **5 head poses** + **eyes closed** + **mouth open**. "
        "Hold each pose for ~1.5 s to auto-capture. Embeddings are computed for gallery enrollment."
    )

    name = st.text_input(
        "Person id (optional)",
        key="ss_name_input",
        placeholder="Leave blank for auto UUID, or e.g. alice",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start / reset 7-step capture", type="primary", key="ss_start"):
            st.session_state.ss_step = 0
            st.session_state.ss_embs = []
            st.session_state.ss_images = []
            st.session_state.ss_strips = []
            st.session_state.ss_name = resolve_person_id(name)
            st.session_state.ss_id_was_uuid = not (name or "").strip()
            reset_seven_step()
            st.rerun()
    with col_b:
        if st.button("Cancel", key="ss_cancel"):
            _reset_seven_step_session()
            st.rerun()

    if "ss_step" not in st.session_state:
        st.info("Click **Start / reset 7-step capture** to begin, then allow camera access below.")
        return

    step = st.session_state.ss_step
    n = len(SEVEN_STEPS)
    active_name = st.session_state.get("ss_name", "")

    if st.session_state.get("ss_id_was_uuid"):
        st.info(f"**Person id (UUID):** `{active_name}`")

    st.progress(min(float(step) / float(n), 1.0))

    if step >= n:
        st.success(f"All {n} steps captured for **{active_name}**!")
        images = st.session_state.get("ss_images", [])
        strips = st.session_state.get("ss_strips", [])
        embs: List[np.ndarray] = st.session_state.get("ss_embs", [])

        if images:
            st.markdown("##### Captured frames")
            cols = st.columns(min(len(images), 7))
            for i, img_bgr in enumerate(images):
                with cols[i % 7]:
                    cap = SEVEN_STEPS[i]["id"] if i < len(SEVEN_STEPS) else f"step {i+1}"
                    st.image(_bgr_to_rgb(img_bgr), caption=cap, use_container_width=True)

        if strips:
            st.markdown("##### ArcFace pipeline (what the model sees)")
            for i, strip in enumerate(strips):
                step_id = SEVEN_STEPS[i]["id"] if i < len(SEVEN_STEPS) else f"step {i+1}"
                if strip is not None:
                    st.image(
                        _bgr_to_rgb(strip),
                        caption=f"Step {i+1}: {step_id}  —  bbox crop + landmarks | aligned chip | network input",
                        use_container_width=True,
                    )
                else:
                    st.caption(f"Step {i+1}: {step_id} — InsightFace did not detect a face in this frame")

        st.caption(f"Embeddings extracted: **{len(embs)}** / {n}")
        if len(embs) < n:
            st.warning(
                f"Only {len(embs)} embeddings could be extracted (InsightFace may miss extreme poses "
                "or closed-eye frames). Gallery will store what was found."
            )
        if embs and st.button("Write to gallery", key="ss_write"):
            gallery = load_gallery(gallery_root)
            gallery = merge_person(gallery, active_name, embs)
            save_gallery(gallery, gallery_root)
            st.success(f"Saved {len(embs)} embeddings to `{gallery_root}`.")
            _reset_seven_step_session()
            st.rerun()
        return

    st.markdown(f"**Step {step + 1} / {n}** — {SEVEN_STEPS[step]['instruction']}")
    configure_seven_step(step)

    webrtc_streamer(
        key="seven_step_webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=LiveSevenStepProcessor,
        async_processing=True,
        video_html_attrs=VideoHTMLAttributes(
            autoPlay=True,
            controls=True,
            style={"width": "100%", "maxWidth": "100%", "height": "auto"},
        ),
    )

    _seven_step_poll_fragment()

    if step > 0:
        st.caption(f"Steps completed: **{step}** / {n}")


def _recognize_ui(gallery_root: Path) -> None:
    st.subheader("Recognize faces in a photo")
    threshold = st.slider("Match threshold (cosine)", 0.25, 0.75, 0.45, 0.01)

    gallery = load_gallery(gallery_root)
    if not gallery:
        st.warning("Gallery is empty. Enroll someone first.")
        return

    st.caption(f"Gallery: **{', '.join(gallery.keys())}**")

    up = st.file_uploader("Image to analyze", type=["jpg", "jpeg", "png", "webp"])
    if up is None:
        return

    bgr = _bytes_or_upload_to_bgr(up.getvalue())
    if bgr is None:
        st.error("Could not decode image.")
        return

    pipeline = get_shared_face_pipeline()
    hits = pipeline.all_faces(bgr)
    if not hits:
        st.error("No faces detected.")
        return

    rows = []
    for i, hit in enumerate(hits, start=1):
        who, score = cosine_best_match(hit.embedding, gallery, threshold)
        label = who if who else "unknown"
        x1, y1, x2, y2 = hit.bbox.astype(int)
        rows.append(
            {
                "Face": i,
                "Match": label,
                "Score": round(score, 3),
                "bbox": f"({x1},{y1})-({x2},{y2})",
            }
        )

    st.dataframe(rows, use_container_width=True)
    vis = _annotate_recognition(bgr, hits, gallery, threshold)
    st.image(_bgr_to_rgb(vis), caption="Annotated result", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Take Talker — Faces", layout="wide")
    st.title("Take Talker")
    st.caption("ArcFace-style embeddings (InsightFace) — enroll with pose prompts, match in photos.")

    gallery_root = default_gallery_dir()
    st.sidebar.markdown(f"**Gallery directory**  \n`{gallery_root}`")
    g = load_gallery(gallery_root)
    if g:
        st.sidebar.markdown("**Enrolled**")
        for name, embs in sorted(g.items()):
            st.sidebar.write(f"- {name} ({len(embs)} embeddings)")
    else:
        st.sidebar.info("No enrolled faces yet.")

    tab1, tab2, tab3, tab4 = st.tabs(["Enroll", "Enroll (live)", "7-Step Capture", "Recognize"])
    with tab1:
        _enrollment_ui(gallery_root)
    with tab2:
        _enrollment_live_ui(gallery_root)
    with tab3:
        _seven_step_capture_ui(gallery_root)
    with tab4:
        _recognize_ui(gallery_root)


if __name__ == "__main__":
    main()
