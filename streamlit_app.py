"""
Streamlit UI for face enrollment (pose snapshots) and recognition.

Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
from face_model import FacePipeline, cosine_best_match
from face_quality import assess_face_quality
from live_face_processor import (
    LiveFaceBoxProcessor,
    get_live_capture_metrics,
    get_shared_face_pipeline,
    take_live_capture_frame,
)
from gallery_store import (
    default_gallery_dir,
    load_gallery,
    merge_person,
    resolve_person_id,
    save_gallery,
)
from main import DEFAULT_POSE_PROMPTS, _annotate_recognition


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _show_cli_enroll_equivalent(person_id: str) -> None:
    """Same gallery / poses as `python main.py enroll --name …`."""
    root = Path(__file__).resolve().parent
    qid = shlex.quote(person_id)
    qroot = shlex.quote(str(root))
    base = f"cd {qroot} && python main.py enroll --name {qid}"
    auto = f"cd {qroot} && python main.py enroll --name {qid} --auto-pose"
    with st.expander("Same as terminal (CLI) — same `data/gallery`", expanded=True):
        st.markdown(
            "This enrollment uses the **same person id**, **same poses**, and **same embeddings file** "
            "as running from the project folder:"
        )
        st.code(base, language="bash")
        st.caption(
            "Optional: `--auto-pose` uses MediaPipe head angles + quality checks (webcam only), "
            "same as:"
        )
        st.code(auto, language="bash")


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
        "Same **5 poses** and **gallery** as `python main.py enroll --name <id>`. "
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

    st.markdown("##### Capture")
    st.caption(
        "Use **Capture** after you press **START** on the video. One row shows: **Live** camera + "
        "**BBox RGB** + **Gray** + **Aligned** + **Net** — scroll sideways on small screens."
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
        else:
            pipeline = get_shared_face_pipeline()
            hits = pipeline.all_faces(bgr)
            hits = FacePipeline._sort_by_area(hits)
            if not hits:
                st.error("No face in frame — stay in view and try again.")
            else:
                h0 = hits[0]
                if len(hits) > 1:
                    st.info(
                        f"Using the **closest** of **{len(hits)}** faces (largest box)."
                    )
                rep = assess_face_quality(bgr, h0.bbox, h0.det_score)
                for msg in rep.messages:
                    st.warning(msg)
                emb = h0.embedding.copy()
                embs.append(emb)
                st.session_state.live_enroll_embs = embs
                st.session_state.live_enroll_step = step + 1
                st.success(f"Stored pose {step + 1}/{n}.")
                with st.expander("Pipeline preview (last capture)", expanded=False):
                    try:
                        strip = pipeline.enrollment_pipeline_strip(bgr, h0)
                        st.image(_bgr_to_rgb(strip), use_container_width=True)
                    except Exception:
                        st.caption("Pipeline strip unavailable.")
                st.rerun()

    if len(embs) > 0:
        st.caption(f"Poses stored this session: **{len(embs)}** / {n}")


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

    tab1, tab2, tab3 = st.tabs(["Enroll", "Enroll (live)", "Recognize"])
    with tab1:
        _enrollment_ui(gallery_root)
    with tab2:
        _enrollment_live_ui(gallery_root)
    with tab3:
        _recognize_ui(gallery_root)


if __name__ == "__main__":
    main()
