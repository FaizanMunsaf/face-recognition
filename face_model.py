"""Face detection and ArcFace-style embeddings via InsightFace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError as e:  # pragma: no cover
    FaceAnalysis = None  # type: ignore
    face_align = None  # type: ignore
    _import_error = e
else:
    _import_error = None

# Order matches insightface.utils.face_align.arcface_dst (used by norm_crop → recognition).
ARCFACE_POINT_NAMES: Tuple[str, ...] = (
    "R-eye",  # subject's right eye (appears on left in image)
    "L-eye",
    "nose",
    "L-mouth",
    "R-mouth",
)

# Light mesh so you see how the five points relate on the face.
_ARC_LM_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 3),
    (2, 4),
    (3, 4),
)


def draw_arcface_recognition_landmarks(
    bgr: np.ndarray,
    kps: Union[np.ndarray, Sequence[Sequence[float]]],
    *,
    radius: int = 7,
    line_color: Tuple[int, int, int] = (200, 200, 200),
    line_thickness: int = 2,
    font_scale: float = 0.48,
    with_labels: bool = True,
) -> None:
    """
    Draw the 5 points the ArcFace pipeline uses for alignment (detector landmarks),
    with connecting lines and 1..5 + short names. Mutates ``bgr`` in place.
    """
    pts = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    h, w = bgr.shape[:2]
    pi = pts.astype(int)
    for a, b in _ARC_LM_EDGES:
        cv2.line(bgr, tuple(pi[a]), tuple(pi[b]), line_color, line_thickness, cv2.LINE_AA)
    colors = [
        (0, 255, 255),
        (255, 140, 0),
        (0, 200, 255),
        (0, 255, 100),
        (255, 80, 180),
    ]
    for i in range(5):
        c = (int(pi[i, 0]), int(pi[i, 1]))
        cv2.circle(bgr, c, radius + 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(bgr, c, radius, colors[i], -1, cv2.LINE_AA)
        if not with_labels:
            continue
        name = f"{i + 1}:{ARCFACE_POINT_NAMES[i]}"
        tx = min(c[0] + radius + 3, w - 2)
        ty = max(c[1] - radius - 4, int(font_scale * 22))
        cv2.putText(
            bgr,
            name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (40, 40, 40),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            name,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            colors[i],
            1,
            cv2.LINE_AA,
        )


def draw_arcface_template_on_chip(bgr_chip: np.ndarray, *, radius: int = 5) -> None:
    """Draw canonical 112-template landmark positions on an aligned square chip (any size)."""
    if face_align is None or bgr_chip.size == 0:
        return
    side = int(bgr_chip.shape[0])
    if side < 8:
        return
    tmpl = face_align.arcface_dst.astype(np.float32) * (float(side) / 112.0)
    draw_arcface_recognition_landmarks(
        bgr_chip, tmpl, radius=radius, font_scale=0.32, line_color=(160, 160, 160)
    )


@dataclass
class FaceHit:
    bbox: np.ndarray  # xyxy
    embedding: np.ndarray  # L2-normalized, cosine sim = dot product
    det_score: float = 1.0  # face detector confidence when available
    kps: Optional[np.ndarray] = None  # 5x2 landmarks -> norm_crop for ArcFace chip


class FacePipeline:
    """Loads buffalo_l (includes ArcFace recognition head)."""

    def __init__(self, ctx_id: int = -1, det_size: Tuple[int, int] = (640, 640)):
        if _import_error is not None:
            raise RuntimeError(
                "Install dependencies: pip install -r requirements.txt"
            ) from _import_error
        self._app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=ctx_id, det_size=det_size)

    def _hits(self, image_bgr: np.ndarray) -> List[FaceHit]:
        faces = self._app.get(image_bgr)
        out: List[FaceHit] = []
        for f in faces:
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            det = float(getattr(f, "det_score", 1.0))
            raw_kps = getattr(f, "kps", None)
            kps = (
                np.asarray(raw_kps, dtype=np.float32).copy()
                if raw_kps is not None
                else None
            )
            out.append(
                FaceHit(
                    bbox=f.bbox.astype(np.float32),
                    embedding=emb,
                    det_score=det,
                    kps=kps,
                )
            )
        return out

    @staticmethod
    def _sort_by_area(hits: List["FaceHit"]) -> List["FaceHit"]:
        def area(h: FaceHit) -> float:
            x1, y1, x2, y2 = h.bbox
            return float((x2 - x1) * (y2 - y1))

        return sorted(hits, key=area, reverse=True)

    def largest_face_hit(self, image_bgr: np.ndarray) -> Optional[FaceHit]:
        hits = self._hits(image_bgr)
        if not hits:
            return None
        h = self._sort_by_area(hits)[0]
        kps = h.kps.copy() if h.kps is not None else None
        return FaceHit(
            bbox=h.bbox.copy(),
            embedding=h.embedding.copy(),
            det_score=h.det_score,
            kps=kps,
        )

    def largest_face_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        h = self.largest_face_hit(image_bgr)
        return None if h is None else h.embedding.copy()

    def all_faces(self, image_bgr: np.ndarray) -> List[FaceHit]:
        return self._hits(image_bgr)

    def _recognition_model(self):
        return self._app.models.get("recognition")

    def _arcface_chip_size(self) -> int:
        rec = self._recognition_model()
        if rec is None or not hasattr(rec, "input_size"):
            return 112
        return int(rec.input_size[0])

    def aligned_face_chip(self, image_bgr: np.ndarray, hit: FaceHit) -> Optional[np.ndarray]:
        """112x112 BGR warp — same geometry as ArcFace ONNX uses before mean/std."""
        if _import_error is not None or face_align is None or hit.kps is None:
            return None
        size = self._arcface_chip_size()
        return face_align.norm_crop(
            image_bgr, landmark=hit.kps, image_size=size, mode="arcface"
        )

    def network_preprocess_preview_bgr(self, aligned_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Visualize ArcFace ONNX input after resize, RGB swap, (x-mean)/std — mapped back to 8-bit BGR.
        Colors differ from (2) because of channel order + normalization.
        """
        rec = self._recognition_model()
        if rec is None:
            return None
        mean = float(getattr(rec, "input_mean", 127.5))
        std = float(getattr(rec, "input_std", 127.5))
        input_size = tuple(int(x) for x in rec.input_size)
        blob = cv2.dnn.blobFromImages(
            [aligned_bgr],
            1.0 / std,
            input_size,
            (mean, mean, mean),
            swapRB=True,
        )
        x = blob[0].transpose(1, 2, 0)
        vis = x * std + mean
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    def enrollment_pipeline_strip(
        self,
        frame_bgr: np.ndarray,
        hit: FaceHit,
        panel: int = 200,
        title_h: int = 40,
    ) -> np.ndarray:
        """
        Three columns: raw bbox crop | aligned ArcFace chip | net tensor (denormalized for display).
        """
        labels = (
            "1) Bbox crop + 5 landmarks",
            "2) Aligned chip + ArcFace template",
            "3) Net input (RGB swap + mean/std)",
        )

        def titled_panel(img: np.ndarray, label: str) -> np.ndarray:
            side = panel
            if img.size == 0:
                tile = np.zeros((side, side, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
            bar = np.zeros((title_h, side, 3), dtype=np.uint8)
            cv2.putText(
                bar,
                label,
                (4, title_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            return np.vstack([bar, tile])

        x1, y1, x2, y2 = hit.bbox.astype(int)
        fh, fw = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 > x1 and y2 > y1:
            raw_crop = frame_bgr[y1:y2, x1:x2]
        else:
            raw_crop = np.zeros((panel, panel, 3), dtype=np.uint8)

        aligned = self.aligned_face_chip(frame_bgr, hit)
        had_kps = hit.kps is not None
        if aligned is None:
            sz = self._arcface_chip_size()
            aligned = cv2.resize(raw_crop, (sz, sz), interpolation=cv2.INTER_AREA)
            labels = (
                labels[0],
                "2) No kps — bbox resize only",
                labels[2],
            )

        net_vis = self.network_preprocess_preview_bgr(aligned)
        if net_vis is None:
            net_vis = aligned.copy()

        raw_vis = raw_crop.copy()
        if had_kps and raw_vis.size > 0:
            kc = hit.kps.astype(np.float32).copy()
            kc[:, 0] -= float(x1)
            kc[:, 1] -= float(y1)
            draw_arcface_recognition_landmarks(
                raw_vis, kc, radius=5, font_scale=0.36, line_color=(190, 190, 190)
            )

        aligned_vis = aligned.copy()
        if had_kps:
            draw_arcface_template_on_chip(aligned_vis, radius=4)

        p0 = titled_panel(raw_vis, labels[0])
        p1 = titled_panel(aligned_vis, labels[1])
        p2 = titled_panel(net_vis, labels[2])
        gap = np.zeros((p0.shape[0], 8, 3), dtype=np.uint8)
        return np.hstack([p0, gap, p1, gap, p2])


def cosine_best_match(
    query: np.ndarray,
    gallery: dict,
    threshold: float,
) -> Tuple[Optional[str], float]:
    """
    gallery: name -> list of embeddings (multiple poses per person).
    Returns (best_name_or_None, best_score).
    """
    query = np.asarray(query, dtype=np.float32).ravel()
    best_name: Optional[str] = None
    best = -1.0
    for name, embs in gallery.items():
        for ref in embs:
            ref = np.asarray(ref, dtype=np.float32).ravel()
            s = float(np.dot(query, ref))
            if s > best:
                best = s
                best_name = name
    if best < threshold:
        return None, best
    return best_name, best


def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
