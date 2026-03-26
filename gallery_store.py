"""Persist enrolled face embeddings per person."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def resolve_person_id(name: Optional[str] = None) -> str:
    """
    Gallery key for a person. Non-empty ``name`` is used as-is; otherwise a new UUID4.
    """
    s = (name or "").strip()
    return s if s else str(uuid.uuid4())

META_SUFFIX = ".json"


def default_gallery_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "gallery"


def _np_path(root: Path) -> Path:
    return root / "embeddings.npz"


def _meta_path(root: Path) -> Path:
    return root / f"meta{META_SUFFIX}"


def load_gallery(root: Path | None = None) -> Dict[str, List[np.ndarray]]:
    root = root or default_gallery_dir()
    npz = _np_path(root)
    meta = _meta_path(root)
    if not npz.exists() or not meta.exists():
        return {}
    with open(meta, "r", encoding="utf-8") as f:
        meta_obj = json.load(f)
    order: List[str] = meta_obj["names"]
    counts: Dict[str, int] = {k: int(v) for k, v in meta_obj["counts"].items()}
    z = np.load(npz)
    gallery: Dict[str, List[np.ndarray]] = {n: [] for n in order}
    idx = 0
    for name in order:
        for _ in range(counts[name]):
            key = f"e_{idx}"
            gallery[name].append(z[key].astype(np.float32))
            idx += 1
    return gallery


def save_gallery(gallery: Dict[str, List[np.ndarray]], root: Path | None = None) -> None:
    root = root or default_gallery_dir()
    os.makedirs(root, exist_ok=True)
    names: List[str] = []
    counts: Dict[str, int] = {}
    arrays = {}
    idx = 0
    for name in sorted(gallery.keys()):
        embs = gallery[name]
        if not embs:
            continue
        names.append(name)
        counts[name] = len(embs)
        for e in embs:
            arrays[f"e_{idx}"] = np.asarray(e, dtype=np.float32)
            idx += 1
    np.savez(_np_path(root), **arrays)
    with open(_meta_path(root), "w", encoding="utf-8") as f:
        json.dump({"names": names, "counts": counts}, f, indent=2)


def merge_person(
    gallery: Dict[str, List[np.ndarray]],
    name: str,
    new_embeddings: List[np.ndarray],
) -> Dict[str, List[np.ndarray]]:
    g = {k: list(v) for k, v in gallery.items()}
    g.setdefault(name, [])
    g[name].extend(new_embeddings)
    return g
