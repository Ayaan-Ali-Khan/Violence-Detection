"""Dataset metadata extraction helpers for local video collections."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".webm"}
LABELS = {
    "fight": (1, "violent"),
    "violence": (1, "violent"),
    "violent": (1, "violent"),
    "nonfight": (0, "nonviolent"),
    "nonviolence": (0, "nonviolent"),
    "nonviolent": (0, "nonviolent"),
}


def _normalize_component(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.lower())


def infer_label(path: Path, root: Path) -> tuple[int, str]:
    """Infer a binary label from a class directory or Hockey Fights filename."""
    for component in reversed(path.relative_to(root).parts[:-1]):
        normalized = _normalize_component(component)
        if normalized in LABELS:
            return LABELS[normalized]
    # The original Hockey Fights collection is flat and uses fi/no filename
    # prefixes for fight/non-fight. Accept its common filename variants, such
    # as ``fi123_xvid.avi`` and ``no65___xvid.avi``.
    prefix = path.name.lower()[:2]
    if prefix in {"fi", "no"} and len(path.name) > 2 and path.name[2].isdigit():
        return (1, "violent") if prefix == "fi" else (0, "nonviolent")
    raise ValueError(f"Cannot infer violent/nonviolent label from path: {path}")


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_hash(frame: np.ndarray) -> str | None:
    """Return a compact 64-bit average hash; ignore near-uniform frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if float(gray.std()) < 5.0:
        return None
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = small >= small.mean()
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def inspect_video(path: Path) -> dict[str, object]:
    """Read container metadata, test decoding, and compute stable fingerprints."""
    stats = path.stat()
    result: dict[str, object] = {
        "fps": None,
        "frame_count": None,
        "duration_sec": None,
        "width": None,
        "height": None,
        "resolution": None,
        "codec": None,
        "file_size_bytes": stats.st_size,
        "decode_ok": False,
        "decode_error": None,
        "sha256": _sha256(path),
        "sample_frame_hashes": "",
    }
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            result["decode_error"] = "OpenCV could not open the video"
            return result

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> shift) & 0xFF) for shift in (0, 8, 16, 24))
        result.update(
            fps=fps if fps > 0 else None,
            frame_count=frame_count if frame_count > 0 else None,
            duration_sec=(frame_count / fps) if fps > 0 and frame_count > 0 else None,
            width=width if width > 0 else None,
            height=height if height > 0 else None,
            resolution=f"{width}x{height}" if width > 0 and height > 0 else None,
            codec=codec.strip("\x00 ") or None,
        )

        ok, first = cap.read()
        if not ok or first is None:
            result["decode_error"] = "Video opened, but the first frame could not be decoded"
            return result
        hashes = []
        if (frame_hash := _frame_hash(first)) is not None:
            hashes.append(frame_hash)
        if frame_count > 1:
            for fraction in (0.25, 0.5, 0.75, 0.95):
                frame_index = min(int((frame_count - 1) * fraction), frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame_hash = _frame_hash(frame)
                    if frame_hash is not None:
                        hashes.append(frame_hash)
        result["sample_frame_hashes"] = ";".join(dict.fromkeys(hashes))
        result["decode_ok"] = True
        return result
    except Exception as exc:
        result["decode_error"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        cap.release()


def build_metadata(dataset: str, root: str | Path) -> pd.DataFrame:
    """Build a deterministic metadata table for all supported videos under root."""
    root = Path(root).resolve()
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES),
        key=lambda p: p.as_posix().lower(),
    )
    rows = []
    for path in files:
        label, label_name = infer_label(path, root)
        relative = path.relative_to(root)
        split = relative.parts[0] if dataset.lower() in {"rwf2000", "rwf-2000"} else None
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "relative_path": path.relative_to(Path.cwd().resolve()).as_posix()
                if path.is_relative_to(Path.cwd().resolve())
                else path.as_posix(),
                "label": label,
                "label_name": label_name,
                **inspect_video(path),
            }
        )
    return pd.DataFrame(rows)
