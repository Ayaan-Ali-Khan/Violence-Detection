"""Record code, dependency, configuration, and checkpoint hashes for a run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_INPUTS = [
    "configs/detection.yaml",
    "configs/bytetrack.yaml",
    "configs/interaction_crops.yaml",
    "configs/preprocess_crops.yaml",
    "configs/preprocess.yaml",
    "configs/train_swin3d_t.yaml",
    "configs/train_x3d_fusion.yaml",
    "configs/train_x3d_m_fusion.yaml",
    "data/splits/rwf2000.csv",
    "data/splits/rlvs_eval.csv",
    "data/splits/hockeyfights_eval.csv",
]
CHECKPOINTS = [
    "model_checkpoints/phase6/swin3d_t_rwf2000_paired_views/best.pt",
    "model_checkpoints/phase6/x3d_m_rwf2000_late_fusion/best.pt",
    "model_checkpoints/phase5/x3d_s_rwf2000_late_fusion/best.pt",
]
DETECTOR_WEIGHTS = "yolo11n.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_manifest(
    checkpoint_root: Path = PROJECT_ROOT / "model_checkpoints",
    detector_weights: Path = PROJECT_ROOT / DETECTOR_WEIGHTS,
    external_artifacts: list[Path] | None = None,
) -> dict[str, Any]:
    inputs = {}
    for relative in TRACKED_INPUTS:
        path = PROJECT_ROOT / relative
        inputs[relative] = {
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    for relative in CHECKPOINTS:
        path = checkpoint_root / Path(relative).relative_to("model_checkpoints")
        inputs[relative] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    inputs[DETECTOR_WEIGHTS] = {
        "path": str(detector_weights.resolve()),
        "exists": detector_weights.is_file(),
        "sha256": sha256(detector_weights) if detector_weights.is_file() else None,
        "size_bytes": detector_weights.stat().st_size
        if detector_weights.is_file()
        else None,
    }
    for path in external_artifacts or []:
        resolved = path.resolve()
        inputs[f"external:{resolved}"] = {
            "path": str(resolved),
            "exists": resolved.is_file(),
            "sha256": sha256(resolved) if resolved.is_file() else None,
            "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
        }
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "pytorchvideo",
        "ultralytics",
        "numpy",
        "opencv-python-headless",
        "pandas",
        "PyYAML",
        "scikit-learn",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = git_value("rev-parse", "HEAD")
        dirty = bool(git_value("status", "--porcelain"))
    except Exception:
        commit, dirty = None, None
    return {
        "schema_version": 1,
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "packages": packages,
        "inputs": inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/phase8/reproducibility_manifest.json"),
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail unless the Git checkout has no tracked or untracked changes.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "model_checkpoints",
        help="Folder containing phase4/, phase5/, and phase6/ checkpoints.",
    )
    parser.add_argument(
        "--detector-weights",
        type=Path,
        default=PROJECT_ROOT / DETECTOR_WEIGHTS,
        help="YOLO11n weights to hash when available.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        type=Path,
        default=[],
        help="Additional file to hash, such as a calibration threshold or reference prediction file. Repeatable.",
    )
    args = parser.parse_args()
    for relative in TRACKED_INPUTS[:5]:
        with (PROJECT_ROOT / relative).open(encoding="utf-8") as stream:
            yaml.safe_load(stream)
    manifest = create_manifest(
        args.checkpoint_root, args.detector_weights, args.artifact
    )
    missing = [name for name, value in manifest["inputs"].items() if not value["exists"]]
    if missing:
        print("Missing optional/required artifact inputs:", *missing, sep="\n  ")
    if args.require_clean and manifest["working_tree_dirty"]:
        raise SystemExit("Git checkout is dirty; commit changes before clean-checkout capture.")
    output = args.output
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote reproducibility manifest: {output}")


if __name__ == "__main__":
    main()
