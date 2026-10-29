"""Exercise the Kaggle checkout, package, seed, tracking, and output path."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from pathlib import Path

import torch
import yaml

from violence_detection.utils.logger import init_run
from violence_detection.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {gpu_name}")

    output_dir = Path(
        config.get("output_dir")
        or os.environ.get("VIOLENCE_OUTPUT_DIR", "outputs/smoke-test")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = init_run(
        config={**config, "seed": seed, "device": str(device), "gpu": gpu_name},
        name=str(config.get("name", "smoke-test")),
        tags=config.get("tags", []),
        project=str(config.get("project", "violence-detection")),
        backend=str(config.get("tracking_backend", "wandb")),
        output_dir=output_dir,
    )

    try:
        for step in range(5):
            tracker.log({"smoke/fake_loss": 1.0 / (step + 1)}, step=step)
        result = {
            "status": "ok",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu_name,
            "tracking_backend": tracker.backend,
            "seed": seed,
        }
        result_path = output_dir / "smoke_test_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote smoke-test result to {result_path}")
        print(json.dumps(result, indent=2))
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
