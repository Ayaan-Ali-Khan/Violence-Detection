"""CLI entry point for building the resumable video tensor cache."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from violence_detection.data.cache_tensors import main  # noqa: E402

if __name__ == "__main__":
    main()
