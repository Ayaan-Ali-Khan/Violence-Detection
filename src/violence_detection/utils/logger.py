"""Small experiment logger with W&B, TensorBoard, and no-op backends."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def _git_commit() -> str | None:
    """Return the current Git revision when the code is in a checkout."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
            )
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


class ExperimentLogger:
    """Uniform ``log``/``finish`` interface for supported tracking backends."""

    def __init__(self, backend: str, run: Any = None) -> None:
        self.backend = backend
        self.run = run

    def log(self, metrics_dict: dict[str, Any], step: int | None = None) -> None:
        if self.backend == "wandb":
            self.run.log(metrics_dict, step=step)
        elif self.backend == "tensorboard":
            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    self.run.add_scalar(key, value, global_step=step)

    def finish(self) -> None:
        if self.backend == "wandb":
            self.run.finish()
        elif self.backend == "tensorboard":
            self.run.flush()
            self.run.close()


def init_run(
    config: dict[str, Any],
    name: str,
    tags: list[str] | None = None,
    *,
    project: str = "violence-detection",
    backend: str = "wandb",
    output_dir: str | Path = "runs",
) -> ExperimentLogger:
    """Start W&B, falling back to TensorBoard or a no-op logger.

    The W&B API key is read from ``WANDB_API_KEY`` by the W&B SDK and should
    be supplied through the runtime's secret manager, never source code.
    """
    run_config = dict(config)
    run_config.setdefault("git_commit", _git_commit())
    run_config.setdefault("seed", config.get("seed", 42))

    if backend == "wandb" and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            run = wandb.init(
                project=project,
                name=name,
                tags=tags or [],
                config=run_config,
                reinit=True,
            )
            return ExperimentLogger("wandb", run)
        except Exception as exc:  # Network/auth problems must not stop training.
            LOGGER.warning("W&B initialization failed; trying TensorBoard: %s", exc)
    elif backend == "wandb":
        LOGGER.info("WANDB_API_KEY is not set; trying TensorBoard fallback.")

    if backend in {"wandb", "tensorboard"}:
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(output_dir) / "tensorboard" / name
            writer = SummaryWriter(log_dir=str(log_dir))
            writer.add_text("config", str(run_config))
            return ExperimentLogger("tensorboard", writer)
        except Exception as exc:  # TensorBoard is optional.
            LOGGER.warning("TensorBoard initialization failed; using no-op logger: %s", exc)

    return ExperimentLogger("noop")
