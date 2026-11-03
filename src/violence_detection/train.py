"""Distributed mixed-precision video-classifier training for Kaggle GPUs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from violence_detection.data.augment import prepare_video_batch
from violence_detection.data.dataset import CachedVideoDataset
from violence_detection.data.preprocess import load_preprocess_spec
from violence_detection.models.x3ds import (
    LateFusionX3D,
    Swin3DViewAveraging,
    build_x3d,
    fusion_parameter_groups,
    initialize_fusion_streams,
    initialize_pretrained_backbone,
    initialize_pretrained_fusion,
    initialize_pretrained_swin3d_t,
    split_parameter_groups,
    swin3d_parameter_groups,
)
from violence_detection.utils.logger import init_run
from violence_detection.utils.seed import make_generator, seed_worker, set_seed

os.environ.setdefault("MPLBACKEND", "Agg")


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "loss": float("nan"),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": auc,
    }


def _forward(model: torch.nn.Module, videos: Any) -> torch.Tensor:
    if isinstance(videos, dict):
        unwrapped = model.module if hasattr(model, "module") else model
        if isinstance(unwrapped, (LateFusionX3D, Swin3DViewAveraging)):
            return model(videos)
        batch = videos["full"].shape[0]
        combined = torch.cat([videos["full"], videos["crop"]], dim=0)
        logits = model(combined)
        return (logits[:batch] + logits[batch:]) / 2.0
    return model(videos)


def _loader(
    config: dict[str, Any],
    *,
    split: str,
    training: bool,
    rank: int,
    world_size: int,
    distributed: bool,
    cache_root: Path,
    input_tag: str,
    mode: str,
    workers: int,
    seed: int,
) -> tuple[CachedVideoDataset, DataLoader, DistributedSampler | None]:
    data_config = config["data"]
    spec_config = config["model"]
    dataset = CachedVideoDataset(
        cache_root,
        input_tag,
        split,
        model_name=spec_config["variant"],
        preprocess_config=data_config["preprocess_config"],
        mode=mode,
        training=training,
        manifest_path=data_config["manifest"],
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            seed=seed,
            drop_last=training,
        )
        if distributed
        else None
    )
    train_config = config["training"]
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(train_config["per_gpu_batch_size"]),
        "shuffle": training and sampler is None,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "drop_last": training,
        "generator": make_generator(seed + rank),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(train_config["prefetch_factor"])
    return dataset, DataLoader(**kwargs), sampler


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    sampler: DistributedSampler | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: torch.nn.Module,
    spec,
    augmentation: dict[str, float],
    gradient_clip_norm: float,
    *,
    epoch: int,
    device: torch.device,
    rank: int,
    world_size: int,
    amp: bool,
    log_every: int,
) -> dict[str, float]:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    started = time.perf_counter()
    for step, (videos, labels) in enumerate(loader, start=1):
        labels = labels.to(device, non_blocking=True)
        inputs = prepare_video_batch(
            videos,
            spec,
            training=True,
            augmentation=augmentation,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = _forward(model, inputs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.size(0)
        loss_sum += loss.detach().to(torch.float64) * batch_size
        correct_sum += (logits.detach().argmax(dim=1) == labels).sum()
        sample_count += batch_size
        if rank == 0 and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(
                f"epoch={epoch + 1} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"elapsed={(time.perf_counter() - started) / 60:.1f}m",
                flush=True,
            )
    if world_size > 1:
        dist.all_reduce(loss_sum)
        dist.all_reduce(correct_sum)
        dist.all_reduce(sample_count)
    total = max(float(sample_count.item()), 1.0)
    return {
        "loss": float((loss_sum / total).item()),
        "accuracy": float((correct_sum / total).item()),
        "samples": total,
    }


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    spec,
    *,
    device: torch.device,
    amp: bool,
    augmentation: dict[str, float],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    sample_count = 0
    for videos, labels in loader:
        labels = labels.to(device, non_blocking=True)
        inputs = prepare_video_batch(
            videos,
            spec,
            training=False,
            augmentation=augmentation,
            device=device,
        )
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = _forward(model, inputs)
            loss = criterion(logits, labels)
        probabilities = logits.softmax(dim=1)[:, 1]
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.float().cpu().numpy())
        loss_sum += float(loss.item()) * labels.size(0)
        sample_count += labels.size(0)
    labels_array = np.concatenate(all_labels)
    probability_array = np.concatenate(all_probabilities)
    summary = _metrics(labels_array, probability_array)
    summary["loss"] = loss_sum / max(1, sample_count)
    return summary, labels_array, probability_array


def _atomic_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(history)
    frame.to_csv(output_dir / "history.csv", index=False)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["train_accuracy"], label="Train")
    axes[1].plot(frame["epoch"], frame["val_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_x3d.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/x3d_s_rwf2000"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--mode", choices=["full", "crop", "both"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--init-full-checkpoint", type=Path)
    parser.add_argument("--init-crop-checkpoint", type=Path)
    parser.add_argument(
        "--allow-cpu", action="store_true", help="Allow a CPU smoke run"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.init_full_checkpoint or args.init_crop_checkpoint:
        initialization = config["model"].setdefault("initialization", {})
        if args.init_full_checkpoint:
            initialization["full_checkpoint"] = str(args.init_full_checkpoint)
        if args.init_crop_checkpoint:
            initialization["crop_checkpoint"] = str(args.init_crop_checkpoint)
    config["training"]["epochs"] = args.epochs or int(config["training"]["epochs"])
    mode = args.mode or config["data"]["mode"]
    cache_root = args.cache_dir or Path(config["data"]["cache_dir"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError("Training requires a CUDA GPU. Run the Kaggle GPU runner.")
    distributed = world_size > 1
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    actual_world_size = dist.get_world_size() if distributed else 1

    training_config = config["training"]
    cpu_threads = int(training_config["cpu_threads_per_rank"])
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = use_cuda
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
    seed = int(config["seed"])
    set_seed(seed + rank, deterministic=bool(config.get("deterministic", False)))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        print(
            f"Training on {actual_world_size} process(es); "
            f"device={device}; GPU(s)="
            + ", ".join(
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            )
            if use_cuda
            else "Training on CPU",
            flush=True,
        )

    model_config = config["model"]
    data_config = config["data"]
    spec = load_preprocess_spec(
        data_config["preprocess_config"], model_config["variant"]
    )
    input_tag = str(data_config.get("input_tag", spec.input_tag))
    if input_tag != spec.input_tag:
        raise ValueError("train config input_tag does not match preprocessing profile")
    train_dataset, train_loader, train_sampler = _loader(
        config,
        split="train",
        training=True,
        rank=rank,
        world_size=actual_world_size,
        distributed=distributed,
        cache_root=cache_root,
        input_tag=input_tag,
        mode=mode,
        workers=int(training_config["workers_per_rank"]),
        seed=seed,
    )
    del train_dataset
    val_dataset = val_loader = None
    if rank == 0:
        val_dataset, val_loader, _ = _loader(
            config,
            split="validation",
            training=False,
            rank=rank,
            world_size=actual_world_size,
            distributed=False,
            cache_root=cache_root,
            input_tag=input_tag,
            mode=mode,
            workers=int(training_config["workers_per_rank"]),
            seed=seed,
        )

    architecture = str(model_config.get("architecture", "x3d"))
    variant = str(model_config["variant"])
    num_classes = int(model_config.get("num_classes", 2))
    if architecture == "late_fusion":
        base_model = LateFusionX3D(variant, num_classes)
        initialization = model_config.get("initialization", {})
        if initialization.get("full_checkpoint") or initialization.get(
            "crop_checkpoint"
        ):
            if not (
                initialization.get("full_checkpoint")
                and initialization.get("crop_checkpoint")
            ):
                raise ValueError(
                    "Late-fusion checkpoint initialization requires both "
                    "full_checkpoint and crop_checkpoint."
                )
            if rank == 0:
                initialize_fusion_streams(
                    base_model,
                    str(initialization["full_checkpoint"]),
                    str(initialization["crop_checkpoint"]),
                )
                print("Initialized late-fusion streams from specialist checkpoints.")
        elif bool(model_config.get("pretrained", True)) and rank == 0:
            missing = initialize_pretrained_fusion(base_model)
            print(
                "Loaded Kinetics-pretrained X3D streams; initialized class heads: "
                f"{missing}"
            )
        optimizer_groups = fusion_parameter_groups(
            base_model,
            float(training_config["learning_rate_backbone"]),
            float(training_config["learning_rate_head"]),
        )
    elif architecture == "x3d":
        base_model = build_x3d(variant, num_classes)
        if bool(model_config.get("pretrained", True)) and rank == 0:
            missing = initialize_pretrained_backbone(base_model, variant)
            print(
                f"Loaded Kinetics-pretrained backbone; initialized new head: {missing}"
            )
        optimizer_groups = split_parameter_groups(
            base_model,
            float(training_config["learning_rate_backbone"]),
            float(training_config["learning_rate_head"]),
        )
    elif architecture == "swin3d_t":
        if variant != "swin3d_t":
            raise ValueError("Swin3D-T requires model.variant='swin3d_t'.")
        base_model = Swin3DViewAveraging(num_classes)
        if bool(model_config.get("pretrained", True)) and rank == 0:
            missing = initialize_pretrained_swin3d_t(base_model)
            print(
                f"Loaded Kinetics-pretrained Swin3D-T; initialized new head: {missing}"
            )
        optimizer_groups = swin3d_parameter_groups(
            base_model,
            float(training_config["learning_rate_backbone"]),
            float(training_config["learning_rate_head"]),
        )
    else:
        raise ValueError(f"Unsupported model architecture: {architecture!r}")
    base_model.to(device)
    if use_cuda:
        base_model.to(memory_format=torch.channels_last_3d)
    model: torch.nn.Module = (
        DistributedDataParallel(
            base_model,
            device_ids=[local_rank] if use_cuda else None,
            output_device=local_rank if use_cuda else None,
            # Rank 0 initializes pretrained weights/checkpoints. Synchronize the
            # BatchNorm running statistics on each DDP forward as well.
            broadcast_buffers=True,
        )
        if distributed
        else base_model
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups, weight_decay=float(training_config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )
    amp = bool(training_config["amp"] and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = torch.nn.CrossEntropyLoss()
    preprocess_config = yaml.safe_load(
        Path(data_config["preprocess_config"]).read_text(encoding="utf-8")
    )
    augmentation = preprocess_config.get("augmentation", {})
    run_logger = None
    if rank == 0:
        tracking = config.get("tracking", {})
        run_logger = init_run(
            {
                **config,
                "world_size": actual_world_size,
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ]
                if use_cuda
                else [],
                "global_batch_size": int(training_config["per_gpu_batch_size"])
                * actual_world_size,
            },
            name=str(tracking.get("run_name", "x3d-rwf2000-baseline")),
            tags=list(tracking.get("tags", [])),
            project=str(tracking.get("project", "violence-detection")),
            backend=str(tracking.get("backend", "wandb")),
            output_dir=output_dir,
        )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = int(training_config["patience"])
    history: list[dict[str, Any]] = []
    epochs = int(training_config["epochs"])
    try:
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            train_metrics = _train_epoch(
                model,
                train_loader,
                train_sampler,
                optimizer,
                scaler,
                criterion,
                spec,
                augmentation,
                float(training_config["gradient_clip_norm"]),
                epoch=epoch,
                device=device,
                rank=rank,
                world_size=actual_world_size,
                amp=amp,
                log_every=int(training_config["log_every_steps"]),
            )
            val_metrics = None
            if rank == 0:
                val_metrics, _, _ = _evaluate(
                    base_model,
                    val_loader,
                    criterion,
                    spec,
                    device=device,
                    amp=amp,
                    augmentation=augmentation,
                )
            values = torch.tensor(
                [
                    val_metrics[key] if rank == 0 else 0.0
                    for key in ("loss", "accuracy", "precision", "recall", "f1", "auc")
                ],
                dtype=torch.float64,
                device=device,
            )
            if distributed:
                dist.broadcast(values, src=0)
            val_values = values.cpu().tolist()
            val_metrics = dict(
                zip(
                    ("loss", "accuracy", "precision", "recall", "f1", "auc"),
                    val_values,
                    strict=True,
                )
            )
            scheduler.step(val_metrics["loss"])
            record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "train_val_accuracy_gap": train_metrics["accuracy"]
                - val_metrics["accuracy"],
                "learning_rate_backbone": optimizer.param_groups[0]["lr"],
                "learning_rate_head": optimizer.param_groups[1]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            history.append(record)
            stop = False
            if rank == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_variant": model_config["variant"],
                    "architecture": architecture,
                    "input_tag": input_tag,
                    "mode": mode,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                }
                _atomic_save(checkpoint, output_dir / "last.pt")
                improved = val_metrics["loss"] < best_val_loss
                if improved:
                    best_val_loss = val_metrics["loss"]
                    epochs_without_improvement = 0
                    _atomic_save(checkpoint, output_dir / "best.pt")
                else:
                    epochs_without_improvement += 1
                _save_history(history, output_dir)
                print(
                    f"epoch={epoch + 1}/{epochs} "
                    f"train_acc={record['train_accuracy']:.4f} "
                    f"val_acc={record['val_accuracy']:.4f} "
                    f"val_loss={record['val_loss']:.4f} "
                    f"gap={record['train_val_accuracy_gap']:+.4f}",
                    flush=True,
                )
                run_logger.log(
                    {
                        "epoch": epoch + 1,
                        **{
                            f"train/{key}": value
                            for key, value in train_metrics.items()
                        },
                        **{f"val/{key}": value for key, value in val_metrics.items()},
                        "train_val/accuracy_gap": record["train_val_accuracy_gap"],
                        "lr/backbone": record["learning_rate_backbone"],
                        "lr/head": record["learning_rate_head"],
                        "system/epoch_seconds": record["epoch_seconds"],
                    },
                    step=epoch + 1,
                )
                stop = epochs_without_improvement >= patience
            stop_tensor = torch.tensor([int(stop)], dtype=torch.int32, device=device)
            if distributed:
                dist.broadcast(stop_tensor, src=0)
            if bool(stop_tensor.item()):
                if rank == 0:
                    print(f"Early stopping after epoch {epoch + 1}.", flush=True)
                break

        if rank == 0:
            best_checkpoint = torch.load(
                output_dir / "best.pt", map_location=device, weights_only=False
            )
            base_model.load_state_dict(best_checkpoint["model_state_dict"])
            train_eval = _loader(
                config,
                split="train",
                training=False,
                rank=rank,
                world_size=actual_world_size,
                distributed=False,
                cache_root=cache_root,
                input_tag=input_tag,
                mode=mode,
                workers=int(training_config["workers_per_rank"]),
                seed=seed,
            )[1]
            train_metrics, _, _ = _evaluate(
                base_model,
                train_eval,
                criterion,
                spec,
                device=device,
                amp=amp,
                augmentation=augmentation,
            )
            val_metrics, _, _ = _evaluate(
                base_model,
                val_loader,
                criterion,
                spec,
                device=device,
                amp=amp,
                augmentation=augmentation,
            )
            final_metrics = {
                "best_epoch": best_checkpoint["epoch"],
                "train": train_metrics,
                "validation": val_metrics,
                "train_validation_accuracy_gap": train_metrics["accuracy"]
                - val_metrics["accuracy"],
                "train_validation_loss_gap": train_metrics["loss"]
                - val_metrics["loss"],
                "training_clips": len(train_loader.dataset),
                "validation_clips": len(val_loader.dataset),
                "test_evaluation": "not run by training script",
            }
            (output_dir / "train_val_gap.json").write_text(
                json.dumps(final_metrics, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            run_logger.log(
                {
                    "final/train_accuracy": train_metrics["accuracy"],
                    "final/val_accuracy": val_metrics["accuracy"],
                    "final/train_val_accuracy_gap": final_metrics[
                        "train_validation_accuracy_gap"
                    ],
                },
                step=len(history) + 1,
            )
            print(json.dumps(final_metrics, indent=2, allow_nan=False), flush=True)
    finally:
        if run_logger is not None:
            run_logger.finish()
        if distributed:
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
