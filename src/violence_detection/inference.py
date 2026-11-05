"""End-to-end clip prediction with YOLO/ByteTrack, interaction crops, and Swin3D-T."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from violence_detection.crops.crop_builder import (
    CropBoxes,
    build_interaction_crops,
)
from violence_detection.data.augment import prepare_video_batch
from violence_detection.data.preprocess import decode_video, load_preprocess_spec
from violence_detection.detection.detect_track import iter_video_tracks, load_yolo
from violence_detection.models.x3ds import Swin3DViewAveraging


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "model_checkpoints/phase6/swin3d_t_rwf2000_paired_views/best.pt"
)
DEFAULT_THRESHOLD_FILE = (
    PROJECT_ROOT / "experiments/phase7/calibration/swin3d_t_paired_views/threshold.json"
)
DEFAULT_THRESHOLD = 0.94775390625


@dataclass(frozen=True)
class ClipCropOutput:
    """Interaction crops, all frame-level tracks, and source video dimensions."""

    crops: CropBoxes
    selection: Any
    frame_count: int
    frame_width: int
    frame_height: int
    detection_frame_indices: np.ndarray
    detection_boxes_xyxy: np.ndarray
    detection_scores: np.ndarray
    detection_track_ids: np.ndarray


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Return a stable, reasonably bright BGR color for one track ID."""
    hue = (int(track_id) * 37) % 180
    color = cv2.cvtColor(
        np.asarray([[[hue, 210, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]
    return tuple(int(channel) for channel in color)


def _write_annotated_video(
    source: str | Path,
    destination: str | Path,
    tracks: ClipCropOutput,
    *,
    verdict: str,
    violent_probability: float,
    threshold: float,
) -> Path:
    """Write source frames with person boxes, ByteTrack IDs, trails, and verdict."""
    source_path = Path(source)
    output_path = Path(destination)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Annotated video output must not overwrite the source clip")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Annotated video output must use the .mp4 extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise OSError(f"Could not reopen video for annotation: {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (tracks.frame_width, tracks.frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create annotated MP4: {output_path}")

    observations: dict[int, list[int]] = defaultdict(list)
    for observation_index, frame_index in enumerate(
        tracks.detection_frame_indices.tolist()
    ):
        observations[int(frame_index)].append(observation_index)
    history: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=24)
    )
    label_color = (40, 40, 220) if verdict == "VIOLENT" else (40, 170, 40)
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            for observation_index in observations.get(frame_index, []):
                x1, y1, x2, y2 = np.rint(
                    tracks.detection_boxes_xyxy[observation_index]
                ).astype(int)
                track_id = int(tracks.detection_track_ids[observation_index])
                confidence = float(tracks.detection_scores[observation_index])
                color = _track_color(track_id) if track_id >= 0 else (230, 180, 40)
                x1 = min(max(x1, 0), tracks.frame_width - 1)
                x2 = min(max(x2, 0), tracks.frame_width - 1)
                y1 = min(max(y1, 0), tracks.frame_height - 1)
                y2 = min(max(y2, 0), tracks.frame_height - 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                person_label = (
                    f"Person {track_id} {confidence:.2f}"
                    if track_id >= 0
                    else f"Person {confidence:.2f}"
                )
                text_size, baseline = cv2.getTextSize(
                    person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                label_top = max(0, y1 - text_size[1] - baseline - 6)
                cv2.rectangle(
                    frame,
                    (x1, label_top),
                    (min(tracks.frame_width - 1, x1 + text_size[0] + 8), y1),
                    color,
                    thickness=-1,
                )
                cv2.putText(
                    frame,
                    person_label,
                    (x1 + 4, max(text_size[1], y1 - baseline - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if track_id >= 0:
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    history[track_id].append(center)

            for observation_index in observations.get(frame_index, []):
                track_id = int(tracks.detection_track_ids[observation_index])
                points = history.get(track_id)
                if track_id >= 0 and points and len(points) > 1:
                    cv2.polylines(
                        frame,
                        [np.asarray(points, dtype=np.int32)],
                        isClosed=False,
                        color=_track_color(track_id),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

            banner = (
                f"CLIP VERDICT: {verdict}  |  P(violent)={violent_probability:.3f}"
                f"  |  threshold={threshold:.3f}"
            )
            cv2.rectangle(frame, (0, 0), (tracks.frame_width, 42), (20, 20, 20), -1)
            cv2.putText(
                frame,
                banner,
                (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                label_color,
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise OSError(f"Annotated video was not written: {output_path}")
    return output_path


def build_clip_crops(
    clip: str | Path,
    *,
    detector: Any,
    detection_config: str | Path = PROJECT_ROOT / "configs/detection.yaml",
    crop_config: str | Path = PROJECT_ROOT / "configs/interaction_crops.yaml",
    device: str = "auto",
) -> ClipCropOutput:
    """Run person detection/tracking and build clip-stable interaction crops."""
    clip_path = Path(clip)
    if not clip_path.is_file():
        raise FileNotFoundError(clip_path)
    detect = yaml.safe_load(Path(detection_config).read_text(encoding="utf-8"))
    crop = yaml.safe_load(Path(crop_config).read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {clip_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count < 1 or frame_width < 1 or frame_height < 1:
        raise OSError(f"Video has invalid dimensions or no frames: {clip_path}")

    frame_indices: list[int] = []
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    track_ids: list[np.ndarray] = []
    for frame_index, tracks in iter_video_tracks(
        detector,
        clip_path,
        tracker=PROJECT_ROOT / detect["tracker"],
        imgsz=int(detect["imgsz"]),
        conf=float(detect["confidence"]),
        iou=float(detect["iou"]),
        device=device if device != "auto" else str(detect.get("device", "auto")),
        frame_stride=int(detect.get("frame_stride", 1)),
    ):
        if len(tracks.boxes_xyxy):
            frame_indices.extend([frame_index] * len(tracks.boxes_xyxy))
            boxes.extend(tracks.boxes_xyxy)
            scores.extend(tracks.scores)
            track_ids.extend(tracks.track_ids)

    crop_result = build_interaction_crops(
        frame_count,
        np.asarray(frame_indices, dtype=np.int32),
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(track_ids, dtype=np.int32),
        frame_width=frame_width,
        frame_height=frame_height,
        scores=np.asarray(scores, dtype=np.float32),
        tau=float(crop["tau"]),
        min_pair_frames=int(crop["min_pair_frames"]),
        group_tau=float(crop["tau"])
        * float(crop.get("group_tau_multiplier", 1.5)),
        margin=float(crop["margin"]),
        smoothing_window=int(crop["smoothing_window"]),
        max_gap=int(crop["max_gap"]),
    )
    detection_frame_indices = np.asarray(frame_indices, dtype=np.int32)
    detection_boxes_xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    detection_scores = np.asarray(scores, dtype=np.float32)
    detection_track_ids = np.asarray(track_ids, dtype=np.int32)
    return ClipCropOutput(
        crops=crop_result.crops,
        selection=crop_result.selection,
        frame_count=frame_count,
        frame_width=frame_width,
        frame_height=frame_height,
        detection_frame_indices=detection_frame_indices,
        detection_boxes_xyxy=detection_boxes_xyxy,
        detection_scores=detection_scores,
        detection_track_ids=detection_track_ids,
    )


class ViolencePredictor:
    """Load the selected checkpoint once and predict individual video clips."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        detector_weights: str | Path | None = None,
        config: str | Path = PROJECT_ROOT / "configs/train_swin3d_t.yaml",
        preprocess_config: str | Path = PROJECT_ROOT / "configs/preprocess_crops.yaml",
        detection_config: str | Path = PROJECT_ROOT / "configs/detection.yaml",
        crop_config: str | Path = PROJECT_ROOT / "configs/interaction_crops.yaml",
        threshold: float | None = None,
        device: str | None = None,
        cpu_threads: int = 2,
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.checkpoint_path}. "
                "Download the Phase 6 checkpoint or pass checkpoint= explicitly."
            )
        self.config_path = Path(config)
        self.preprocess_config_path = Path(preprocess_config)
        self.detection_config_path = Path(detection_config)
        self.crop_config_path = Path(crop_config)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.preprocess_config = yaml.safe_load(
            self.preprocess_config_path.read_text(encoding="utf-8")
        )
        self.detection_config = yaml.safe_load(
            self.detection_config_path.read_text(encoding="utf-8")
        )
        self.crop_config = yaml.safe_load(
            self.crop_config_path.read_text(encoding="utf-8")
        )
        self.spec = load_preprocess_spec(
            self.preprocess_config_path, str(self.config["model"]["variant"])
        )
        if threshold is None:
            # The calibration artifact lives under ignored experiment outputs;
            # retain the recorded value when loading from a clean Git checkout.
            threshold = DEFAULT_THRESHOLD
            if DEFAULT_THRESHOLD_FILE.is_file():
                threshold = float(
                    json.loads(DEFAULT_THRESHOLD_FILE.read_text(encoding="utf-8"))[
                        "threshold"
                    ]
                )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between zero and one")
        self.threshold = float(threshold)
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {self.device}")
        torch.set_num_threads(max(1, int(cpu_threads)))
        cv2.setNumThreads(1)

        checkpoint_data = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint_data.get("architecture") != "swin3d_t":
            raise ValueError(
                "This inference package currently supports the selected "
                "Swin3D-T paired-view checkpoint only."
            )
        self.model = Swin3DViewAveraging(
            int(self.config["model"].get("num_classes", 2))
        )
        self.model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        self.model.to(self.device).eval()
        if self.device.type == "cuda":
            self.model.to(memory_format=torch.channels_last_3d)
        detector_source: str | Path
        if detector_weights is not None:
            detector_source = detector_weights
        else:
            local_weights = PROJECT_ROOT / str(self.detection_config["weights"])
            # Ultralytics can fetch its named pretrained weight when the local
            # file is absent; Kaggle inference therefore needs Internet enabled.
            detector_source = (
                local_weights
                if local_weights.is_file()
                else str(self.detection_config["weights"])
            )
        self.detector = load_yolo(detector_source)
        self.detection_config["tracker"] = str(
            PROJECT_ROOT / self.detection_config["tracker"]
        )

    @torch.inference_mode()
    def predict(
        self,
        clip: str | Path,
        *,
        output_video: str | Path | None = None,
    ) -> dict[str, Any]:
        """Classify a clip and save an MP4 with person boxes, IDs, and verdict."""
        crop_output = build_clip_crops(
            clip,
            detector=self.detector,
            detection_config=self.detection_config_path,
            crop_config=self.crop_config_path,
            device=str(self.device),
        )
        decoded = decode_video(
            clip,
            self.spec,
            include_crop=True,
            crop_boxes_xyxy=crop_output.crops.boxes_xyxy,
            crop_fallback_mask=crop_output.crops.fallback_mask,
        )
        # Deterministic eval sampling uses the first candidate in each segment.
        sample_indices = np.arange(self.spec.num_frames) * (
            self.spec.cache_frames // self.spec.num_frames
        )
        views = {
            "full": torch.from_numpy(
                np.ascontiguousarray(decoded.full_frames[sample_indices])[None]
            ),
            "crop": torch.from_numpy(
                np.ascontiguousarray(decoded.crop_frames[sample_indices])[None]
            ),
        }
        prepared = prepare_video_batch(
            views,
            self.spec,
            training=False,
            augmentation=self.preprocess_config.get("augmentation", {}),
            device=self.device,
        )
        amp_enabled = self.device.type == "cuda" and bool(
            self.config["training"].get("amp", True)
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = self.model(prepared)
        probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
        violent_probability = float(probabilities[1])
        nonviolent_probability = float(probabilities[0])
        verdict = (
            "VIOLENT"
            if violent_probability >= self.threshold
            else "NONVIOLENT"
        )
        if output_video is None:
            output_video = (
                PROJECT_ROOT
                / "experiments/inference"
                / f"{Path(clip).stem}_annotated.mp4"
            )
        annotated_path = _write_annotated_video(
            clip,
            output_video,
            crop_output,
            verdict=verdict,
            violent_probability=violent_probability,
            threshold=self.threshold,
        )
        selection = crop_output.selection
        unique_track_ids = np.unique(crop_output.detection_track_ids)
        unique_track_ids = unique_track_ids[unique_track_ids >= 0]
        return {
            "clip": str(clip),
            "prediction": verdict.lower(),
            "final_verdict": "Violent" if verdict == "VIOLENT" else "NonViolent",
            "violent_probability": violent_probability,
            "nonviolent_probability": nonviolent_probability,
            "verdict_confidence": (
                violent_probability
                if verdict == "VIOLENT"
                else nonviolent_probability
            ),
            "decision_threshold_violent": self.threshold,
            "annotated_video": str(annotated_path),
            "person_detection_count": int(len(crop_output.detection_track_ids)),
            "unique_tracked_person_count": int(len(unique_track_ids)),
            "tracked_person_ids": [int(track_id) for track_id in unique_track_ids],
            "frames_with_person_detections": int(
                len(np.unique(crop_output.detection_frame_indices))
            ),
            "crop_fallback_rate": crop_output.crops.fallback_rate,
            "selected_pair_track_ids": list(selection.pair_track_ids)
            if selection
            else [],
            "selected_group_track_ids": list(selection.group_track_ids)
            if selection
            else [],
            "pair_support_fraction": selection.support_fraction
            if selection
            else 0.0,
            "sampled_crop_found_rate": float(
                np.mean(decoded.crop_found_mask[sample_indices])
            ),
            "device": str(self.device),
            "model": "swin3d_t_paired_views",
        }


def predict(
    clip: str | Path,
    *,
    predictor: ViolencePredictor | None = None,
    **predictor_kwargs: Any,
) -> dict[str, Any]:
    """Convenience API; pass a predictor to reuse loaded weights across clips."""
    instance = predictor or ViolencePredictor(**predictor_kwargs)
    return instance.predict(clip)
