"""Validated PyTorch inference for Synthetic Sight.

The deployment code reads preprocessing, threshold, and label metadata from the
checkpoint so notebook and application behavior cannot silently drift apart.
"""

from __future__ import annotations

import io
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import (
    CLASS_NAMES,
    FAKE_LABEL,
    HEAD_DROPOUT,
    HEAD_HIDDEN_UNITS,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    REAL_LABEL,
)
from .model import build_resnet50_binary


class CheckpointCompatibilityError(RuntimeError):
    """Raised when a checkpoint does not match the final model contract."""


@dataclass(frozen=True)
class Prediction:
    """One model prediction with the decision information kept explicit."""

    label: str
    fake_probability: float
    real_probability: float
    decision_threshold: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a JSON-friendly representation."""
        return asdict(self)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a trusted project checkpoint with PyTorch's safer mode when available."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}. "
            "See models/README.md for the required final artifact."
        )

    try:
        checkpoint = torch.load(path, map_location=device,
                                weights_only=True, mmap=True,)
    except TypeError:  # older PyTorch
        checkpoint = torch.load(path, map_location=device)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        raise CheckpointCompatibilityError(
            "The checkpoint could not be loaded in safe tensor-only mode. "
            "Only use a checkpoint produced by the final project notebook."
        ) from exc

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise CheckpointCompatibilityError(
            "Expected a checkpoint dictionary containing 'model_state_dict'."
        )
    return checkpoint


def _validate_checkpoint_contract(checkpoint: dict[str, Any]) -> None:
    """Reject legacy artifacts that can load but do not represent the final model."""
    head = checkpoint.get("head_configuration")
    if not isinstance(head, dict):
        raise CheckpointCompatibilityError(
            "Checkpoint has no final head_configuration metadata. "
            "The original repository contained an older epoch-9 artifact with "
            "a different classifier head; it is intentionally unsupported."
        )

    expected_head = {
        "hidden_units": HEAD_HIDDEN_UNITS,
        "batch_normalization": True,
        "dropout": HEAD_DROPOUT,
        "output_features": 1,
    }
    for key, expected in expected_head.items():
        actual = head.get(key)
        if actual != expected:
            raise CheckpointCompatibilityError(
                f"Checkpoint head mismatch for {key!r}: expected {expected!r}, "
                f"received {actual!r}."
            )

    labels = checkpoint.get("label_convention", {})
    if labels.get("Real") != REAL_LABEL or labels.get("Fake") != FAKE_LABEL:
        raise CheckpointCompatibilityError(
            "Checkpoint label mapping must be Real=0 and Fake=1."
        )

    image_size = tuple(checkpoint.get("image_size", ()))
    if image_size != IMAGE_SIZE:
        raise CheckpointCompatibilityError(
            f"Checkpoint image_size must be {IMAGE_SIZE}; received {image_size}."
        )


def _build_transform(checkpoint: dict[str, Any]) -> transforms.Compose:
    """Build deterministic evaluation preprocessing from checkpoint metadata."""
    image_size = tuple(checkpoint.get("image_size", IMAGE_SIZE))
    mean = tuple(checkpoint.get("normalize_mean", IMAGENET_MEAN))
    std = tuple(checkpoint.get("normalize_std", IMAGENET_STD))
    return transforms.Compose(
        [
            transforms.Resize(
                image_size, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class SyntheticSightDetector:
    """Load the final checkpoint once and perform deterministic image inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = _load_checkpoint(self.model_path, self.device)
        _validate_checkpoint_contract(checkpoint)

        head = checkpoint["head_configuration"]

        # Build the model structure without allocating parameter storage first.
        # The checkpoint tensors are assigned directly into the model below.
        with torch.device("meta"):
            self.model = build_resnet50_binary(
                hidden_units=int(head["hidden_units"]),
                dropout=float(head["dropout"]),
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=True, assign=True,)
        self.model.to(self.device)
        self.model.eval()

        self.transform = _build_transform(checkpoint)
        self.class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
        self.decision_threshold = float(checkpoint["decision_threshold"])
        self.epoch = int(checkpoint.get("epoch", -1))
        self.monitor_metric = str(checkpoint.get("monitor_metric", "unknown"))

        if not 0.0 <= self.decision_threshold <= 1.0:
            raise CheckpointCompatibilityError(
                "Checkpoint decision_threshold must be between 0 and 1."
            )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Apply EXIF correction, RGB conversion, resize, tensor conversion, and normalization."""
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        return self.transform(corrected).unsqueeze(0)

    def predict(self, image: Image.Image) -> Prediction:
        """Classify one image and return both class scores plus the threshold."""
        tensor = self.preprocess(image).to(self.device)
        with torch.inference_mode():
            logit = self.model(tensor).reshape(-1)
            if logit.numel() != 1:
                raise RuntimeError(
                    f"Expected one binary logit, received shape {tuple(logit.shape)}."
                )
            fake_probability = float(torch.sigmoid(logit[0]).item())

        real_probability = 1.0 - fake_probability
        label = (
            self.class_names[FAKE_LABEL]
            if fake_probability >= self.decision_threshold
            else self.class_names[REAL_LABEL]
        )
        return Prediction(
            label=str(label),
            fake_probability=fake_probability,
            real_probability=real_probability,
            decision_threshold=self.decision_threshold,
        )

    def predict_bytes(self, image_bytes: bytes) -> Prediction:
        """Decode uploaded bytes and classify one image."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                return self.predict(image)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "The uploaded file is not a readable image.") from exc
