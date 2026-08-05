from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

# Exact checkpoint used for the presentation results.
EXPECTED_SHA256 = (
    "d9a7fd6a692c942b550f9848500dc3ff"
    "b10d5809cb0d0091990648bf369ad21c"
)
EXPECTED_EPOCH = 13
EXPECTED_THRESHOLD = 0.51
EXPECTED_CLASS_NAMES = ["Real", "Fake"]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}\n"
            "Copy the final epoch-13 best_resnet50.pth into the models folder."
        )

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError):
        # Use only for a checkpoint created by your own trusted notebook.
        return torch.load(path, map_location="cpu", weights_only=False)


def validate_presentation_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> str:
    actual_hash = file_sha256(checkpoint_path)
    problems: list[str] = []

    if actual_hash != EXPECTED_SHA256:
        problems.append(
            "SHA-256 does not match the presentation checkpoint.\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Found:    {actual_hash}"
        )

    if int(checkpoint.get("epoch", -1)) != EXPECTED_EPOCH:
        problems.append(
            f"Expected epoch {EXPECTED_EPOCH}, found "
            f"{checkpoint.get('epoch', 'missing')}."
        )

    threshold = float(checkpoint.get("decision_threshold", -1.0))
    if abs(threshold - EXPECTED_THRESHOLD) > 1e-9:
        problems.append(
            f"Expected decision threshold {EXPECTED_THRESHOLD:.2f}, "
            f"found {threshold}."
        )

    class_names = list(checkpoint.get("class_names", []))
    if class_names != EXPECTED_CLASS_NAMES:
        problems.append(
            f"Expected class names {EXPECTED_CLASS_NAMES}, found {class_names}."
        )

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        problems.append("model_state_dict is missing from the checkpoint.")
    else:
        # The final presentation head is:
        # Linear -> BatchNorm1d -> ReLU -> Dropout(0.30) -> Linear
        required_head_keys = {
            "fc.0.weight",
            "fc.0.bias",
            "fc.1.weight",
            "fc.1.bias",
            "fc.1.running_mean",
            "fc.1.running_var",
            "fc.4.weight",
            "fc.4.bias",
        }
        missing = sorted(required_head_keys.difference(state_dict.keys()))
        if missing:
            problems.append(
                "Checkpoint does not contain the presentation classifier head. "
                f"Missing keys: {missing}"
            )

    if problems:
        raise RuntimeError(
            "The app is not using the model from the slide presentation:\n\n"
            + "\n\n".join(problems)
        )

    return actual_hash


def build_presentation_model(checkpoint: dict[str, Any]) -> nn.Module:
    head_config = checkpoint.get("head_configuration", {})
    hidden_units = int(head_config.get("hidden_units", 256))
    dropout = float(head_config.get("dropout", 0.30))

    if head_config and not bool(head_config.get("batch_normalization", False)):
        raise RuntimeError(
            "The checkpoint metadata says BatchNorm is disabled, but the "
            "presentation model requires BatchNorm1d in the classifier head."
        )

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden_units),
        nn.BatchNorm1d(hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


class DeepfakeDetector:
    def __init__(self, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = load_trusted_checkpoint(self.checkpoint_path)
        self.checkpoint_sha256 = validate_presentation_checkpoint(
            self.checkpoint_path,
            self.checkpoint,
        )

        self.model = build_presentation_model(self.checkpoint)
        self.class_names = list(self.checkpoint["class_names"])
        self.decision_threshold = float(
            self.checkpoint["decision_threshold"]
        )
        self.epoch = int(self.checkpoint["epoch"])
        self.monitor_metric = str(self.checkpoint.get("monitor_metric", ""))
        self.monitor_value = float(self.checkpoint.get("monitor_value", 0.0))

        image_size = tuple(self.checkpoint.get("image_size", (224, 224)))
        normalize_mean = self.checkpoint.get(
            "normalize_mean", [0.485, 0.456, 0.406]
        )
        normalize_std = self.checkpoint.get(
            "normalize_std", [0.229, 0.224, 0.225]
        )

        # This exactly matches the notebook's validation/test preprocessing.
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=normalize_mean,
                    std=normalize_std,
                ),
            ]
        )

    def predict(self, image: Image.Image) -> dict[str, float | str | int]:
        if image is None:
            raise ValueError("No image was provided.")

        image = ImageOps.exif_transpose(image).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.inference_mode():
            logit = self.model(input_tensor).reshape(-1)[0]
            fake_probability = float(torch.sigmoid(logit).item())

        real_probability = 1.0 - fake_probability
        prediction = (
            self.class_names[1]
            if fake_probability >= self.decision_threshold
            else self.class_names[0]
        )
        confidence = (
            fake_probability
            if prediction == self.class_names[1]
            else real_probability
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "decision_threshold": self.decision_threshold,
            "epoch": self.epoch,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
