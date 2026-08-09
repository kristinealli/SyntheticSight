"""Shared model and preprocessing constants for Synthetic Sight."""

from __future__ import annotations

from pathlib import Path

REAL_LABEL = 0
FAKE_LABEL = 1
CLASS_NAMES = ("Real", "Fake")

IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HEAD_HIDDEN_UNITS = 256
HEAD_DROPOUT = 0.30
FINAL_DECISION_THRESHOLD = 0.51
FINAL_SELECTED_EPOCH = 13


def repository_root() -> Path:
    """Return the repository root when running from a source checkout."""
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    """Return the conventional final-checkpoint path in this repository."""
    return repository_root() / "models" / "best_resnet50.pth"
