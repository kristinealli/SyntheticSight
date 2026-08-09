"""Model construction for the final Synthetic Sight ResNet-50 classifier."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .config import HEAD_DROPOUT, HEAD_HIDDEN_UNITS


def build_resnet50_binary(
    *,
    hidden_units: int = HEAD_HIDDEN_UNITS,
    dropout: float = HEAD_DROPOUT,
) -> nn.Module:
    """Build the exact binary ResNet-50 architecture used by the final run.

    The backbone is instantiated without downloading ImageNet weights because
    inference immediately loads a complete project checkpoint. The training
    notebook uses ImageNet pretrained weights before fine-tuning.
    """
    model = models.resnet50(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.BatchNorm1d(hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, 1),
    )
    return model
