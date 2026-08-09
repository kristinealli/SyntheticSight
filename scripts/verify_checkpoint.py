#!/usr/bin/env python3
"""Verify that a checkpoint matches the final Synthetic Sight run contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

EXPECTED = {
    "checkpoint_version": 2,
    "epoch": 13,
    "decision_threshold": 0.51,
    "monitor_metric": "Fake F1",
    "image_size": [224, 224],
    "label_convention": {"Real": 0, "Fake": 1},
}
EXPECTED_HEAD = {
    "hidden_units": 256,
    "batch_normalization": True,
    "dropout": 0.30,
    "output_features": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    errors: list[str] = []

    for key, expected in EXPECTED.items():
        actual = checkpoint.get(key)
        if key == "image_size" and isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, received {actual!r}")

    head = checkpoint.get("head_configuration") or {}
    for key, expected in EXPECTED_HEAD.items():
        if head.get(key) != expected:
            errors.append(
                f"head_configuration.{key}: expected {expected!r}, "
                f"received {head.get(key)!r}"
            )

    report = {
        "checkpoint": str(args.checkpoint),
        "sha256": sha256(args.checkpoint),
        "valid_final_artifact": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
