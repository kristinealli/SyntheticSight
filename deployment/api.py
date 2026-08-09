"""FastAPI interface for Synthetic Sight inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from synthetic_sight.inference import CheckpointCompatibilityError, SyntheticSightDetector

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "SYNTHETIC_SIGHT_MODEL_PATH",
        REPOSITORY_ROOT / "models" / "best_resnet50.pth",
    )
)
MAX_UPLOAD_BYTES = int(os.environ.get("SYNTHETIC_SIGHT_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="Synthetic Sight API",
    version="1.0.0",
    description=(
        "Research-prototype inference API for a ResNet-50 classifier trained to "
        "separate FFHQ real faces from StyleGAN-generated faces. Predictions are "
        "review signals, not proof of authenticity."
    ),
)

allowed_origins = [
    origin.strip()
    for origin in os.environ.get("SYNTHETIC_SIGHT_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


class PredictionResponse(BaseModel):
    label: str
    synthetic_probability: float
    real_probability: float
    decision_threshold: float
    caveat: str


@lru_cache(maxsize=1)
def get_detector() -> SyntheticSightDetector:
    """Load and validate the model once per API process."""
    return SyntheticSightDetector(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report service health without forcing model loading."""
    return {"status": "ok", "model_artifact_present": MODEL_PATH.exists()}


@app.get("/model")
def model_info() -> dict[str, str | int | float]:
    """Return metadata for the loaded model used by this process."""
    try:
        detector = get_detector()
    except (FileNotFoundError, CheckpointCompatibilityError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return {
        "architecture": "ResNet-50",
        "selected_epoch": detector.epoch,
        "monitor_metric": detector.monitor_metric,
        "decision_threshold": detector.decision_threshold,
        "scope": "FFHQ real faces vs StyleGAN-generated still faces",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify one uploaded face image."""
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Supported image types: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}",
        )

    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    try:
        result = get_detector().predict_bytes(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except CheckpointCompatibilityError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return PredictionResponse(
        label="Synthetic" if result.label == "Fake" else "Real",
        synthetic_probability=result.fake_probability,
        real_probability=result.real_probability,
        decision_threshold=result.decision_threshold,
        caveat="Review signal only; this prediction does not authenticate image provenance.",
    )
