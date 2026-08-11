# Project History

## Overview

**Synthetic Sight** began as an AI4ALL Ignite team project exploring whether machine-learning models could distinguish real human faces from AI-generated faces. The project moved through several modeling, analysis, and deployment approaches before converging on the final **ResNet-50 transfer-learning pipeline** presented by the team.

The original collaborative repository is available at [Shloka-16/deepfake-detection](https://github.com/Shloka-16/deepfake-detection).

This repository is a consolidated, portfolio-ready version of that work. It preserves the final implementation, verified model artifact, evaluation results, reproducibility information, and relevant development history while removing duplicate or superseded code paths.

---

## Project Evolution

### 1. Exploratory Modeling

Early development compared several approaches to synthetic-face classification, including:

- Random Forest baselines
- Custom convolutional neural networks (CNNs)
- Frequency-domain / FFT experiments
- PCA-based exploratory analysis
- Dataset and representation analysis
- Multiple deployment prototypes

These experiments helped the team understand the problem space and compare possible modeling directions. They are part of the project's development history but are not presented as the final production path.

### 2. Transition to ResNet-50

The project ultimately adopted **ResNet-50 pretrained on ImageNet** as the final backbone.

Training used a two-stage transfer-learning strategy:

1. **Epochs 1–3:** train the new binary classifier head while the pretrained backbone remains frozen.
2. **Epochs 4–15:** unfreeze `layer4` and fine-tune the final ResNet stage while earlier backbone layers remain frozen.

Training-only augmentation was limited to horizontal flipping and light color jitter. Rotation was explored earlier and removed from the final pipeline.

The final model artifact is the **epoch 13 checkpoint** with a decision threshold of `0.51`.

For the complete architecture and optimization settings, see [Architecture](architecture.md).

### 3. Final Evaluation and Deployment

After model-development decisions were completed using validation data, the final checkpoint was evaluated on the locked 20,000-image test split.

The completed system uses:

- ResNet-50 transfer learning
- RGB inputs resized to `224 × 224`
- ImageNet normalization
- `Real = 0`
- `Synthetic/Fake = 1`
- Decision threshold `0.51`
- Final checkpoint `models/best_resnet50.pth`

The repository provides both Streamlit and FastAPI interfaces over a single shared inference implementation in `src/synthetic_sight/`.

Final performance and metric interpretation are documented in [Evaluation](evaluation.md).

---

## Repository Consolidation

The collaborative repository accumulated experimental notebooks, intermediate metrics, duplicate deployment implementations, and multiple model artifacts as the project evolved.

For the final repository, the codebase was reorganized around a **single authoritative ResNet-50 pipeline**. Superseded implementations were removed from the main code path so that a reviewer can identify the final model, preprocessing contract, threshold, and deployment logic without having to infer which artifacts are current.

The consolidation removed or retired:

- Exploratory Random Forest, custom CNN, FFT, and PCA implementations from the primary code path
- Duplicate and intermediate ResNet training notebooks
- Duplicate model checkpoints
- Stale metric files
- Experimental ONNX deployment artifacts
- Duplicate Streamlit and API implementations
- Unfinished React/Vite frontend

The role of these experiments is retained in the project history rather than presented as part of the final system.

---

## Final Checkpoint Recovery

During consolidation, the model packaged with the earlier API and Streamlit implementations was found **not to match the final model used for the completed project**.

The older deployment checkpoint reflected an earlier configuration, including epoch 9, threshold `0.50`, dropout `0.40`, and a classifier head without the final BatchNorm layer.

Rather than treating that artifact as final, the discrepancy was traced back through the executed training workflow. The true final checkpoint was recovered from the saved `deepfake_detection_checkpoints` artifacts produced by the original notebook:

```text
training_exploration_reinteration_kristine.ipynb
```

The recovered checkpoint matches the completed training configuration:

- Epoch `13`
- ResNet-50 backbone
- 256-unit classifier hidden layer
- Batch normalization
- ReLU activation
- Dropout `0.30`
- Single binary output
- Decision threshold `0.51`
- Random seed `13`

The verified checkpoint is now stored at:

```text
models/best_resnet50.pth
```

and can be validated with:

```bash
python scripts/verify_checkpoint.py models/best_resnet50.pth
```

Detailed findings from the consolidation are documented in [Repository Review and Consolidation Notes](repository-review.md).

---

## Authoritative Project Structure

The completed system is organized around the following primary files:

```text
README.md
PROJECT_STATUS.md

src/synthetic_sight/
    config.py
    model.py
    inference.py

models/
    best_resnet50.pth
    model_metadata.json
    training_history.csv
    validation_metrics.csv
    final_test_metrics.csv

notebooks/
    01_resnet50_final_training.ipynb
    02_apparent_lightness_audit.ipynb

deployment/
    streamlit_app.py
    api.py

scripts/
    verify_checkpoint.py

tests/
    test_model_contract.py
    test_preprocessing_contract.py

docs/
    architecture.md
    bias-and-ethics.md
    data.md
    deployment.md
    evaluation.md
    project-history.md
    references.md
    repository-review.md
```

---

## Documentation Map

The final documentation separates technical concerns so that each topic has one primary source of detail:

- [Architecture](architecture.md) — final model architecture and optimization
- [Dataset and Data Protocol](data.md) — dataset, preprocessing, splits, and leakage controls
- [Evaluation](evaluation.md) — checkpoint selection and final test results
- [Bias, Representation, and Responsible Use](bias-and-ethics.md) — limitations, representation audit, and responsible-use guidance
- [Deployment](deployment.md) — Streamlit, FastAPI, model verification, and configuration
- [References](references.md) — project sources and supporting literature

This structure preserves the project's exploratory history while keeping the completed system clear and auditable.
