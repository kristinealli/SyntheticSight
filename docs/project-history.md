# Project History

## Overview

**Synthetic Sight** began as an exploratory AI4ALL Ignite team project investigating whether machine-learning models could distinguish real human faces from AI-generated faces. Over the course of the project, the codebase evolved through several modeling approaches, data-analysis experiments, and deployment prototypes before converging on the final **ResNet-50 transfer-learning pipeline** presented by the team.

The original collaborative repository is available at:

https://github.com/Shloka-16/deepfake-detection

This repository is a consolidated, portfolio-ready version of that work. It preserves the final model, evaluation artifacts, reproducibility information, and relevant project history while removing duplicate or superseded implementation paths.

---

## Project evolution

### 1. Early exploration

The project began by comparing several possible approaches to synthetic-face detection. Early work included:

- Random Forest baselines
- Custom convolutional neural networks (CNNs)
- Frequency-domain / FFT experiments
- PCA-based exploratory analysis
- Dataset and bias-analysis notebooks
- Multiple deployment experiments

These approaches were useful for understanding the problem space, testing assumptions, and identifying which direction was most promising. They were not, however, the final deployed system.

### 2. Transition to transfer learning

The project ultimately moved to **ResNet-50 pretrained on ImageNet**. Transfer learning provided a stronger visual feature extractor than the smaller experimental models while keeping training practical.

The final training strategy used two stages:

1. **Epochs 1–3:** train the new binary classifier head while keeping the pretrained ResNet-50 backbone frozen.
2. **Epochs 4–15:** unfreeze ResNet-50 `layer4` and fine-tune the higher-level feature representations.

Training-only augmentation was limited to random horizontal flipping and light color jitter. Rotation was explored earlier but was removed from the final pipeline.

The final selected checkpoint came from **epoch 13**.

### 3. Final evaluation and deployment

After model and threshold selection were completed using validation data, the model was evaluated on the locked test set.

The final system uses:

- ResNet-50 transfer learning
- RGB inputs resized to `224 × 224`
- ImageNet normalization
- Real = `0`
- Synthetic/Fake = `1`
- Decision threshold = `0.51`
- Final checkpoint = `models/best_resnet50.pth`

The final repository also includes a Streamlit interface and FastAPI endpoint that both use the same shared inference implementation from `src/synthetic_sight/`.

---

## What changed from the original project archive

The original team repository contained many valuable exploratory files, but it also accumulated duplicate notebooks, intermediate results, multiple deployment implementations, and stale model artifacts as the project evolved.

For this consolidated version, the repository was reorganized around the **single final ResNet-50 pipeline** used for the presentation.

The following were removed from the main code path:

- Exploratory Random Forest implementations
- Custom CNN experiments
- FFT-based experiments
- PCA experiments
- Duplicate training notebooks
- Duplicate model checkpoints
- Intermediate and stale metric files
- Experimental ONNX deployment artifacts
- Duplicate API and Streamlit implementations
- The unfinished React frontend

These files were removed because they no longer represented the final system and made it difficult for a reviewer to determine which code and model were authoritative.

Their role in the project is documented here rather than being presented as production-ready components.

---

## Checkpoint integrity and recovery

During repository consolidation, an important model-integrity issue was identified.

The checkpoint copied into the original API and Streamlit deployment folders was **not the final model used for the completed project**. It was an older checkpoint with characteristics including:

- Epoch 9
- Decision threshold `0.50`
- Dropout `0.40`
- No BatchNorm layer in the custom classifier head

Rather than silently treating that checkpoint as final, the discrepancy was investigated.

The true final checkpoint was recovered from the project's saved `deepfake_detection_checkpoints` artifact archive produced by the executed training workflow originally named:

`training_exploration_reinteration_kristine.ipynb`

The recovered checkpoint matches the final training configuration:

- Epoch 13
- ResNet-50 backbone
- Classifier hidden layer of 256 units
- Batch normalization in the classifier head
- ReLU activation
- Dropout `0.30`
- Final binary output layer
- Decision threshold `0.51`
- Random seed `13`

The recovered checkpoint is now stored at:

`models/best_resnet50.pth`

It is validated with:

```bash
python scripts/verify_checkpoint.py models/best_resnet50.pth
```

The verification script checks the artifact metadata and SHA-256 checksum before the model is used for deployment.

---

## Why the repository was consolidated

The goal of the consolidation was not to rewrite the history of the project or make the development process appear more linear than it was.

Exploration, failed approaches, duplicated work, and changing implementation decisions are normal parts of an applied machine-learning project. However, a final repository should make it clear which artifacts represent the completed system.

The consolidated structure therefore separates:

- **final implementation** from exploratory work,
- **validated results** from stale intermediate outputs,
- **shared inference logic** from duplicate deployment code, and
- **documented limitations** from unsupported generalization claims.

This makes the repository easier to audit, reproduce, review, and maintain.

---

## Current authoritative project structure

The primary files for the completed system are:

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

## Scope of the final claim

The final model should be understood as a **synthetic-face classifier evaluated on the 140k Real and Fake Faces benchmark**, not as a universal deepfake detector.

The benchmark primarily contains clean, centered face images, with synthetic examples generated by StyleGAN. Strong performance on this benchmark does not establish equivalent performance on:

- newer image generators,
- face swaps,
- video deepfakes,
- screenshots,
- heavily compressed images,
- filtered images,
- multi-face scenes, or
- arbitrary synthetic media.

For that reason, the deployed output is best treated as a **review signal** that can support digital literacy and human judgment rather than as proof that an image is authentic or synthetic.

---

## Final repository philosophy

This version of Synthetic Sight is intended to show both the technical outcome and the engineering process behind it.

The final repository emphasizes:

- reproducibility,
- model provenance,
- clear separation of train/validation/test responsibilities,
- a single authoritative inference pipeline,
- explicit model limitations,
- responsible communication of results, and
- transparent documentation of the changes made during consolidation.

The exploratory work helped produce the final result. The consolidated repository makes that result understandable.
