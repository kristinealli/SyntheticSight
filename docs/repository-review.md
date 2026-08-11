# Repository Review and Consolidation Notes

## Purpose

This document records the code-quality and artifact-integrity review performed while converting the collaborative Synthetic Sight repository located at [Shloka-16/deepfake-detection](https://github.com/Shloka-16/deepfake-detection) into a clear, final package.

The goal of this review is to maintain transparency and identify which model, metrics, preprocessing rules, and deployment paths represented the completed project and to remove ambiguity created by earlier experimental artifacts.

---
### 1. Deployment Checkpoint Did Not Match the Final Run

The original archive contained two identical deployment copies of `best_resnet50.pth` with SHA-256:

```text
3a74dd6f1c01ee093d977416718d938cb34aa14899b4d82ddf300a9090478390
```

The checkpoint contract reflected an earlier model:

- Epoch `9`
- Threshold `0.50`
- Dropout `0.40`
- Classifier head without the final BatchNorm layer
- No final `head_configuration` metadata

The completed training workflow and presentation instead documented the final configuration:

- Epoch `13`
- Threshold `0.51`
- `BatchNorm1d` in the classifier head
- Dropout `0.30`
- Final test accuracy and Fake F1 of `99.69%`
- ROC-AUC of `99.995%`

**Resolution:** the older deployment checkpoint was not relabeled or treated as final. The correct epoch 13 artifact was recovered from the saved `deepfake_detection_checkpoints` output produced by `training_exploration_reinteration_kristine.ipynb`, validated, and placed at:

```text
models/best_resnet50.pth
```

The recovered final checkpoint SHA-256 is:

```text
d9a7fd6a692c942b550f9848500dc3ffb10d5809cb0d0091990648bf369ad21c
```

---

### 2. Deployment Logic Had Diverged

The collaborative repository contained several deployment experiments with conflicting assumptions, including:

- A Streamlit application importing functionality not provided by its local inference module
- A separate ONNX Streamlit path using threshold `0.45`
- ONNX artifacts that were not usable as final model binaries in the supplied archive
- An API implementation using an older classifier head and a working-directory-dependent model path
- An unfinished Vite/React frontend 

**Resolution:** Streamlit and FastAPI now use the same shared PyTorch inference package in `src/synthetic_sight/`. Model loading, preprocessing, label mapping, checkpoint validation, and threshold logic are centralized rather than duplicated.

---

### 3. Result Artifacts Reflected Earlier Experiments

Some root-level CSV files described earlier evaluation runs, including a threshold of `0.78` and test accuracy of approximately `98.2%`, which did not match the completed model.

**Resolution:** authoritative result files were aligned with the final executed notebook and presentation outputs. The final evaluation is documented in `docs/evaluation.md` and uses the epoch 13 checkpoint with threshold `0.51`.

---

### 4. Documentation and Reproducibility Scaffolding Was Incomplete

The collaborative repository contained placeholder or incomplete documentation and dependency scaffolding, making it difficult for an outside reviewer to identify the final workflow.

**Resolution:** the consolidated repository now separates the major concerns into dedicated documentation for:

- Architecture
- Dataset and preprocessing
- Evaluation
- Bias, representation, and responsible use
- Deployment
- Project history
- References
- Repository consolidation

The final package also includes checkpoint metadata, verification tooling, tests for the model and preprocessing contracts, and a single shared inference implementation.

---

## Final Training Provenance

The executed source notebook that produced the final model was originally named:

```text
training_exploration_reinteration_kristine.ipynb
```

The notebook wrote training artifacts to:

```text
/content/drive/MyDrive/deepfake_detection_checkpoints/
```

Saved artifacts included the model checkpoint, validation and test predictions, metrics, the data manifest, and error-image outputs.

The polished repository preserves the final workflow as:

```text
notebooks/01_resnet50_final_training.ipynb
```

while documenting the original artifact provenance rather than obscuring it.

---

## Files Removed from the Authoritative Code Path

The following exploratory or superseded components were removed from the primary implementation path:

- `deepfake_detector.ipynb` — exploratory baseline work
- `resnet50_training.ipynb` — intermediate ResNet iteration
- `resnet50_training_christina.ipynb` — intermediate team iteration
- `kristinealli_data_exploration.ipynb` — exploratory analysis represented in project history
- `PROJECT_SCOPE.ipynb` — project context consolidated into documentation
- Duplicate Streamlit inference files
- Experimental ONNX deployment code and artifacts
- Unfinished React/Vite frontend
- Stale model binaries and result CSVs

These experiments remain part of the development history but are not presented as final implementation components.

---

## Final Consistency Contract

A deployable Synthetic Sight release should maintain agreement across the following artifacts:

| Contract Item | Final Value / Source |
|---|---|
| Model architecture | ResNet-50 + final binary classifier head |
| Label mapping | `Real = 0`, `Synthetic/Fake = 1` |
| Input size | `224 × 224` RGB |
| Normalization | ImageNet |
| Final checkpoint | `models/best_resnet50.pth` |
| Selected epoch | `13` |
| Decision threshold | `0.51` |
| Shared inference | `src/synthetic_sight/` |
| Final evaluation | Locked 20,000-image test split |

The checkpoint can be checked directly with:

```bash
python scripts/verify_checkpoint.py models/best_resnet50.pth
```

For the final model specification, see [Architecture](architecture.md). For the complete evaluation results, see [Evaluation](evaluation.md).
