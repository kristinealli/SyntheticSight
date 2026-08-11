# Dataset and Data Protocol

## Dataset Source

Synthetic Sight uses the **140k Real and Fake Faces** benchmark from Kaggle. The dataset contains an equal number of real and synthetic face images:

- **70,000 real images** derived from FFHQ (*Flickr-Faces-HQ*)
- **70,000 synthetic images** generated with StyleGAN

**Dataset:** [140k Real and Fake Faces — Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  
**Accessed:** July 2026

The image dataset is **not redistributed in this repository**. Reproducing the project requires downloading the dataset separately and accepting the source platform's terms.

---

## Data Splits

The benchmark is organized into predefined training, validation, and test partitions. The final Synthetic Sight pipeline used the full balanced benchmark:

| Split | Real | Synthetic | Total |
|---|---:|---:|---:|
| Training | 50,000 | 50,000 | 100,000 |
| Validation | 10,000 | 10,000 | 20,000 |
| Test | 10,000 | 10,000 | 20,000 |
| **Total** | **70,000** | **70,000** | **140,000** |

A fixed random seed of `13` was used to support reproducible sampling, data ordering, and training behavior.

During training, a manifest of training and validation image paths and labels was generated for reproducibility and leakage checks. The original Colab workflow saved this artifact as:

```text
resnet50_sample_manifest.csv
```

Dataset images and environment-specific file paths are intentionally excluded from this repository.

> **Audit sample note:** The apparent-lightness audit used smaller analysis samples than the model-training pipeline. Those sample sizes and audit limitations are documented separately in [Bias, Representation, and Responsible Use](bias-and-ethics.md).

---

## Labels

The project uses binary classification:

| Label | Class | Role |
|---:|---|---|
| `0` | Real | Negative class |
| `1` | Synthetic / Fake | Positive class |

Synthetic/Fake is treated as the positive class when calculating precision, recall, and F1 score.

---

## Image Preprocessing

All ResNet-50 inputs are converted to RGB, resized to `224 × 224`, converted to tensors, and normalized using ImageNet statistics. Consistent preprocessing ensures that training and inference inputs follow the same model contract.

### Validation and Test Transform

Validation and test preprocessing is deterministic:

```python
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

### Training Transform

Training images receive limited augmentation before tensor conversion and normalization:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
        saturation=0.10,
        hue=0.02,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
```

The final augmentation strategy uses horizontal flipping and small color variations to introduce modest training variation without substantially altering image content. Rotation was explored during development but was **not included in the final pipeline**. Validation and test images receive no random augmentation.

---

## Reproducibility Controls

The final pipeline includes several controls intended to make the experiment easier to reproduce and audit:

- Fixed random seed: `13`
- Explicit training, validation, and test partitions
- Deterministic validation and test preprocessing
- Recorded image-path and label manifest
- Consistent ImageNet normalization
- Fixed `224 × 224` ResNet-50 input dimensions
- Validation-only model-development and threshold decisions
- Held-out test data reserved for final evaluation
- Checkpoint verification before deployment

The supplied final artifacts do not establish exact Python, PyTorch, and TorchVision versions from the original Colab training environment, so this document does not invent or infer them. Current runtime dependencies should be taken from the repository's dependency configuration, while checkpoint compatibility can be checked with:

```bash
python scripts/verify_checkpoint.py models/best_resnet50.pth
```

The final architecture and training configuration are documented in [Architecture](architecture.md).

---

## Data-Leakage Safeguards

Data separation was treated as part of the evaluation protocol:

- File paths were checked for **zero overlap** across training, validation, and test partitions.
- Training data was used to fit model parameters.
- Validation data was used for model-development and threshold decisions.
- The test split was not used for model selection, threshold selection, augmentation decisions, or early stopping.
- The locked test split was reserved for final performance measurement.

These safeguards reduce direct leakage between experimental partitions. However, path-level separation cannot detect every relationship between images and does not rule out:

- Near-duplicate images
- Repeated identities
- Source-level correlations
- Shared image-processing artifacts

These remain limitations of the benchmark and evaluation design.

---

## Dataset Limitations

The benchmark is well suited to a controlled supervised-learning experiment, but it also defines the limits of the claims that can reasonably be made from the results.

### Generator Coverage

The synthetic class represents **StyleGAN-generated images**. Strong benchmark performance does not establish equivalent performance on diffusion models, face swaps, video deepfakes, or future generation methods.

### Image Conditions

The benchmark primarily contains clean, centered facial images. It does not establish performance on more complex real-world inputs such as screenshots, multiple faces, strong filters, heavy compression, significant cropping, video frames, or other heavily edited images.

### Representation

Balanced Real/Synthetic class counts do **not** imply balanced demographic or visual-condition representation within those classes. Representation and audit limitations are discussed in [Bias, Representation, and Responsible Use](bias-and-ethics.md).

### Dataset-Specific Cues

High benchmark performance may partially reflect characteristics specific to the dataset, including generator fingerprints, compression artifacts, resolution differences, cropping patterns, or other source-specific image characteristics.

For this reason, benchmark performance should not automatically be interpreted as equivalent real-world deepfake-detection performance.

---

## Data Availability

The image dataset is intentionally **not included in this repository**.

To reproduce the project, download the original [140k Real and Fake Faces dataset from Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) and configure the training environment to reference the downloaded data.
