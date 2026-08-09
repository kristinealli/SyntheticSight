# Dataset and Data Protocol

## Source benchmark

The project uses **140k Real and Fake Faces** on Kaggle. The benchmark contains equal numbers of real and synthetic face images:

- **70,000 real images** derived from FFHQ (Flickr-Faces-HQ).
- **70,000 synthetic images** generated with StyleGAN.

The project uses the [140k Real and Fake Faces Kaggle dataset](
https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
(accessed July 2026). This repository does not redistribute the images;
reproduction requires accepting the source platform's dataset terms.

The Kaggle benchmark is organized into predefined training, validation, and test partitions. The final pipeline used 50,000 Real and 50,000 Synthetic training images, 10,000 Real and 10,000 Synthetic validation images, and a locked test set of 10,000 Real and 10,000 Synthetic images. A fixed random seed of 13 was used throughout the pipeline for reproducible sampling, data ordering, and training behavior.

During training, a manifest of the training and validation image paths and labels was generated to support reproducibility and leakage checks. The original Colab run wrote this artifact as resnet50_sample_manifest.csv. Dataset images and environment-specific file paths are not redistributed in this repository.

| Split | Real | Synthetic | Total |
|---|---:|---:|---:|
| Train | 50,000 | 50,000 | 100,000 |
| Validation | 10,000 | 10,000 | 20,000 |
| Test | 10,000 | 10,000 | 20,000 |

## Labels

`0` = Real
`1` = Fake/Synthetic — the positive class for precision, recall, and F1

## Preprocessing

All ResNet-50 inputs are converted to RGB, resized to `224 × 224`, converted to tensors, and normalized with ImageNet statistics:

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

### Training-only augmentation

```text
RandomHorizontalFlip(p=0.5)
ColorJitter(
    brightness=0.10,
    contrast=0.10,
    saturation=0.10,
    hue=0.02,
)
```

Validation and test transforms are deterministic. Rotation was explored earlier but is **not** part of the final pipeline.

## Environment and reproducibility

- Python: `3.x`
- PyTorch: `x.x`
- TorchVision: `x.x`
- Backbone initialization: `EXACT_RESNET50_WEIGHTS`
- Random seed: `YOUR_SEED`
- Device: `YOUR_DEVICE`

## Leakage safeguards

- File paths were checked for zero overlap across training, validation,
  and test partitions.
- The test split was not used for model selection, threshold selection,
  augmentation decisions, or early stopping.
- Path-level separation cannot detect near-duplicate images, shared
  identities, or source-level correlations; these remain limitations.

## Important dataset limitations

The benchmark is well suited to a controlled supervised-learning project, but it narrows the claim:

- The synthetic class represents **StyleGAN**, not every modern generator.
- Images are primarily clean, centered faces rather than uncontrolled social-media content.
- The benchmark does not establish performance on face swaps, video deepfakes, screenshots, multiple faces, strong filters, or heavy compression.
- Equal Real/Fake counts do not imply equal demographic or visual-condition representation.
- High benchmark performance may reflect dataset-specific cues—such as
  generator fingerprints, compression, resolution, or image-processing
  differences—rather than a general ability to detect synthetic media.

The repository intentionally does not redistribute the image dataset. Download it from the cited Kaggle source when reproducing the notebooks.
