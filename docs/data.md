# Dataset and Data Protocol

## Source benchmark

The project uses **140k Real and Fake Faces** on Kaggle. The benchmark contains equal numbers of real and synthetic face images:

- **70,000 real images** derived from FFHQ (Flickr-Faces-HQ).
- **70,000 synthetic images** generated with StyleGAN.

The final training notebook uses the official benchmark split counts:

| Split | Real | Synthetic | Total |
|---|---:|---:|---:|
| Train | 50,000 | 50,000 | 100,000 |
| Validation | 10,000 | 10,000 | 20,000 |
| Test | 10,000 | 10,000 | 20,000 |

## Labels

- `0` = Real
- `1` = Fake/Synthetic — the positive class for precision, recall, and F1

## Preprocessing

All ResNet-50 inputs are converted to RGB, resized to `224 × 224`, converted to tensors, and normalized with ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
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

## Leakage safeguards

- Training and validation paths were checked for zero overlap.
- The official test split was withheld until development decisions were complete.
- A stronger future reproducibility audit should additionally detect **near-duplicate images** and source-level correlations; path-level separation alone does not prove that none exist.

## Important dataset limitations

The benchmark is well suited to a controlled supervised-learning project, but it narrows the claim:

- The synthetic class represents **StyleGAN**, not every modern generator.
- Images are primarily clean, centered faces rather than uncontrolled social-media content.
- The benchmark does not establish performance on face swaps, video deepfakes, screenshots, multiple faces, strong filters, or heavy compression.
- Equal Real/Fake counts do not imply equal demographic or visual-condition representation.

The repository intentionally does not redistribute the image dataset. Download it from the cited Kaggle source when reproducing the notebooks.
