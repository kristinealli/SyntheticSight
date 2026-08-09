# Bias, Representation, and Responsible Use

## Core principle

Synthetic Sight is an image-level classifier that provides a review signal for the project’s real-versus-synthetic face-image task. It can reduce inconsistency in human visual judgment, but it may also reflect dataset gaps, rely on source-specific artifacts, and encourage overconfidence in a probability score. Results should never be treated as proof of authenticity or manipulation.

This project does not claim to detect all deepfakes or to generalize reliably to new generators, video, face swaps, heavily edited images, or conditions not represented in its evaluation data.

## How bias can enter the system

**Representation gaps:** error rates may vary across underrepresented people or image conditions even when aggregate accuracy is high.
**Generator drift:** a StyleGAN detector may not transfer to diffusion models, face swaps, video, or future generators.
**Shortcut learning:** the network may exploit cropping, resolution, compression, lighting, or background cues associated with benchmark sources.
**Evaluation bias:** repeated use of a test set can turn it into another development set and inflate reported performance.
**Interface overtrust:** a polished probability display can make a model score look more certain than it is.

## Preliminary apparent-lightness audit

The project intentionally did **not** infer race, ethnicity, ancestry, gender, or biological skin color from faces. Instead, the audit measured a narrower, observable image characteristic:

1. Detect a face with OpenCV.
2. Select the largest face in the centered portrait benchmark.
3. Sample conservative forehead and cheek regions.
4. Convert sampled color values to CIE Lab.
5. Use `L*` as a measure of **apparent image lightness**.

`L*` is still affected by lighting, exposure, white balance, makeup, editing, cropping, and detector errors. The audit is therefore a **representation check**, not a demographic label and not a fairness certification.

### Face-detection success in the audit

| Split / class | Detected |
|---|---:|
| Training synthetic | 99.32% (4,966 / 5,000) |
| Training real | 94.24% (4,712 / 5,000) |
| Validation synthetic | 99.20% (992 / 1,000) |
| Validation real | 95.90% (959 / 1,000) |

The detector itself therefore changes which images receive an `L*` measurement and can introduce selection bias into the audit.

![Validation apparent-lightness distribution](../assets/apparent_lightness_validation_distribution.png)

### Relative lightness bands

In this validation sample, the real and synthetic images had similar average measured L* values. This descriptive result does not establish comparable representation, demographic balance, or comparable model performance across image-lightness conditions. Relative lightness bands were created using fixed L* ranges; they are analytical bins, not demographic categories.

![Relative apparent-lightness bands](../assets/apparent_lightness_relative_bands.png)

## Mitigations used in this project

- Consistent image size, color mode, and normalization.
- Balanced Real/Fake labels.
- Separate validation and locked test data.
- Threshold selection on validation data only.
- Explicit reporting of false positives and false negatives.
- Preliminary representation audit with limitations documented.
- Deployment language that describes results as review signals, not proof.


## Privacy: 

Uploaded images are used only for the current inference request. The application does not use them to retrain the model
  
