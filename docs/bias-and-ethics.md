# Bias, Representation, and Responsible Use

## Responsible Use

Synthetic Sight is an **image-level classifier** designed to provide a screening signal for the project's **real-versus-synthetic face-image** task.

The model can help reduce inconsistency in human visual judgment, but its output should **not** be treated as proof that an image is authentic or synthetic. Like any machine-learning system, Synthetic Sight may reflect gaps in its training data, rely on source-specific visual artifacts, or produce confident predictions when presented with images that differ from those it was trained to recognize.

For that reason, results should be interpreted as a **decision aid for human review**, not as a definitive determination of authenticity.

This project does **not** claim to detect all forms of deepfakes or to generalize reliably beyond the benchmark conditions represented in its evaluation data. Performance may differ substantially when the model encounters:

- Newer image-generation models
- Diffusion-generated images
- Face swaps
- Video deepfakes
- Heavily edited or compressed images
- Other out-of-distribution cases

---

## Where Bias and Error Can Enter the System

Our high aggregate accuracy does not eliminate the possibility of bias or uneven model performance. We identified several avenues through which bias, error, or overconfidence might enter the system.

### Representation Gaps

Error rates may vary across underrepresented people, visual characteristics, or image conditions, even when overall performance is high.

### Generator Drift

The synthetic images represented in this project are associated with StyleGAN-generated data. A detector that performs well on these images may not transfer reliably to diffusion models, face swaps, video, or future image-generation methods.

### Shortcut Learning

Rather than learning a general concept of image authenticity, the network may partially rely on unintended visual cues associated with the benchmark data, including:

- Cropping
- Resolution
- Compression artifacts
- Lighting
- Background characteristics
- Dataset-specific preprocessing

These shortcuts may produce strong benchmark performance while reducing reliability on real-world images.

### Evaluation Bias

Repeatedly examining or tuning decisions against a test set can gradually turn that test set into another development set and lead to overly optimistic estimates of model performance.

### Interface Overtrust

A polished probability display can make a prediction appear more certain than it actually is. Model confidence reflects the patterns learned by the classifier; it is not independent verification that an image is authentic or synthetic.

---

## Mitigations Used in This Project

We incorporated several controls into the dataset preparation, training, evaluation, representation audit, and deployment process. These steps were intended to reduce avoidable sources of bias, overfitting, and overconfidence while making the model's limitations more transparent.

- **Standardized Image Preprocessing:**  
  Images were processed using consistent rules for image size, color mode, and normalization so that the model received inputs in a uniform format. This reduces the likelihood that differences caused purely by inconsistent preprocessing would influence the model's predictions.

- **Balanced Training Classes:**  
  The training subset contained an equal number of Real and Synthetic images. This prevented one class from dominating training simply because it appeared more frequently in the dataset.

- **Light Training Augmentation:**  
  Training images received limited augmentation, including horizontal flips and small color variations. These changes introduced some visual variation during training and reduced dependence on the exact presentation of individual training images. Validation and test images were kept clean so that evaluation remained consistent. Slight image rotation was expolored in an attempt to introduce additional variations in feature position, however was removed from the augmentation pipeline due to concerns that the edges introduced by the skewed image would also skew the training.

- **Separate Training, Validation, and Test Data:**  
  Model development and final evaluation were kept separate. Training data was used to fit the model, validation data was used to make development decisions, and the test set was reserved for final evaluation. This helps prevent information from the final evaluation set from influencing model development.

- **Validation-Based Model Selection:**  
  Training was monitored using validation performance, and the final checkpoint was selected based on the **lowest validation loss** rather than simply using the last training epoch. This helped identify the model that generalized best to unseen validation data during training.

- **Threshold Selection Without Using the Test Set:**  
  The classification threshold was selected using validation data rather than tuned against final test results. The final operating threshold was **0.51**, after which it was applied to the held-out test set without further adjustment.

- **Locked Final Test Evaluation:**  
  Final performance was measured on a held-out test set of **20,000 images**, evenly divided between Real and Synthetic images. The test set was used to estimate final performance rather than as another source of training or tuning feedback.

- **Reporting Errors, Not Only Aggregate Accuracy:**  
  In addition to overall performance metrics, we examined the model's actual mistakes. On the final test set, the model produced **39 false positives** (real images classified as synthetic) and **23 false negatives** (synthetic images classified as real). Reporting both error types makes it easier to understand how failures could affect real-world use.

- **Preliminary Representation Aaudit:**  
  We conducted an apparent-lightness audit to examine whether the Real and Synthetic image sets occupied similar ranges of visible facial lightness. Because reliable self-identified demographic labels were unavailable, the audit deliberately avoided inferring race, ethnicity, ancestry, gender, or biological skin color. Its limitations—including unequal face-detection success—are documented alongside the results.

- **Explicit Consideration of Detector-Related Selection Bias:**  
  The representation audit also measured whether OpenCV successfully detected faces at comparable rates across the Real and Synthetic datasets. Because an image must first be detected before it can receive an `L*` measurement, differences in detection success were treated as a potential source of selection bias rather than ignored.

- **Responsible Deployment Language:**  
  The Streamlit application describes its output as a **screening signal rather than proof** of authenticity or manipulation. The documentation also states that performance demonstrated on this benchmark should not be assumed to transfer to newer generators, face swaps, video, heavily edited images, or other out-of-distribution cases.

These measures improve the reliability and transparency of the project, but they do not fully eliminate bias or establish that the classifier is equally reliable across all people, generators, or real-world image conditions. They are safeguards within the scope of this experiment, not a fairness certification.

---

## Preliminary Apparent-Lightness Audit

Because reliable self-identified demographic labels were not available, the project intentionally did **not** attempt to infer race, ethnicity, ancestry, gender, or biological skin color from facial appearance. Instead, the audit examined a narrower and directly observable image characteristic: **apparent facial lightness**. The purpose of the audit was to explore whether the real and synthetic image sets occupied similar ranges of visible image lightness without assigning demographic identities to the people represented.

### Audit Method

The audit followed five main steps:

1. Detect a face with OpenCV.
2. Select the largest face in the centered portrait benchmark.
3. Sample conservative forehead and cheek regions.
4. Convert sampled color values to CIE Lab.
5. Use `L*` as a measure of **apparent image lightness**.

In CIE Lab color space, `L*` represents perceptual lightness.

As the measured value can also be influenced by lighting, exposure, white balance, makeup, editing, cropping, and face-detection errors. The audit should therefore be understood as a representation check, not a demographic classification and not a fairness certification.

---

### Face-Detection Success

Before an image could receive an `L*` measurement, the audit first had to successfully detect a usable face.

| Split / class        |               Detected |
| -------------------- | ---------------------: |
| Training synthetic   | 99.32% (4,966 / 5,000) |
| Training real        | 94.24% (4,712 / 5,000) |
| Validation synthetic |   99.20% (992 / 1,000) |
| Validation real      |   95.90% (959 / 1,000) |

Detection success differed between the real and synthetic image sets. This matters because images that are not successfully detected are automatically excluded from the lightness analysis.

The face detector therefore influences which images are represented in the audit, creating an additional potential source of selection bias before the `L*` comparison is performed. While this audit is best understood as an initial diagnostic, it helps identify representation patterns and limitations that deserve further investigation.

![Validation apparent-lightness distribution](../assets/apparent_lightness_validation_distribution.png)

---

### Relative Lightness Bands

Within the validation sample, the real and synthetic images had similar average measured `L*` values.

This descriptive similarity does **not** establish:

- Equivalent demographic representation
- Balanced representation across the full lightness range
- Comparable model performance across image-lightness conditions
- Overall fairness of the classifier

To make the distribution easier to interpret, measured faces were grouped into fixed relative `L*` ranges. In the validation sample, the highest and lowest lightness bands appear to contain more real faces than synthetic faces. This pattern may reflect differences in the underlying dataset composition, face-detection success, or the distribution of lighting conditions across the samples rather than evidence of demographic bias. Because these bands are analytic bins rather than social categories, they should be read as descriptive summaries, not as evidence of fairness or population representation. More work is needed to evaluate model behavior across lightness strata to ensure equity and representativeness.

![Relative apparent-lightness bands](../assets/apparent_lightness_relative_bands.png)

---

## Privacy

Images uploaded to the Synthetic Sight application are used **only for the current inference request**. The application does not retain or use uploaded images to retrain the model.
