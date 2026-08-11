# Evaluation

## Evaluation Protocol

Model development used training and validation data. The official **20,000-image test split** was reserved for final evaluation after the final model and operating threshold had been established.

Keeping the test set outside the development loop reduces the risk of tuning decisions to test feedback and preserves it as an estimate of same-benchmark generalization.

---

## Checkpoint and Threshold Selection

The final training implementation evaluates validation performance at each epoch and searches decision thresholds from `0.05` through `0.95` in `0.01` increments. Threshold selection is based on **validation Fake F1**, with ties resolved toward `0.50`.

The final checkpoint is **epoch 13** with a decision threshold of **0.51**. Epoch 13 also recorded the lowest validation loss in the run (`0.0077`). As a result, the implementation's validation-F1 checkpointing behavior and the final presentation's description of selecting the lowest-validation-loss model identify the same final artifact.

The test set was not used to choose the epoch or threshold.

---

## Final Locked-Test Results

The final checkpoint was evaluated on **20,000 held-out images**: 10,000 Real and 10,000 Synthetic.

| Metric | Result |
|---|---:|
| Accuracy | 99.69% |
| Fake F1 | 99.69% |
| ROC-AUC | 99.995% |
| Decision threshold | 0.51 |

The model classified **19,938 of 20,000 images correctly**, with 62 total errors.

### Confusion Matrix

| Ground truth | Predicted Real | Predicted Synthetic |
|---|---:|---:|
| Real (10,000) | 9,961 TN | 39 FP |
| Synthetic (10,000) | 23 FN | 9,977 TP |

Derived from the confusion matrix:

| Measure | Result |
|---|---:|
| Fake precision | ≈ 99.61% |
| Fake recall / sensitivity | 99.77% |
| Specificity | 99.61% |
| False-positive rate | 0.39% |
| False-negative rate | 0.23% |

![Final test confusion matrix](../assets/resnet50_final_test_confusion_matrix.png)

---

## Metric Interpretation

- **Accuracy** — proportion of all test images classified correctly at the selected threshold.
- **Precision** — among images predicted Synthetic, the proportion that were actually synthetic.
- **Recall** — among all synthetic images, the proportion correctly detected.
- **F1 score** — harmonic mean of precision and recall.
- **ROC-AUC** — measures class-ranking and separation quality across many possible thresholds; it is **not** the percentage of images classified correctly.
- **Decision threshold** — cutoff applied to the model's synthetic score. A threshold of `0.51` does **not** mean the model is "51% sure."

![ROC and precision-recall curves](../assets/resnet50_roc_pr_curves.png)

---

## Error Interpretation

A **false positive** occurs when a real image is incorrectly classified as Synthetic. In a real-world setting, this could cast unnecessary doubt on legitimate content.

A **false negative** occurs when a synthetic image is incorrectly classified as Real. This could create false reassurance if the prediction were mistaken for provenance verification.

Both error types matter, and the appropriate operating threshold depends on the intended use and the relative cost of each error.

---

## Interpreting the Results in Context

The results demonstrate very strong separation **within the 140k Real and Fake Faces benchmark**. They do not establish equivalent performance on unrelated datasets, newer generators, face swaps, video, heavily transformed images, or other out-of-distribution inputs.

Outside-distribution evaluation is therefore a priority for future work.

For a fuller discussion of representation, benchmark limitations, and responsible interpretation, see [Bias, Representation, and Responsible Use](bias-and-ethics.md).
