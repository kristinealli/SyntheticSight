# Evaluation

## Evaluation protocol

Model development used training and validation data. The final checkpoint and operating threshold were fixed before the **20,000-image official test split** was evaluated. Treating the test set as a final-only measurement reduces the risk of tuning to test feedback.

## Checkpoint and threshold selection

The final training implementation checkpoints on **validation Fake F1**. For each epoch, the decision threshold is searched from `0.05` through `0.95` in `0.01` increments, selecting the threshold with the highest Fake F1 and breaking ties toward `0.50`.

The selected checkpoint is **epoch 13** with threshold **0.51**. Epoch 13 also has the lowest recorded validation loss (`0.0077`) in the run, so the implementation and the final presentation's “lowest validation loss” description identify the same model even though the code's explicit checkpoint monitor is Fake F1.

## Final locked-test results

| Metric | Result |
|---|---:|
| Accuracy | 99.69% |
| Fake F1 | 99.69% |
| ROC-AUC | 99.995% |
| Threshold | 0.51 |

### Confusion matrix

| Ground truth | Predicted Real | Predicted Synthetic |
|---|---:|---:|
| Real (10,000) | 9,961 TN | 39 FP |
| Synthetic (10,000) | 23 FN | 9,977 TP |

Derived from the matrix:

- Fake precision ≈ **99.61%**
- Fake recall / sensitivity = **99.77%**
- Specificity = **99.61%**
- False-positive rate = **0.39%**
- False-negative rate = **0.23%**

![Final test confusion matrix](../assets/resnet50_final_test_confusion_matrix.png)

## Metric interpretation

- **Accuracy:** proportion of all test images classified correctly at the chosen threshold.
- **Precision:** among images predicted Synthetic, the proportion that were actually synthetic.
- **Recall:** among all synthetic images, the proportion detected.
- **F1:** harmonic mean of precision and recall.
- **ROC-AUC:** class-ranking/separation quality across many possible thresholds. It is **not** the percentage of images classified correctly.
- **Decision threshold:** the cutoff applied to the synthetic score. `0.51` does **not** mean the model is “51% sure.”

![ROC and precision-recall curves](../assets/resnet50_roc_pr_curves.png)

## Error meaning

A **false positive** is a real image incorrectly flagged Synthetic. This can cast doubt on legitimate content, so the output should never be presented as an accusation or proof.

A **false negative** is a synthetic image incorrectly labeled Real. This can create false reassurance if a user mistakes the model's label for provenance verification.

Both error types matter. The appropriate operating threshold depends on the use case and the relative cost of each error.

## Why the score can be excellent and still limited

Development and test images come from the same overall benchmark source families. Strong same-benchmark separation does not prove generalization to unrelated datasets, newer generators, or real-world transformations. Outside-distribution evaluation is the highest-priority technical next step.
