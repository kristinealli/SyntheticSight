# Model Artifacts

## Final deployable checkpoint

The employer-facing repository includes the verified final model at:

```text
models/best_resnet50.pth
```

This checkpoint was recovered from the project's `deepfake_detection_checkpoints` Google Drive artifact archive. That archive was produced by the executed final training notebook originally named:

```text
training_exploration_reinteration_kristine.ipynb
```

The filename contains the original `reinteration` spelling. In this polished repository, the consolidated training notebook is named `notebooks/01_resnet50_final_training.ipynb`.

## Verified model contract

The recovered checkpoint reports:

- checkpoint version: `2`
- architecture: `resnet50`
- selected epoch: `13`
- training stage: `Layer4 fine-tuning`
- checkpoint monitor: `Fake F1`
- validation Fake F1: approximately `0.9977`
- decision threshold: `0.51`
- classifier head: `2048 → 256 → BatchNorm1d → ReLU → Dropout(0.30) → 1`
- labels: `Real=0`, `Fake=1`
- input size: `224 × 224`
- seed: `13`

SHA-256:

```text
d9a7fd6a692c942b550f9848500dc3ffb10d5809cb0d0091990648bf369ad21c
```

Verify the artifact at any time with:

```bash
python scripts/verify_checkpoint.py models/best_resnet50.pth
```

A successful verification returns:

```json
{
  "valid_final_artifact": true,
  "errors": []
}
```

## Legacy artifact note

The original team repository also contained two identical deployment copies named `best_resnet50.pth`. Those were an older epoch-9 checkpoint with threshold `0.50` and an earlier classifier head. They are not used by the final package. Their metadata fingerprint is retained in `legacy_checkpoint_report.json` for provenance.

## GitHub / Git LFS

The final checkpoint is about 220 MB, so `.gitattributes` is configured for Git LFS. Before publishing, ensure Git LFS is enabled in the target repository and confirm that the model artifact is tracked rather than committed as a normal Git blob.
