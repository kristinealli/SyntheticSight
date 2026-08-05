from pathlib import Path

from inference import DeepfakeDetector

checkpoint_path = Path(__file__).resolve().parent / "models" / "best_resnet50.pth"
detector = DeepfakeDetector(checkpoint_path)

print("Checkpoint verified successfully.")
print(f"Path: {checkpoint_path}")
print(f"SHA-256: {detector.checkpoint_sha256}")
print(f"Epoch: {detector.epoch}")
print(f"Monitor: {detector.monitor_metric} = {detector.monitor_value:.4f}")
print(f"Decision threshold: {detector.decision_threshold:.2f}")
