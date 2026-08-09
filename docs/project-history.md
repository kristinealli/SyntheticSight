# Project Evolution

Synthetic Sight evolved from a broad exploration into one reproducible final pipeline.

## 1. Proposal

The team framed the problem around AI-generated faces, digital literacy, and supervised classification. Early plans considered both traditional machine learning and neural-network approaches.

## 2. Exploration

Approaches explored during development included:

- **Random Forest** as a traditional machine-learning baseline;
- a **custom CNN** trained specifically for the task;
- **FFT (Fast Fourier Transform)** features to explore frequency-domain artifacts;
- **PCA** for exploratory visualization;
- several ResNet-50 training iterations.

The final presentation did not report a comparable metric table for every exploratory model, so this repository does not invent exact rankings for them.

## 3. Consolidation on ResNet-50

The project narrowed to ImageNet-pretrained **ResNet-50 transfer learning** because a focused pipeline made it possible to improve one model, preserve evaluation discipline, document the implementation, and deploy a coherent prototype.

Key final changes included:

- complete 100k/20k/20k benchmark split usage;
- seed 13;
- training-only horizontal flip and light color jitter;
- classifier head with BatchNorm and dropout `0.30`;
- first 3 epochs head-only, then `layer4` fine-tuning;
- Adam with weight decay;
- validation-only threshold selection;
- final locked-test evaluation;
- explicit false-positive/false-negative review;
- apparent-lightness representation audit;
- Streamlit and FastAPI prototype interfaces.

## 4. Employer-facing repository cleanup

The final portfolio package removes duplicate or unfinished code paths that could obscure the project story. The original exploratory work is still represented in the project history and final executed notebook, while the runnable code path is intentionally small: one model definition, one inference implementation, two deployment interfaces, and clear documentation.
