"""Streamlit interface for the final Synthetic Sight PyTorch pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from synthetic_sight.inference import CheckpointCompatibilityError, SyntheticSightDetector


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "SYNTHETIC_SIGHT_MODEL_PATH",
        REPOSITORY_ROOT / "models" / "best_resnet50.pth",
    )
)

st.set_page_config(page_title="Synthetic Sight", page_icon="👁️", layout="centered")


@st.cache_resource
def load_detector() -> SyntheticSightDetector:
    """Load and validate the final model checkpoint once per app process."""
    return SyntheticSightDetector(MODEL_PATH)


st.title("Synthetic Sight")
st.subheader("AI-generated face image detection")
st.write(
    "Upload a face image to receive the model's synthetic-image score. "
    "Synthetic Sight is a research prototype trained to separate FFHQ real faces "
    "from StyleGAN-generated faces; it is not an authentication service."
)
st.warning(
    "Treat the result as a review flag, not proof. A Real result does not verify "
    "authenticity, and a Synthetic result can be wrong."
)

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Best results are expected for clear, centered face images similar to the training benchmark.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("That file could not be opened as an image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze image", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing image…"):
                result = load_detector().predict(image)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.info("Add the verified final checkpoint described in models/README.md.")
            st.stop()
        except CheckpointCompatibilityError as exc:
            st.error(f"Checkpoint validation failed: {exc}")
            st.stop()

        display_label = "Synthetic" if result.label == "Fake" else "Real"
        if display_label == "Synthetic":
            st.error(f"Prediction: {display_label}")
        else:
            st.success(f"Prediction: {display_label}")

        left, right = st.columns(2)
        left.metric("Synthetic score", f"{result.fake_probability:.1%}")
        right.metric("Real score", f"{result.real_probability:.1%}")

        st.progress(result.fake_probability, text=f"Synthetic score: {result.fake_probability:.1%}")
        st.caption(
            f"Decision threshold: {result.decision_threshold:.2f}. The model labels an "
            "image Synthetic when its synthetic score is at or above this cutoff. "
            "The score is not independently calibrated proof of image provenance."
        )

with st.expander("Model scope and limitations"):
    st.markdown(
        """
- Binary still-image classifier: **Real (0)** vs. **Synthetic/Fake (1)**.
- Final benchmark: FFHQ real faces and StyleGAN-generated faces.
- Input: RGB face image resized to **224 × 224** with ImageNet normalization.
- Not validated here for video, face swaps, screenshots, heavy compression, multiple faces, or newer generator families.
- Designed for digital-literacy support and human review, not legal, identity, or security authentication.
        """
    )

st.divider()
st.caption("Synthetic Sight · AI4ALL Ignite · Team 9C · Educational research prototype")
