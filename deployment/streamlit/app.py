from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from inference import DeepfakeDetector

APP_DIRECTORY = Path(__file__).resolve().parent
CHECKPOINT_PATH = APP_DIRECTORY / "models" / "best_resnet50.pth"

st.set_page_config(
    page_title="Synthetic Sight",
    page_icon="👁️",
    layout="centered",
)


@st.cache_resource
def load_detector() -> DeepfakeDetector:
    return DeepfakeDetector(CHECKPOINT_PATH)


st.title("Synthetic Sight")
st.subheader("AI-Generated Face Image Detection")
st.write(
    "Upload a face image to receive an estimate of whether the image is "
    "real or AI-generated. The prediction is produced by the final "
    "ResNet-50 model used in the slide presentation."
)
st.warning(
    "This is an educational research prototype. A prediction is not "
    "definitive proof that an image is authentic or AI-generated."
)

# Load immediately so a wrong checkpoint is reported before the user tests images.
try:
    detector = load_detector()
except Exception as error:
    st.error("The presentation model could not be loaded.")
    st.code(str(error))
    st.stop()

with st.expander("Loaded model details"):
    st.write(f"Checkpoint: `{CHECKPOINT_PATH}`")
    st.write(f"Best epoch: **{detector.epoch}**")
    st.write(f"Decision threshold: **{detector.decision_threshold:.2f}**")
    st.write(
        f"Validation monitor: **{detector.monitor_metric} "
        f"{detector.monitor_value:.4f}**"
    )
    st.write(f"SHA-256: `{detector.checkpoint_sha256}`")

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Upload an image to begin.")
else:
    try:
        image = Image.open(uploaded_file)
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be opened as an image.")
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze image", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing the image..."):
                result = detector.predict(image)

            prediction = str(result["prediction"])
            confidence = float(result["confidence"])
            fake_probability = float(result["fake_probability"])
            real_probability = float(result["real_probability"])
            threshold = float(result["decision_threshold"])

            if prediction == "Fake":
                st.error(f"Prediction: Fake — {confidence:.1%} confidence")
            else:
                st.success(f"Prediction: Real — {confidence:.1%} confidence")

            left, right = st.columns(2)
            left.metric("Fake probability", f"{fake_probability:.1%}")
            right.metric("Real probability", f"{real_probability:.1%}")

            st.progress(fake_probability, text=f"Fake probability: {fake_probability:.1%}")
            st.caption(
                f"The image is labeled Fake when its fake probability is "
                f"greater than or equal to the validation-selected threshold "
                f"of {threshold:.2f}."
            )
        except Exception as error:
            st.error("The model could not complete the prediction.")
            st.code(str(error))

st.divider()
st.caption("Synthetic Sight · AI4ALL Ignite · Educational prototype")
