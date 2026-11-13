
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Horse Prediction",
    page_icon="🐎",
    layout="centered"
)

# App title
st.title("🐎 Horse Image Prediction")
st.write("Upload an image to run your trained CNN model.")

MODEL_PATH = "horse_cnn_final.keras"
IMG_SIZE = (160, 160)

# Load the trained model
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_cnn_model()

if model is None:
    st.error(
        "Model file not found. Place horse_cnn_final.keras "
        "in the same folder as app.py."
    )
    st.stop()

st.success("CNN model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a horse image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict", type="primary"):
        with st.spinner("Predicting..."):
            # Preprocess image
            image_resized = image.resize(IMG_SIZE)

            image_array = np.array(
                image_resized, dtype=np.float32
            )
            image_array = np.expand_dims(image_array, axis=0)

            # Run CNN prediction
            predictions = model.predict(
                image_array, verbose=0
            )[0]

            confidence = float(np.max(predictions)) * 100

        st.subheader("Prediction Result")
        st.success("🐎 Horse")
        st.write(f"**Model output confidence:** {confidence:.2f}%")
