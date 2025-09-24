import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your trained model (make sure this file exists in the repo)
@st.cache_resource
def load_trained_model():
    return load_model('pneumonia_model.h5')

model = load_trained_model()

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure it's RGB
    image = image.resize((150, 150))  # Adjust size if your model uses different input
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit UI
st.title("🩺 Pneumonia Detector")
st.write("Upload a chest X-ray image and the model will predict whether the person has pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Assuming binary classification: 0 = Normal, 1 = Pneumonia
    result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
    confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else 1 - float(prediction[0][0])

    st.markdown(f"### 🧠 Prediction: **{result}**")
    st.markdown(f"Confidence: `{confidence:.2%}`")

