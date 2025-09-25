# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Load trained model ---
@st.cache_resource
def load_trained_model():
    model = load_model("model.h5") 
    return model

model = load_trained_model()

# --- App UI ---
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray and the AI will predict if pneumonia is present.")

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # --- Preprocess the image ---
    img_resized = img.resize((64, 64))  
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    # --- Make prediction ---
    prediction = model.predict(img_array)[0][0]

    # --- Display result ---
    if prediction > 0.5:  # sigmoid output threshold
        st.error(f"⚠️ Prediction: Pneumonia detected (probability: {prediction:.2f})")
    else:
        st.success(f"✅ Prediction: Normal lungs (probability: {1 - prediction:.2f})")
