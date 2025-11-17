# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
import gdown

# --- Configuration ---
MODEL_PATH = 'model_new.keras'
# Extract the ID from your Google Drive share link
DRIVE_FILE_ID = '1gXYyVyZyRraq_jYY8omDw3dHBjP9NNQX'

if not os.path.exists(MODEL_PATH):
    st.info("Model not found locally. Downloading from Google Drive...")
    try:
        # Construct the gdown URL using the ID and output path
        gdown.download(id=DRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Error: Could not download the model from Google Drive. Please check the file ID and sharing settings. Details: {e}")
        st.stop()

# --- 1. Load the Model ---
@st.cache_resource
def load_pneumonia_model(path):
    return load_model(path)

model = load_pneumonia_model(MODEL_PATH)

# --- 2. Image Preprocessing Function ---
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((64, 64))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# --- 3. Prediction Function ---
def predict_pneumonia(processed_img):
    prediction = model.predict(processed_img)
    prob = prediction[0][0]

    if prob > 0.5:
        return "PNEUMONIA Detected", prob
    else:
        return "NORMAL (No Pneumonia)", 1 - prob

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("AI Chest X-Ray Pneumonia Detector 🩺")
st.markdown("Upload a chest X-ray image (JPG/PNG) to receive an instant diagnosis.")

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Chest X-Ray", use_column_width=True)
    
    st.subheader("Analysis Results:")
    with st.spinner("Analyzing with VGG16 model..."):
        try:
            processed = preprocess_image(uploaded_file)
            result, confidence = predict_pneumonia(processed)

            st.markdown(f"**Diagnosis:** <span style='font-size: 24px;'>{result}</span>", unsafe_allow_html=True)

            if "PNEUMONIA" in result:
                st.error(f"🚨 **Confidence in Pneumonia:** {confidence:.2f}")
            else:
                st.success(f"✅ **Confidence in Normal:** {confidence:.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
