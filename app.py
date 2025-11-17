# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os

# --- Configuration ---
MODEL_PATH = 'model.keras'
IMAGE_SIZE = (64, 64) # Must match the input size your model was trained on

# Check if the model exists before trying to load
if not os.path.exists(MODEL_PATH):
    st.error("Error: The trained model 'model.h5' was not found.")
    st.markdown("Please run **`cleaned_script.py`** first to train and save the model.")
    st.stop()


# --- 1. Load the Model ---
# st.cache_resource loads the model once, speeding up the app
@st.cache_resource
def load_pneumonia_model(path):
    """Loads the pre-trained Keras model."""
    return load_model(path)

model = load_pneumonia_model(MODEL_PATH)


# --- 2. Image Preprocessing Function ---
def preprocess_image(img_path, target_size):
    """Loads and preprocesses an image for the model."""
    # Read the image file bytes
    file_bytes = np.asarray(bytearray(img_path.read()), dtype=np.uint8)
    # Decode the image to OpenCV format (BGR)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Resize and normalize
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    
    return img

# --- 3. Prediction Function ---
def predict_pneumonia(processed_img):
    """Makes a prediction using the loaded model."""
    prediction = model.predict(processed_img)
    probability = prediction[0][0] # Probability of being PNEUMONIA (class 1)
    
    if probability > 0.5:
        # PNEUMONIA detected
        return "PNEUMONIA Detected", probability
    else:
        # NORMAL detected
        return "NORMAL (No Pneumonia)", 1.0 - probability # Return confidence in the NORMAL class

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("AI Chest X-Ray Pneumonia Detector 🩺")
st.markdown("Upload a chest X-ray image (JPG/PNG) to receive an instant diagnosis from the trained model.")

uploaded_file = st.file_uploader(
    "Choose a Chest X-Ray image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded X-Ray', use_column_width=True)
    st.subheader("Analysis Results:")
    
    # Process the image and get prediction
    with st.spinner('Analyzing image with VGG16 model...'):
        try:
            processed_img = preprocess_image(uploaded_file, IMAGE_SIZE)
            result, confidence = predict_pneumonia(processed_img)
            
            # Display the result
            st.markdown(f"**Diagnosis:** <span style='font-size: 24px;'>{result}</span>", unsafe_allow_html=True)
            
            # Use color-coding for feedback
            if "PNEUMONIA" in result:
                st.error(f"🚨 **Confidence in Pneumonia:** {confidence:.2f}")
            else:
                st.success(f"✅ **Confidence in Normal:** {confidence:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction: Check the image format or model integrity. Error: {e}")
