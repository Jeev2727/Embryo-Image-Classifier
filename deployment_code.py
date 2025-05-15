import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from PIL import Image

# Load the trained model
MODEL_PATH = "/Users/mandalajeevan/Downloads/embryo_classification_model_finetuned.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (modify based on your training labels)
class_labels = [
    "8Cell_A", "8Cell_B", "8Cell_C", "Blastocyst_A", "Blastocyst_B", 
    "Blastocyst_C", "Morula_A", "Morula_B", "Morula_C"
]

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)
    return img_array

# Set page configuration
st.set_page_config(
    page_title="Embryo Classification",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            background-color: #f0f2f6;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2C3E50;
        }
        .subheader {
            text-align: center;
            font-size: 20px;
            color: #003300;
        }
        .uploaded-img {
            text-align: center;
            colour: #333366;
        }
        .result-box {
            padding: 15px;
            background-color: #2ECC71;
            color: white;
            text-align: center;
            font-size: 20px;
            border-radius: 10px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# UI Layout
st.markdown("<p class='main-title'>🧬 Embryo Classification App</p>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Upload an embryo image to classify its type</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    predicted_class_name = class_labels[predicted_class_index]
    confidence_score = predictions[0][predicted_class_index] * 100  # Convert to %

    # Display results
    st.markdown(f"<div class='result-box'>🔍 Prediction: {predicted_class_name} <br> 🎯 Confidence: {confidence_score:.2f}%</div>", unsafe_allow_html=True)

    # Extra info
    st.success("✅ Classification Complete!")
