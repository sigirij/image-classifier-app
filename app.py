import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os # Add this import

st.warning("Debugging File Path:")
st.write("Current Directory:", os.getcwd())
st.write("Files here:", os.listdir("."))

# 1. Cache the model load so it doesn't crash your app's memory
@st.cache_resource
def load_classifier():
    # Change this line to match the exact file in your repository
    return tf.keras.models.load_model("mobilenetv2_model.h5")

st.title("AI Image Classification App (MobileNetV2)")

# 2. Safely load the model and display the exact error if it fails
try:
    model = load_classifier()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.info("Check your Streamlit Cloud logs or verify your TensorFlow version in requirements.txt.")
    st.stop() # Stops the rest of the app from running and crashing

class_names = ['class1', 'class2', 'class3']  # update from training

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB') # 3. Ensure image has 3 channels (no alpha channel from PNGs)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Standard MobileNetV2 preprocessing (scales pixels from 0-255 to -1 to 1)
    # Note: If your custom model already has a Rescaling layer built inside it, you can delete this line.
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: {predicted_class}")
    
    # Optional: Display confidence scores for debugging
    st.write("Confidence Scores:", prediction)
