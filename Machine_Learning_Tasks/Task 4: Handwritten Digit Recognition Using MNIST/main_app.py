import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("D:/Data_Science/Final_Capstone_Project/Deep_learning--fashion-mnist-style-detector/clothing_model.h5")

# Class labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

st.set_page_config(page_title="Fashion MNIST Style Detector", layout="wide")

st.title("👗 Fashion MNIST Style Detector")
st.write("Upload an image (28x28 grayscale) to predict its fashion category!")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess image
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"👕 Predicted Category: **{class_names[predicted_label]}**")
    st.write(f"🧭 Confidence: {confidence:.2f}")

    st.progress(int(confidence * 100))
else:
    st.info("Please upload a fashion image to start prediction.")
