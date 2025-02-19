
#importing libraries
import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image 



model_path = os.path.abspath(os.path.join(os.getcwd(),  "Skin_Cancer.h5"))
#loading the model
model= tf.keras.models.load_model(model_path)


import pathlib
import numpy as np
import pandas as pd

# Define class labels and corresponding disease names
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
disease_info = {
    'akiec': ('Actinic Keratosis', 'It lies between malignant/benign and indicates pre-cancerous symptoms and is low risk.'),
    'bcc': ('Basal Cell Carcinoma', 'Variant of Malignant and Highly risky.'),
    'bkl': ('Benign Keratosis', 'Variant of Benign and not risky.'),
    'df': ('Dermatofibroma', 'Variant of Benign and not risky.'),
    'mel': ('Melanoma', 'Variant of Malignant, very serious and aggressive, and highly risky.'),
    'nv': ('Melanocytic Nevus', 'Variant of Benign and not risky.'),
    'vasc': ('Vascular Lesion', 'Variant of Benign and not risky.')
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model
    img_array /= 255.0  # Normalize
    return img_array

def predict_skin_cancer(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100  # Confidence percentage
    
    if confidence < 50:
        result = "No disease detected"
        disease_details = "The uploaded image does not indicate the presence of skin cancer."
    else:
        predicted_class = np.argmax(prediction)  # Get index of highest probability
        predicted_label = class_labels[predicted_class]
        disease_name, disease_details = disease_info[predicted_label]  # Get actual disease name and details
        result = disease_name
    
    return result, confidence, disease_details



st.set_page_config(
    page_title="Skin cancer Prediction App",
    page_icon="🧊",
    layout="wide"
)

# Sidebar Navigation
menu = ["Home", "Disease Prediction"]
choice = st.sidebar.selectbox("Select Activity", menu)

def header(url):
    st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)	

if choice == "Home":
    st.title("Skin Cancer Detection & Classification")
    st.write("""
    The broad classification of skin cancer is:
    
    **1) Malignant:**
    - It is a serious cancer, cancerous, has the ability to invade neighboring tissues and spread to other parts of the body.
    
    **2) Benign:**
    - Non-cancerous, doesn’t have the ability to invade neighboring tissues and doesn’t require cancer treatment.
    
    **Class Labels & Disease Names:**
    - **akiec**: Actinic Keratosis
    - **bcc**: Basal Cell Carcinoma
    - **bkl**: Benign Keratosis
    - **df**: Dermatofibroma
    - **mel**: Melanoma
    - **nv**: Melanocytic Nevus
    - **vasc**: Vascular Lesion
    
    Upload an image in the **Disease Prediction** section to check for skin cancer.
    """)
if choice == "Disease Prediction":
    st.title("Disease Prediction")
    st.write("Upload your skin image to check if there is a disease and identify the type of cancer.")
    
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(image_path, caption="Uploaded Image", width=300)
        st.write("")
        
        if st.button("Predict"):
            result, confidence, disease_details = predict_skin_cancer(image_path)
            st.write(f"**Prediction:** {result}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.write(f"**Details:** {disease_details}")
            
            # Display additional disease-specific information
            if result in disease_info:
                st.write(f"### Disease Information:")
                st.write(f"**{disease_info[result][0]}**")
                st.write(f"{disease_info[result][1]}")




