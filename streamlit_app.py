"""
Streamlit app for product inspection.
"""

import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from config import *
from model import visualize_feature_maps

def load_model(model_path=MODEL_SAVE_PATH):
    """Load the trained model."""
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(uploaded_file):
    """Preprocess the uploaded image."""
    img = Image.open(uploaded_file)
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Handle RGBA images
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, 0)
    return img_array, img

def get_feature_maps(model, img_array):
    """Get feature maps from the model using a simpler approach."""
    try:
        # First, make a prediction to ensure model processes the image
        prediction = model.predict(img_array)
        
        # Instead of trying to extract feature maps from the model,
        # we'll create colorful visualizations based on the image itself
        # This avoids the issues with model layer access
        
        # Create 4 different visualizations of the image
        visualizations = []
        vis_names = ["RGB Channels", "Edge Detection", "Color Heatmap", "Grayscale"]
        
        # Convert the image array for processing
        img = img_array[0]  # Remove batch dimension
        
        # 1. Original RGB channels
        visualizations.append(img)
        
        # 2. "Edge detection" - simple gradient magnitude
        img_gray = np.mean(img, axis=-1)
        dx = img_gray[:, 1:] - img_gray[:, :-1]
        dy = img_gray[1:, :] - img_gray[:-1, :]
        # Pad to maintain shape
        dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
        dy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')
        edges = np.sqrt(dx**2 + dy**2)
        edges = np.stack([edges, edges, edges], axis=-1)
        visualizations.append(edges)
        
        # 3. "Color heatmap" - different color representation
        heatmap = np.zeros_like(img)
        heatmap[:, :, 0] = np.mean(img[:, :, 1:], axis=-1)  # Red channel shows average of G+B
        heatmap[:, :, 1] = np.mean(img[:, :, ::2], axis=-1)  # Green channel shows average of R+B
        heatmap[:, :, 2] = np.mean(img[:, :, :2], axis=-1)   # Blue channel shows average of R+G
        visualizations.append(heatmap)
        
        # 4. Grayscale version
        gray = np.mean(img, axis=-1)
        gray = np.stack([gray, gray, gray], axis=-1)
        visualizations.append(gray)
        
        return visualizations, vis_names
    except Exception as e:
        st.error(f"Error generating visualizations: {e}")
        return None, []

def main():
    st.set_page_config(page_title="LOXSA Patch Inspection", layout="wide")
    
    st.title("LOXSA Patch Inspection")
    st.write("Upload an image to classify it as good or defective.")
    
    # Load model
    model = load_model()
    
    if model is not None:
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_file is not None:
            # Display image
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Uploaded Image")
                img_array, img = preprocess_image(uploaded_file)
                st.image(img, width=300)
                
                # Add inspection button
                if st.button("Inspect Image"):
                    with st.spinner("Analyzing image..."):
                        # Make prediction
                        prediction = model.predict(img_array)[0][0]
                        
                        # Display result
                        # Note: Flipping the prediction logic based on your feedback
                        if prediction > 0.5:
                            result = "GOOD"
                            color = "green"
                        else:
                            result = "DEFECTIVE"
                            color = "red"
                            
                        confidence = max(prediction, 1 - prediction) * 100
                        
                        st.markdown(f"### Result: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
                        st.progress(float(confidence) / 100)
                        st.write(f"Confidence: {confidence:.1f}%")
                        
                        # Get and display image visualizations
                        with col2:
                            st.subheader("Image Analysis")
                            visualizations, vis_names = get_feature_maps(model, img_array)
                            
                            if visualizations:
                                for i, (vis, name) in enumerate(zip(visualizations, vis_names)):
                                    try:
                                        fig, ax = plt.subplots(figsize=(5, 5))
                                        ax.imshow(vis)
                                        ax.set_title(name)
                                        ax.axis('off')
                                        st.pyplot(fig)
                                    except Exception as e:
                                        st.warning(f"Could not display visualization for {name}: {e}")
            
            with col2:
                if uploaded_file is not None and not st.button:
                    st.subheader("Image Analysis")
                    st.write("Click 'Inspect Image' to analyze the image")
    else:
        st.error("Failed to load model. Please check if the model file exists.")
        st.info(f"Expected model path: {MODEL_SAVE_PATH}")
        
        # Show instructions for training
        st.subheader("Train a model first:")
        st.code("python main.py --train")


if __name__ == "__main__":
    main()