"""
Module for the inspection screen to test the model on new images.
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config import *
from model import visualize_feature_maps

class InspectionScreen:
    def __init__(self, model_path=MODEL_SAVE_PATH):
        """
        Initialize the inspection screen.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        
        # Load the trained model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # Setup the main window
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        self.root = tk.Tk()
        self.root.title("Product Inspection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Create main frames
        self.left_frame = tk.Frame(self.root, width=500, height=800, bg="#f0f0f0")
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH)
        
        self.right_frame = tk.Frame(self.root, width=700, height=800, bg="#f0f0f0")
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Left frame - Image selection and display
        self.setup_left_frame()
        
        # Right frame - Results and feature maps
        self.setup_right_frame()
    
    def setup_left_frame(self):
        """Set up the left frame with image selection and display."""
        # Title
        title_label = tk.Label(
            self.left_frame, 
            text="Product Image Inspection", 
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack(pady=10)
        
        # Button to upload image
        self.upload_button = tk.Button(
            self.left_frame,
            text="Upload Image",
            command=self.upload_image,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5
        )
        self.upload_button.pack(pady=10)
        
        # Frame for image display
        self.image_frame = tk.Frame(self.left_frame, bg="white", width=400, height=400)
        self.image_frame.pack(pady=10)
        
        # Label for image display
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Label for file path
        self.file_path_label = tk.Label(
            self.left_frame, 
            text="No image selected", 
            bg="#f0f0f0",
            wraplength=400
        )
        self.file_path_label.pack(pady=5)
        
        # Button to inspect image
        self.inspect_button = tk.Button(
            self.left_frame,
            text="Inspect Image",
            command=self.inspect_image,
            state=tk.DISABLED,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12),
            padx=10,
            pady=5
        )
        self.inspect_button.pack(pady=10)
    
    def setup_right_frame(self):
        """Set up the right frame with results and feature maps."""
        # Results section
        results_frame = tk.LabelFrame(
            self.right_frame, 
            text="Inspection Results", 
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        results_frame.pack(fill=tk.X, pady=10)
        
        # Result label
        self.result_label = tk.Label(
            results_frame,
            text="No inspection performed yet",
            font=("Arial", 14),
            bg="#f0f0f0",
            pady=10
        )
        self.result_label.pack()
        
        # Confidence score progress bar
        self.confidence_frame = tk.Frame(results_frame, bg="#f0f0f0")
        self.confidence_frame.pack(pady=10, fill=tk.X, padx=20)
        
        self.confidence_label = tk.Label(
            self.confidence_frame,
            text="Confidence: ",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.confidence_label.pack(side=tk.LEFT)
        
        self.confidence_value = tk.Label(
            self.confidence_frame,
            text="N/A",
            font=("Arial", 12),
            bg="#f0f0f0"
        )
        self.confidence_value.pack(side=tk.RIGHT)
        
        self.progress_canvas = tk.Canvas(
            results_frame, 
            width=400, 
            height=20, 
            bg="white",
            highlightthickness=1,
            highlightbackground="#999"
        )
        self.progress_canvas.pack(pady=10)
        
        # Feature maps section
        feature_maps_frame = tk.LabelFrame(
            self.right_frame, 
            text="Feature Maps Visualization", 
            font=("Arial", 12, "bold"),
            bg="#f0f0f0"
        )
        feature_maps_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Matplotlib figure for feature maps
        self.fig, self.axes = plt.subplots(2, 2, figsize=(8, 6))
        self.fig.suptitle("Key Feature Maps")
        
        # Clear all the axes and add placeholder text
        for ax in self.axes.flat:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No feature maps yet', 
                   horizontalalignment='center', 
                   verticalalignment='center',
                   transform=ax.transAxes)
        
        # Add the figure to the Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=feature_maps_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def upload_image(self):
        """Open file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        
        if file_path:
            try:
                # Display the selected image
                self.selected_image_path = file_path
                self.file_path_label.config(text=file_path)
                
                # Open and resize the image for display
                img = Image.open(file_path)
                img = img.resize((400, 400), Image.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(img)
                
                self.image_label.config(image=self.tk_img)
                self.inspect_button.config(state=tk.NORMAL)
                
                # Reset results
                self.result_label.config(text="No inspection performed yet")
                self.confidence_value.config(text="N/A")
                self.progress_canvas.delete("all")
                
                # Clear feature maps
                for ax in self.axes.flat:
                    ax.clear()
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'No feature maps yet', 
                           horizontalalignment='center', 
                           verticalalignment='center')
                self.canvas.draw()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")
    
    def inspect_image(self):
        """Inspect the selected image using the trained model."""
        if not self.model:
            messagebox.showerror("Error", "Model not loaded. Please check the model path.")
            return
            
        if not hasattr(self, 'selected_image_path'):
            messagebox.showerror("Error", "No image selected")
            return
            
        try:
            # Preprocess the image
            img = tf.keras.preprocessing.image.load_img(
                self.selected_image_path, 
                target_size=(IMG_HEIGHT, IMG_WIDTH)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = tf.expand_dims(img_array, 0)
            
            # Make prediction
            prediction = self.model.predict(img_array)[0][0]
            
            # Update result
            if prediction > 0.5:  # Flipped the logic based on your feedback
                result_text = "GOOD PRODUCT"
                result_color = "#4CAF50"
            else:
                result_text = "DEFECTIVE PRODUCT"
                result_color = "#F44336"
                
            confidence = max(prediction, 1 - prediction) * 100
            
            self.result_label.config(text=result_text, fg=result_color)
            self.confidence_value.config(text=f"{confidence:.1f}%")
            
            # Update progress bar
            self.progress_canvas.delete("all")
            self.progress_canvas.create_rectangle(
                0, 0, 
                confidence * 4, 20, 
                fill=result_color, 
                outline=""
            )
            
            # Generate and display feature maps
            self.generate_feature_maps(img_array)
            
        except Exception as e:
            messagebox.showerror("Error", f"Inspection failed: {e}")
    
    def generate_feature_maps(self, img_array):
        """
        Generate and display feature maps for the given image.
        
        Args:
            img_array: Preprocessed image array
        """
        # Get feature maps from convolutional layers
        layer_names = [layer.name for layer in self.model.layers if 'conv2d' in layer.name]
        
        if not layer_names:
            messagebox.showinfo("Info", "No convolutional layers found in the model")
            return
            
        # Limit to the first 4 convolutional layers or fewer if there are less
        layer_names = layer_names[:min(4, len(layer_names))]
        
        # Build a model that returns feature maps
        layer_outputs = [self.model.get_layer(name).output for name in layer_names]
        feature_map_model = tf.keras.Model(inputs=self.model.input, outputs=layer_outputs)
        
        # Get feature maps
        feature_maps = feature_map_model.predict(img_array)
        
        # Clear existing plots
        for ax in self.axes.flat:
            ax.clear()
            ax.axis('off')
            
        # Plot feature maps (first channel from each layer)
        for i, feature_map in enumerate(feature_maps):
            if i >= len(self.axes.flat):
                break
                
            # Get the first channel of the feature map
            self.axes.flat[i].imshow(feature_map[0, :, :, 0], cmap='viridis')
            self.axes.flat[i].set_title(f"Layer {layer_names[i]}")
            self.axes.flat[i].axis('off')
            
        self.fig.suptitle("Key Feature Maps", fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw()
    
    def run(self):
        """Run the inspection screen application."""
        if self.model is None:
            messagebox.showwarning(
                "Warning", 
                f"Model not found at {self.model_path}. Some features may not work."
            )
        self.root.mainloop()


def main():
    """Main function to run the inspection screen."""
    inspection_app = InspectionScreen()
    inspection_app.run()


if __name__ == "__main__":
    main()