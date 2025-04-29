"""
Module for loading and preprocessing image data.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from config import *

def create_data_generators():
    """
    Create train and validation data generators with augmentation.
    
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Create data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=VALIDATION_SPLIT
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        IMAGE_DATA_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        follow_links=True    # Follow symbolic links
    )
    
    # Validation generator
    validation_generator = validation_datagen.flow_from_directory(
        IMAGE_DATA_PATH,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        follow_links=True    # Follow symbolic links
    )
    
    return train_generator, validation_generator

def visualize_augmentation(image_path):
    """
    Visualize data augmentation effects on a sample image.
    
    Args:
        image_path (str): Path to the sample image
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 3, 1)
    plt.imshow(img_array[0] / 255.0)
    plt.title("Original")
    
    i = 2
    for batch in datagen.flow(img_array, batch_size=1):
        plt.subplot(3, 3, i)
        plt.imshow(batch[0] / 255.0)
        plt.title(f"Augmented {i-1}")
        i += 1
        if i > 9:
            break
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.show()

def count_images_recursively(directory):
    """
    Count all image files in a directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        int: Number of image files
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file.lower())[1] in image_extensions:
                count += 1
                
    return count

def check_data_distribution():
    """
    Check and print the distribution of images in each class, recursively searching subdirectories.
    """
    good_dir = os.path.join(IMAGE_DATA_PATH, 'good')
    defective_dir = os.path.join(IMAGE_DATA_PATH, 'defective')
    
    if not os.path.exists(good_dir) or not os.path.exists(defective_dir):
        print(f"Error: Data directories not found at {IMAGE_DATA_PATH}")
        return 0, 0
    
    good_images = count_images_recursively(good_dir)
    defective_images = count_images_recursively(defective_dir)
    
    print(f"Data distribution (including subdirectories):")
    print(f"- Good products: {good_images} images")
    print(f"- Defective products: {defective_images} images")
    print(f"- Total: {good_images + defective_images} images")
    
    # Create bar chart of class distribution
    plt.figure(figsize=(8, 5))
    plt.bar(['Good', 'Defective'], [good_images, defective_images], color=['green', 'red'])
    plt.title('Class Distribution')
    plt.ylabel('Number of Images')
    plt.savefig('class_distribution.png')
    
    return good_images, defective_images