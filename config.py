"""
Configuration settings for the image classification system.
"""

# Paths
IMAGE_DATA_PATH = "data/"  # Path to image data directory
MODEL_SAVE_PATH = "models/product_classifier.h5"  # Path to save trained model
LOGS_PATH = "logs/"  # Path for TensorBoard logs

# Image preprocessing parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Model parameters
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5

# Class names
CLASS_NAMES = ['good', 'defective']