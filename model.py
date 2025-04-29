"""
Module for creating and training the CNN model.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
from config import *

def create_model():
    """
    Create a CNN model for image classification.
    
    Returns:
        tf.keras.Model: The compiled model
    """
    model = models.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Fourth convolutional block
        layers.Conv2D(256, (1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(1, activation='sigmoid')  # Binary classification (good vs defective)
    ])
    
    # Compile the model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(train_generator, validation_generator):
    """
    Train the model and save it.
    
    Args:
        train_generator: Training data generator
        validation_generator: Validation data generator
    
    Returns:
        tuple: (model, history) - Trained model and training history
    """
    # Create model
    model = create_model()
    model.summary()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=LOGS_PATH,
        histogram_freq=1
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, tensorboard]
    )
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

def plot_training_history(history):
    """
    Plot training & validation accuracy and loss.
    
    Args:
        history: Training history
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_feature_maps(model, image_path):
    """
    Visualize feature maps of the model for a given image.
    
    Args:
        model: Trained model
        image_path: Path to the image to visualize
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    # Get feature maps from first four convolutional layers
    layer_outputs = [layer.output for layer in model.layers if 'conv2d' in layer.name][:4]
    feature_map_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
    feature_maps = feature_map_model.predict(img_array)
    
    # Plot feature maps
    for i, feature_map in enumerate(feature_maps):
        plt.figure(figsize=(15, 8))
        plt.suptitle(f'Feature Maps of Conv Layer {i+1}')
        
        # Plot a subset of feature maps (first 16 or less)
        n_features = min(16, feature_map.shape[3])
        for j in range(n_features):
            plt.subplot(4, 4, j+1)
            plt.imshow(feature_map[0, :, :, j], cmap='viridis')
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f'feature_maps_layer_{i+1}.png')
    
    # Original image for reference
    plt.figure(figsize=(6, 6))
    plt.imshow(img_array[0])
    plt.title('Original Image')
    plt.axis('off')
    plt.savefig('original_image.png')
    plt.show()