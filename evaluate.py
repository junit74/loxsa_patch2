"""
Module for evaluating the trained model.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from config import *

def load_model(model_path=MODEL_SAVE_PATH):
    """
    Load the trained model.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tf.keras.Model: Loaded model
    """
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_test_generator(test_data_path=None):
    """
    Create a data generator for the test dataset.
    
    Args:
        test_data_path (str): Path to test data. If None, uses the same path as training data.
        
    Returns:
        tf.keras.preprocessing.image.DirectoryIterator: Test data generator
    """
    if test_data_path is None:
        test_data_path = IMAGE_DATA_PATH
        
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,  # Important for maintaining order for confusion matrix
        follow_links=True   # Follow symbolic links
    )
    
    return test_generator

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data.
    
    Args:
        model (tf.keras.Model): The trained model
        test_generator: Test data generator
        
    Returns:
        tuple: (test_loss, test_accuracy)
    """
    # Get the model's predictions
    predictions = model.predict(test_generator)
    
    # Convert predictions to binary classes
    predicted_classes = (predictions > 0.5).astype(int)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Calculate metrics
    loss, accuracy, precision, recall = model.evaluate(test_generator)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    
    # Create and plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES))
    
    # Plot ROC curve
    plot_roc_curve(true_classes, predictions)
    
    # Create examples of correct and incorrect predictions
    plot_prediction_examples(test_generator, predicted_classes, true_classes, predictions)
    
    return loss, accuracy

def plot_roc_curve(y_true, y_pred):
    """
    Plot ROC curve for the model.
    
    Args:
        y_true: True binary labels
        y_pred: Target scores (probability estimates)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')

def plot_prediction_examples(test_generator, pred_classes, true_classes, probabilities, num_examples=5):
    """
    Plot examples of correct and incorrect predictions.
    
    Args:
        test_generator: Test data generator
        pred_classes: Predicted classes
        true_classes: True classes
        probabilities: Prediction probabilities
        num_examples: Number of examples to show for each category
    """
    # Reset generator to start from the beginning
    test_generator.reset()
    
    # Get all images and labels
    images = []
    
    for i, (batch_images, _) in enumerate(test_generator):
        images.append(batch_images)
        if len(images) * BATCH_SIZE >= len(true_classes) or i >= len(test_generator) - 1:
            break
    
    # Combine all batches into a single array
    all_images = np.vstack(images)
    
    # Find indices of correct and incorrect predictions
    correct_indices = np.where(pred_classes.flatten() == true_classes)[0]
    incorrect_indices = np.where(pred_classes.flatten() != true_classes)[0]
    
    # Plot correct predictions
    plt.figure(figsize=(15, 5))
    plt.suptitle("Correct Predictions", fontsize=16)
    
    for i in range(min(num_examples, len(correct_indices))):
        idx = correct_indices[i]
        if idx >= len(all_images):
            continue
            
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(all_images[idx])
        plt.title(f"True: {CLASS_NAMES[true_classes[idx]]}\nPred: {probabilities[idx][0]:.2f}")
        plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('correct_predictions.png')
    
    # Plot incorrect predictions
    if len(incorrect_indices) > 0:
        plt.figure(figsize=(15, 5))
        plt.suptitle("Incorrect Predictions", fontsize=16)
        
        for i in range(min(num_examples, len(incorrect_indices))):
            idx = incorrect_indices[i]
            if idx >= len(all_images):
                continue
                
            plt.subplot(1, num_examples, i + 1)
            plt.imshow(all_images[idx])
            plt.title(f"True: {CLASS_NAMES[true_classes[idx]]}\nPred: {probabilities[idx][0]:.2f}")
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('incorrect_predictions.png')

def main():
    """Main function to evaluate the model."""
    model = load_model()
    if model:
        test_generator = create_test_generator()
        evaluate_model(model, test_generator)


if __name__ == "__main__":
    main()