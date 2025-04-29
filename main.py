"""
Main module to run the image classification system.
"""

import os
import argparse
from data_preprocessing import create_data_generators, check_data_distribution, visualize_augmentation
from model import train_model, visualize_feature_maps
from evaluate import evaluate_model, load_model, create_test_generator
from inspection_screen import InspectionScreen
from config import *

def setup_directories():
    """
    Set up necessary directories for the project.
    """
    os.makedirs(IMAGE_DATA_PATH, exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DATA_PATH, 'good'), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_DATA_PATH, 'defective'), exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)
    
    print(f"Created directory structure:")
    print(f"- Image data: {IMAGE_DATA_PATH}")
    print(f"  - Good products: {os.path.join(IMAGE_DATA_PATH, 'good')}")
    print(f"  - Defective products: {os.path.join(IMAGE_DATA_PATH, 'defective')}")
    print(f"- Model save path: {MODEL_SAVE_PATH}")
    print(f"- Logs: {LOGS_PATH}")
    print(f"\nNote: You can use symbolic links with 'ln -s' to link to image directories.")

def train():
    """
    Train the model using the training data.
    """
    print("\n--- Starting Training Process ---")
    
    # Check data distribution
    good_count, defective_count = check_data_distribution()
    
    if good_count == 0 or defective_count == 0:
        print("\nERROR: Both classes must have images. Please add images to the data directory.")
        return
    
    # Create data generators
    print("\nCreating data generators...")
    train_generator, validation_generator = create_data_generators()
    
    # Train the model
    print("\nTraining model...")
    model, history = train_model(train_generator, validation_generator)
    
    print(f"\nModel trained successfully and saved to {MODEL_SAVE_PATH}")
    
    return model

def evaluate(model=None):
    """
    Evaluate the trained model.
    
    Args:
        model: The trained model (will be loaded from file if None)
    """
    print("\n--- Starting Evaluation Process ---")
    
    if model is None:
        model = load_model()
        
    if model is None:
        print(f"ERROR: Could not load model from {MODEL_SAVE_PATH}")
        return
    
    # Create test generator
    test_generator = create_test_generator()
    
    # Evaluate model
    evaluate_model(model, test_generator)

def inspect():
    """
    Launch the inspection screen.
    """
    print("\n--- Launching Inspection Screen ---")
    app = InspectionScreen()
    app.run()

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Product Image Classification System')
    
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--inspect', action='store_true', help='Launch inspection screen')
    parser.add_argument('--setup', action='store_true', help='Set up directory structure')
    parser.add_argument('--all', action='store_true', help='Run all steps (setup, train, evaluate, inspect)')
    parser.add_argument('--reset', action='store_true', help='Reset the model (delete existing model file)')
    
    parser.add_argument('--data_path', type=str, help=f'Path to image data directory (default: {IMAGE_DATA_PATH})')
    parser.add_argument('--model_path', type=str, help=f'Path to save/load model (default: {MODEL_SAVE_PATH})')
    
    return parser.parse_args()

def reset_model():
    """
    Reset the model by deleting the model file.
    """
    print("\n--- Resetting Model ---")
    
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            os.remove(MODEL_SAVE_PATH)
            print(f"Model file deleted successfully: {MODEL_SAVE_PATH}")
        except Exception as e:
            print(f"Error deleting model file: {e}")
    else:
        print(f"No model file found at {MODEL_SAVE_PATH}")
    
    # Also check for related files
    model_dir = os.path.dirname(MODEL_SAVE_PATH)
    if os.path.exists(model_dir):
        for filename in os.listdir(model_dir):
            if filename.endswith('.png') or filename.endswith('.h5'):
                try:
                    file_path = os.path.join(model_dir, filename)
                    os.remove(file_path)
                    print(f"Related file deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

def main():
    """
    Main function to run the image classification system.
    """
    args = parse_arguments()
    
    # Update paths from command line if provided
    global IMAGE_DATA_PATH, MODEL_SAVE_PATH
    if args.data_path:
        IMAGE_DATA_PATH = args.data_path
    if args.model_path:
        MODEL_SAVE_PATH = args.model_path
    
    # Reset model if requested
    if args.reset:
        reset_model()
        return
    
    # Run the requested operations
    if args.all or args.setup:
        setup_directories()
    
    model = None
    if args.all or args.train:
        model = train()
    
    if args.all or args.evaluate:
        evaluate(model)
    
    if args.all or args.inspect:
        inspect()
    
    # If no action specified, show help
    if not (args.all or args.setup or args.train or args.evaluate or args.inspect or args.reset):
        print("No action specified. Launching inspection screen by default.")
        inspect()


if __name__ == "__main__":
    main()