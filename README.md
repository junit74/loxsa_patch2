# Product Image Classification System

A TensorFlow-based image classification system that distinguishes between good and defective products using images.

## Features

- **Deep Learning Model**: Uses a CNN architecture with TensorFlow/Keras for image classification
- **Data Preprocessing**: Includes resizing, normalization, and augmentation for robust training
- **Inspection Screen**: Interactive UI for testing new images and visualizing feature maps
- **Evaluation Tools**: Complete metrics evaluation with accuracy, precision, recall, and confusion matrix

## Project Structure

```
.
├── config.py               # Configuration settings
├── data_preprocessing.py   # Image loading and preprocessing utilities
├── model.py                # CNN model creation and training
├── inspection_screen.py    # User interface for testing
├── evaluate.py             # Model evaluation tools
├── main.py                 # Main entry point
├── requirements.txt        # Package dependencies
├── README.md               # This file
└── data/                   # Directory for storing image data
    ├── good/               # Good product images
    └── defective/          # Defective product images
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up the directory structure:

```bash
python main.py --setup
```

3. Add images:
   - Place good product images in the `data/good/` directory
   - Place defective product images in the `data/defective/` directory
   - You can also use symbolic links to image directories with `ln -s`

## Usage

### Setup

Set up the directory structure:

```bash
python main.py --setup
```

### Training

Train the model on your image data:

```bash
python main.py --train
```

### Evaluation

Evaluate the trained model's performance:

```bash
python main.py --evaluate
```

### Inspection Screen

Launch the interactive inspection screen to test new images. You can use one of two options:

1. Tkinter-based UI (might have issues on some macOS systems):
```bash
python main.py --inspect
```

2. Streamlit-based UI (web interface, more reliable):
```bash
streamlit run streamlit_app.py
```

### Reset Model

Reset the model by deleting the model file and related files:

```bash
python main.py --reset
```

### All-in-One

Run all steps (setup, train, evaluate, and launch inspection screen):

```bash
python main.py --all
```

## Custom Paths

You can specify custom paths for your data and model:

```bash
python main.py --train --data_path /path/to/custom/data --model_path /path/to/save/model.h5
```

## Inspection Screen

The inspection screen provides:

1. Image upload functionality
2. Classification results with confidence scores
3. Feature map visualization to understand the model's decision-making
4. Simple and intuitive interface

## Model Architecture

The system uses a 4-layer CNN with:
- Convolutional layers with increasing filter counts (32, 64, 128, 256)
- Batch normalization for training stability
- MaxPooling for dimension reduction
- Dropout for regularization
- Binary classification output with sigmoid activation

## Evaluation Metrics

The system evaluates using:
- Accuracy
- Precision and Recall
- ROC curve and AUC
- Confusion matrix
- Visual examples of correct and incorrect predictions