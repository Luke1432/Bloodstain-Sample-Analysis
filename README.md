# Bloodstain Sample Analysis

This repository presents a comprehensive approach to bloodstain pattern analysis using deep learning techniques. By leveraging convolutional neural networks (CNNs), the project aims to classify bloodstain images into two categories: "blunt" and "gunshot." The methodology encompasses data preparation, model training, evaluation, and visualization.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Visualization](#visualization)
- [Usage](#usage)

## Overview

Bloodstain pattern analysis is a critical component in forensic investigations, aiding in reconstructing crime scenes. This project utilizes deep learning to automate the classification of bloodstain images, distinguishing between patterns resulting from blunt force trauma and gunshot wounds.

## Project Structure
```bash
Bloodstain-Sample-Analysis/
│
├── SIZE_120_rescaled_max_area_1024/ # Image dataset directory
│
├── .vscode/ # VS Code settings
│
├── pycache/ # Compiled Python files
│
├── Figure_1.png # Sample output image
│
├── cnn_bloodstain_gun_blunt_best_v1.h5 # Trained model file
│
├── cnn_bloodstain_gun_blunt_weights_best_v1.h5 # Model weights
│
├── train_cnn.py # Script for training the model and evaluating performance
│
├── visualize_bloodstain_images.ipynb # Jupyter notebook for visualization

## Data Preparation

The dataset comprises images categorized into two folders: `120_blunt` and `120_gun`. Each image is processed to:

- Resize to 120x120 pixels
- Normalize pixel values to the range [0, 1]
- Augment the dataset using techniques like rotation, shifting, and zooming

Class weights are computed to address class imbalance, ensuring the model treats both classes equally during training.

## Model Architecture

The model is a CNN built using TensorFlow/Keras, featuring:

- Convolutional layers with ReLU activation
- MaxPooling layers
- Batch normalization
- Dropout for regularization
- L2 regularization to prevent overfitting

The final output layer uses a sigmoid activation function for binary classification.

## Training and Evaluation

The model is trained using:

- Adam optimizer with a learning rate of 0.0001
- Binary cross-entropy loss function
- Accuracy as the evaluation metric

Training employs early stopping and learning rate reduction callbacks to optimize performance. Evaluation is performed on a test set within the `train_cnn.py` script.

## Visualization

Training and validation accuracy and loss are plotted to visualize the model's performance over epochs. This helps in diagnosing issues like overfitting or underfitting.

## Usage

To replicate the analysis:

1. Clone the repository:

   ```bash
   git clone https://github.com/Luke1432/Bloodstain-Sample-Analysis.git
   cd Bloodstain-Sample-Analysis

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the training and evaluation script:
   ```bash
   python train_cnn.py

4. Check the output plots and test accuracy printed in the console. 

