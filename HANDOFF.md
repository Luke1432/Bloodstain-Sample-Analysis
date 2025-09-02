# ﻿Bloodstain Sample Analysis – Transition & Handoff Document
## Project Overview
### Project Purpose:
 The Bloodstain Sample Analysis project automates classification of bloodstain images into blunt-force impact and gunshot categories using a Convolutional Neural Network (CNN). This system reduces bias present in traditional forensic analysis and provides reproducible, scalable results for crime scene reconstruction.
Project Goals:
* Train a CNN to classify bloodstain images with high accuracy.

* Apply data augmentation and class weighting to handle limited and imbalanced datasets.

* Implement regularization and early stopping to prevent overfitting.

* Evaluate the model on a held-out test set and provide visual feedback.

Key Deliverables:
   * train_cnn.py – main training script.

   * Trained CNN model (cnn_bloodstain_gun_blunt_best_v1.h5).

   * Dataset folder (SIZE_120_rescaled_max_area_1024/).

   * Sample plots (Figure_1.png, sampleOutput3.png).

   * Documentation (README.md, HANDOFF.md).

Project Importance:
 Accurate classification of bloodstain patterns aids forensic investigators in determining the nature of injuries and weapons used, providing critical evidence for legal proceedings.


## Deliverables and File Descriptions
### Dataset
      * Location: SIZE_120_rescaled_max_area_1024/

      * Structure: Two subfolders (blunt, gunshot) containing .png and .jpg images.

#### Usage:
         * Automatically split into training, validation, and test sets by train_cnn.py.

         * Images should remain in original folder structure; new images should be added to appropriate subfolders.

#### Next Steps:
            * Expand the dataset to improve model accuracy.

            * Ensure image size consistency.



### Training Script (train_cnn.py)
Purpose: Handles preprocessing, model creation, training, validation, and evaluation.
Key Features:
               * Data augmentation: rotation, zoom, shift, brightness adjustment

               * L2 regularization and Dropout

               * Early stopping and learning rate reduction

               * Balanced class weighting



## Usage Instructions:
python train_cnn.py


Outputs:
                  * Trained CNN (.h5)

                  * Training/validation accuracy and loss plots

                  * Test set evaluation metrics

Operational Notes:
                     * Hyperparameters (batch size, learning rate, dropout) can be adjusted in the script.

                     * Monitor validation metrics to detect overfitting.

### Trained CNN Model
                        * File: cnn_bloodstain_gun_blunt_best_v1.h5

                        * Purpose: Ready-to-use for inference on new bloodstain images.

                        * Usage:

from tensorflow.keras.models import load_model
model = load_model('cnn_bloodstain_gun_blunt_best_v1.h5')


                           * Next Steps: Integrate into operational pipelines or retrain after dataset expansion
### Visualizations
                           * File: Figure_1.png

                           * Shows training/validation accuracy and loss over epochs.

                           * Usage: Provides quick insights into model performance.

                           * Next Steps: Update with new plots after retraining.



## Documentation
                              * README.md: High-level project overview and setup instructions.

                              * HANDOFF.md: Explains file usage, operational guidance, and next steps.

## Environment & Dependencies
Python Version: 3.x
Libraries Required:
                                 * TensorFlow / Keras

                                 * NumPy

                                 * Pandas

                                 * Matplotlib

                                 * scikit-learn

Setup Instructions:
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt


Next Steps:
                                    * Maintain requirements.txt for reproducibility.



## Workflow & Operational Guidelines
### Training & Evaluation Flowchart
Dataset (SIZE_120_rescaled_max_area_1024/)
         |
         v
Preprocessing & Augmentation (ImageDataGenerator)
         |
         v
CNN Model Training (train_cnn.py)
         |
         v
Validation Monitoring (accuracy/loss)
         |
         v
Early Stopping / LR Reduction
         |
         v
Final Model Save (.h5) & Test Evaluation
         |
         v
Plot Results (accuracy/loss curves)




### Operational Guidelines
                                       * Class Weighting: Ensures balanced learning on imbalanced datasets.

                                       * L2 Regularization & Dropout: Reduces overfitting on small datasets.

                                       * Early Stopping: Monitors validation loss to restore best weights.

                                       * Test Set Evaluation: Measures generalization on unseen data.

Knowledge Transfer Notes:
                                          * Team members should understand CNN architecture, data preprocessing, and flow_from_dataframe usage.

                                          * Maintain dataset versioning for reproducibility.



## Suggested Next Steps
                                             1. Operational Deployment: Integrate the trained model into forensic workflows.

                                             2. Dataset Expansion: Add new images to improve accuracy and generalization.

                                             3. Monitoring & Maintenance: Track model performance on new data.

                                             4. Documentation Updates: Keep README and HANDOFF.md current.

                                             5. Enhancements: Explore transfer learning or explainable AI techniques (e.g., Grad-CAM).



## Access & Login Information
                                                * GitHub Repository: https://github.com/Luke1432/Bloodstain-Sample-Analysis

                                                * Dataset: Included in repository; no login required.

                                                * Python Environment: Local virtual environment; no external logins needed.

                                                * Optional Cloud GPU Access: Credentials for Colab, AWS, or Azure must be provided separately.



























7. Appendix (Screenshots / Examples)
                                                   1. Directory Structure Screenshot
  


SIZE_120_rescaled_max_area_1024 contents: 
  















                                                      2. Example Training Output
