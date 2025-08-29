"""
File: train_cnn.py
Purpose: Train a convolutional neural network (CNN) from scratch for bloodstain pattern classification
         using a small image dataset. Includes data augmentation, class weighting, L2 regularization,
         and early stopping to improve generalization on a limited dataset.

Dependencies:
    - Python 3.x
    - TensorFlow / Keras
    - NumPy
    - Pandas
    - Matplotlib
    - scikit-learn

Outputs:
    - Trained CNN model
    - Training and validation accuracy/loss plots
    - Test accuracy evaluation

See README.md for full project overview and dataset details.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# -------------------------------
# Paths and parameters
# -------------------------------
data_dir = "SIZE_120_rescaled_max_area_1024"
img_size = (120, 120)
batch_size = 8
test_ratio = 0.1
val_ratio = 0.2

# -------------------------------
# Step 1: Build dataframe with file paths and labels
# -------------------------------
filepaths = []
labels = []

for cls in os.listdir(data_dir):
    cls_path = os.path.join(data_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    for f in os.listdir(cls_path):
        if f.endswith((".png", ".jpg", ".jpeg")):
            filepaths.append(os.path.join(cls_path, f))
            labels.append(cls)

df = pd.DataFrame({"filename": filepaths, "class": labels})

# -------------------------------
# Step 2: Split into train, val, test
# -------------------------------
train_val_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['class'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, stratify=train_val_df['class'], random_state=42)

# -------------------------------
# Step 3: Data generators
# -------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.9,1.1],
    rescale=1./255
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# -------------------------------
# Step 4: Compute class weights
# -------------------------------
y_train = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# -------------------------------
# Step 5: Build the CNN
# -------------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    MaxPool2D((2,2)),

    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(1e-5)),
    BatchNormalization(),
    MaxPool2D((2,2)),

    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(1e-5)),
    Dropout(0.1),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(1e-5))
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------------
# Step 6: Callbacks
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

# -------------------------------
# Step 7: Train the model
# -------------------------------
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=[early_stop, reduce_lr]
)

# -------------------------------
# Step 8: Evaluate on test set
# -------------------------------
test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)

# -------------------------------
# Step 9: Plot training history
# -------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
