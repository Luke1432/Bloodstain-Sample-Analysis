"""
bloodstain_cnn.py

Description:
------------
This script implements a final optimized tiny CNN for classifying bloodstain images 
into two categories: "120_blunt" and "120_gun". It is designed to train on a very 
small dataset (~94 images) using Keras. The script includes:

- Data loading and preprocessing
- Data augmentation
- Tiny CNN architecture with Dropout
- Early stopping and learning rate reduction callbacks
- Training and validation evaluation
- Optional smoothed plots of training curves

Note:
-----
Given the extremely small dataset, training a CNN from scratch is limited by data 
availability. Accuracy may remain modest (~50-60%) due to the inherent data scarcity.
"""

# -------------------------------
# IMPORTS
# -------------------------------
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# -------------------------------
# REPRODUCIBILITY SETTINGS
# -------------------------------
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 200
DATA_DIR = "SIZE_120_rescaled_max_area_1024"
CATEGORIES = ["120_blunt", "120_gun"]

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def smooth_curve(values, factor=0.8):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * factor + (1 - factor) * v
        smoothed.append(last)
    return smoothed

# -------------------------------
# DATA LOADING AND PREPROCESSING
# -------------------------------
images, labels = [], []

for idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATA_DIR, category)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(idx)

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=seed
)

# -------------------------------
# DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=20,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# COMPUTE CLASS WEIGHTS
# -------------------------------
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights))

# -------------------------------
# TINY CNN MODEL DEFINITION
# -------------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', padding='same', 
           input_shape=(IMG_SIZE[0], IMG_SIZE[1],3), 
           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(32, (3,3), activation='relu', padding='same', 
           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation='relu', padding='same', 
           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    MaxPooling2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# -------------------------------
# COMPILE MODEL
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # smaller LR
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=1
)

# -------------------------------
# EVALUATE MODEL
# -------------------------------
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

# -------------------------------
# PLOT TRAINING CURVES
# -------------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc', alpha=0.3)
plt.plot(history.history['val_accuracy'], label='Val Acc', alpha=0.3)
plt.plot(smooth_curve(history.history['accuracy']), label='Train Acc (smoothed)')
plt.plot(smooth_curve(history.history['val_accuracy']), label='Val Acc (smoothed)')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', alpha=0.3)
plt.plot(history.history['val_loss'], label='Val Loss', alpha=0.3)
plt.plot(smooth_curve(history.history['loss']), label='Train Loss (smoothed)')
plt.plot(smooth_curve(history.history['val_loss']), label='Val Loss (smoothed)')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------------
# VISUALIZE PREDICTIONS
# -------------------------------
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

plt.figure(figsize=(12,8))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(X_val[i])
    true_label = CATEGORIES[y_val[i]]
    pred_label = CATEGORIES[y_pred_classes[i]]
    color = "green" if true_label == pred_label else "red"
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.show()
