# -------------------------------
# BLOODSTAIN CLASSIFIER - OPTIMIZED TINY CNN
# -------------------------------
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random

# -------------------------------
# SEEDS FOR REPRODUCIBILITY
# -------------------------------
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# -------------------------------
# CONFIG
# -------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 8
EPOCHS = 200  # we will use early stopping
DATA_DIR = "SIZE_120_rescaled_max_area_1024"
CATEGORIES = ["120_blunt", "120_gun"]

# -------------------------------
# LOAD DATA
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

# -------------------------------
# TRAIN/VALIDATION SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=seed
)

# -------------------------------
# DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()  # validation is just rescaled

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# TINY CNN MODEL
# -------------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1],3)),
    MaxPooling2D((2,2)),
    Dropout(0.1),

    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.1),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

# -------------------------------
# COMPILE MODEL
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -------------------------------
# EVALUATE
# -------------------------------
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
