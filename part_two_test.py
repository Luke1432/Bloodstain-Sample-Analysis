from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# --- CONFIGURATION ---
EPOCHS = 30
BATCH_SIZE = 32  # smaller batch size to handle small datasets safely
IMG_SIZE = (128, 128)
DATA_DIR = "SIZE_120_rescaled_max_area_1024"

# --- DATA AUGMENTATION PIPELINE ---
# Artificially expands dataset to improve generalization
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomContrast(0.2),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# --- LOAD AND SPLIT DATA ---
# 80% training, 20% validation
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- COMPUTE CLASS WEIGHTS ---
# Balances training if dataset is imbalanced
class_names = train_ds.class_names
num_classes = len(class_names)
print("Class Names:", class_names)

all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(int(labels.numpy()))
counts = Counter(all_labels)
total = sum(counts.values())
class_weight = {i: total / (len(class_names) * counts[i]) for i in range(len(class_names))}
print("Class Weights:", class_weight)

# --- NORMALIZATION AND AUGMENTATION ---
normalizer = layers.Rescaling(1./255)

if num_classes == 2:
    # Cast labels to float32 for binary_crossentropy
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(normalizer(x)), tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: (normalizer(x), tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
else:
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(normalizer(x)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: (normalizer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

# --- PREFETCH FOR EFFICIENCY ---
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# --- MODEL ARCHITECTURE ---
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
])

# --- ADD OUTPUT LAYER AND COMPILE ---
if num_classes == 2:
    model.add(layers.Dense(1, activation="sigmoid"))  # binary output
    loss_fn = "binary_crossentropy"
else:
    model.add(layers.Dense(num_classes, activation="softmax"))  # multi-class
    loss_fn = "sparse_categorical_crossentropy"

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

# --- CALLBACKS ---
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# --- TRAINING ---
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# --- PLOTTING RESULTS ---
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(train_acc) + 1)

plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss', marker='o')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
