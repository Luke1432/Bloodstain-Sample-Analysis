# -------------------------------
# BLOODSTAIN CLASSIFIER - ENHANCED CNN
# -------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 200
DATA_DIR = "SIZE_120_rescaled_max_area_1024"
CATEGORIES = ["120_blunt", "120_gun"]

# -------------------------------
# HELPER FUNCTION - SMOOTH CURVES
# -------------------------------
def smooth_curve(values, factor=0.8):
    smoothed = []
    last = values[0]
    for v in values:
        smoothed_val = last * factor + (1 - factor) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# -------------------------------
# LOAD DATA
# -------------------------------
images = []
labels = []

for idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATA_DIR, category)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(idx)

images = np.array(images)
labels = np.array(labels, dtype=int)

print(f"Total images: {len(images)}")
print(f"Class distribution: {np.bincount(labels)}")

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# Class weights
# class_weights = compute_class_weight(
#     class_weight='balanced', classes=np.unique(y_train), y=y_train
# )
# class_weights_dict = dict(enumerate(class_weights))
# print(f"Class weights: {class_weights_dict}")

# -------------------------------
# ENHANCED DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # zoom_range=0.3,
    # horizontal_flip=True,
    # vertical_flip=True,
    # brightness_range=[0.8, 1.2],
    # shear_range=0.2,
    # channel_shift_range=0.1,
    # fill_mode='reflect'
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------------
# BUILD ENHANCED MODEL WITH REGULARIZATION
# -------------------------------
# Simplified CNN Model
def create_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------
# CALLBACKS
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_cnn_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = create_cnn()
model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    # class_weight=class_weights_dict,
    callbacks=[reduce_lr, checkpoint],
    verbose=1
)

# -------------------------------
# EVALUATE & VISUALIZE
# -------------------------------
model.load_weights('best_cnn_model.h5')
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

# Plot curves
plt.figure(figsize=(12,5))
# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc', alpha=0.3)
plt.plot(history.history['val_accuracy'], label='Val Acc', alpha=0.3)
plt.plot(smooth_curve(history.history['accuracy']), label='Train Acc (smoothed)')
plt.plot(smooth_curve(history.history['val_accuracy']), label='Val Acc (smoothed)')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
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
plt.savefig('training_history.png')
plt.show()

# -------------------------------
# ADDITIONAL DIAGNOSTICS
# -------------------------------
# Check if dataset is too small
if len(images) < 100:
    print("\nWARNING: Dataset is very small. Consider:")
    print("1. Collecting more data")
    print("2. Using transfer learning with a pretrained model")
    print("3. Trying simpler model architectures")

# Check class balance
class_counts = np.bincount(labels)
if max(class_counts) / min(class_counts) > 5:
    print("\nWARNING: Severe class imbalance detected")
    print("Consider using more aggressive class weighting or oversampling")