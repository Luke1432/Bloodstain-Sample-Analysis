# -------------------------------
# BLOODSTAIN PATTERN CNN CLASSIFIER (SMALL CNN, SMALL DATA FRIENDLY)
# -------------------------------

# -------------------------------
# SECTION 1: Import Libraries
# -------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# -------------------------------
# SECTION 2: Configuration
# -------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 50
DATA_DIR = "SIZE_120_rescaled_max_area_1024"
CATEGORIES = ["120_blunt", "120_gun"]

# -------------------------------
# SECTION 3: Load Images
# -------------------------------
images = []
labels = []

for idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATA_DIR, category)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        images.append(img_array)
        labels.append(idx)

images = np.array(images)
labels = np.array(labels)

# -------------------------------
# SECTION 4: Split Dataset
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

# -------------------------------
# SECTION 5: Handle Class Imbalance
# -------------------------------
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# -------------------------------
# SECTION 6: Data Augmentation
# -------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# -------------------------------
# SECTION 7: Build SMALL CNN Model
# -------------------------------
model = Sequential([
    # First conv block
    Conv2D(16, (3,3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Second conv block
    Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
    MaxPooling2D((2,2)),
    Dropout(0.2),

    # Flatten and dense layers
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),

    # Output layer
    Dense(2, activation='softmax')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# SECTION 8: Train the Model
# -------------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights_dict
)

# -------------------------------
# SECTION 9: Evaluate and Plot
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# -------------------------------
# SECTION 10: Print Final Statistics
# -------------------------------
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

for images, labels in val_generator:
    # Get predictions from the model
    predictions = model.predict(images)
    
    if model.output_shape[-1] == 2:
        # For binary classification, threshold predictions at 0.5
        predicted_labels = (predictions >= 0.5).astype(int).flatten()
    else:
        # For multi-class classification, use argmax to get the predicted class
        predicted_labels = np.argmax(predictions, axis=1)
    
    # Loop through each image in the batch
    for i in range(len(images)):
        # Display the image
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.axis("off")
        plt.title(f"Guess: {predicted_labels[i]}  Label: {labels[i].numpy()}")
        plt.show()