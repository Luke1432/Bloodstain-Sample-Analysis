import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Set the directory for your dataset
data_dir = 'SIZE_120_rescaled_max_area_1024'

# Step 2: Load and split the data into training and validation sets
IMG_SIZE = (120, 120)
BATCH_SIZE = 16

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2)
])
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.6,  # 80% for training, 20% for validation
    subset="training",  # Use this for training data
    seed=123,  # Random seed for reproducibility
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.4,  # Ensure the same split
    subset="validation",  # Use this for validation data
    seed=123,  # Same seed for reproducibility
)

# Normalize images to [0, 1] range
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x) / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

# Step 3: Build your CNN model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(120, 120, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset)

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(val_dataset)
print(f"Validation Accuracy: {test_acc}")

# Step 7: Visualize training progress (accuracy and loss)
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
