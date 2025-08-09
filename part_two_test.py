import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Constants ---
EPOCHS = 1000  # Number of epochs for training
BATCH_SIZE = 32  # Batch size for training
IMG_SIZE = (128, 128)  # Image size for resizing
DATA_DIR = "SIZE_120_rescaled_max_area_1024"  # Path to your dataset

# --- Data Augmentation ---
# Apply random transformations to increase dataset diversity
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),  # Only flip images horizontally
])

# --- Load the dataset ---
# Load all images from the directory without splitting
full_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True  # Shuffle the dataset for randomness
)

# --- Get dataset size and split into training and testing ---
ds_size = tf.data.experimental.cardinality(full_ds).numpy()  # Get total number of samples
train_size = int(0.8 * ds_size)  # Use 80% for training
train_ds = full_ds.take(train_size)  # Take the first 80% for training
test_ds = full_ds.skip(train_size)  # Skip the first 80% for testing

# Apply data augmentation to the training dataset
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x), y),  # Apply augmentation
    num_parallel_calls=tf.data.AUTOTUNE
)

# Batch the dataset after augmentation
train_ds = train_ds.shuffle(1000).prefetch(tf.data.AUTOTUNE)  # Shuffle, batch, and prefetch
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)  # Batch and prefetch for testing

# Debugging: Check the shape of the dataset
for images, labels in train_ds.take(1):
    print("Images shape:", images.shape)  # Should be (batch_size, 128, 128, 3)
    print("Labels shape:", labels.shape)  # Should match the batch size

# --- Define the Model ---
model = keras.Sequential([
    layers.Conv2D(128, 3, activation="relu", kernel_regularizer=keras.regularizers.l2(0.02), input_shape=(128, 128, 3)),  # Specify input shape
    layers.BatchNormalization(),  # Batch normalization to stabilize learning
    layers.MaxPooling2D(),  # Max pooling layer
    layers.Dropout(0.35),  # Slightly increased dropout rate
    layers.Flatten(),  # Flatten the feature maps into a 1D vector
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.02)),  # Fully connected layer with stronger L2 regularization
    layers.Dropout(0.55),  # Increased dropout before the final layer
    layers.Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# --- Compile the Model ---
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00005,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# --- Add Callbacks ---
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.5,  # Reduce learning rate by half
    patience=5,  # Wait for 5 epochs of no improvement
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss instead of accuracy
    patience=10,  # Allow more epochs before stopping
    restore_best_weights=True  # Restore weights from the best epoch
)

# --- Train the Model ---
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight={0: 1.0, 1: 4.0},  # Adjust weights based on class distribution
    callbacks=[early_stopping, reduce_lr]  # Include ReduceLROnPlateau
)

# --- Visualize Training Metrics ---
epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()

# --- Debugging Output ---
for images, labels in train_ds.take(1):
    print(images.shape)  # Should be (batch_size, 128, 128, 3)
    print(labels.shape)  # Should match the number of samples in the batch