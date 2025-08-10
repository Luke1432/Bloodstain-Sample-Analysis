from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf

# --- Constants ---
EPOCHS = 100  # Number of epochs for training
BATCH_SIZE = 64  # Batch size for training
IMG_SIZE = (128, 128)  # Image size for resizing
DATA_DIR = "SIZE_120_rescaled_max_area_1024"  # Path to your dataset

# --- Data Augmentation ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomContrast(0.1),
    layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),  # Force fixed size
])

# --- Load the dataset ---
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
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
test_ds = full_ds.skip(train_size)  # Skip the first 80% for testing

# --- Normalize the Testing Dataset ---
test_ds = test_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# --- Ensure reproducibility ---
full_ds = keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Shuffle the dataset for randomness
    seed=42  # Set a seed for reproducibility
)

# --- Verify augmentation normalization ---
train_ds = train_ds.map(lambda x, y: (normalizer(data_augmentation(x)), y), num_parallel_calls=tf.data.AUTOTUNE)


# Apply normalization directly to the datasets
train_ds = train_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalizer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch the datasets
train_ds = train_ds.shuffle(1000).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# --- Define the Model ---
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Input layer with specified image size
    layers.Conv2D(32, 3, activation="relu", kernel_regularizer=keras.regularizers.l2(0.02)),  # First Conv Layer
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, padding="same", activation="relu"),  # Second Conv Layer
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.GlobalAveragePooling2D(),  # Replaces Flatten to reduce parameters
    layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),  # Dense Layer
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),  # Output layer for binary classification
])

# --- Compile the Model ---
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)  # Adjusted learning rate
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# --- Add Callbacks ---
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

# --- Train the Model ---
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight={0: 1.0, 1: 5.0},  # Adjust weights based on class distribution
    callbacks=[reduce_lr, early_stopping]
)

# --- Visualize Training Metrics ---
epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot loss
plt.figure(figsize=(18, 6))
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
