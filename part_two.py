# --- Import necessary libraries ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt
import numpy as np

# --- Define constants ---
DATA_DIR = pathlib.Path("SIZE_120_rescaled_max_area_1024")  # Path to the dataset
IMG_SIZE = (120, 120)  # Image dimensions
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 60  # Number of epochs for training

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

layers.Dropout(0.3),  # Increase dropout rate
keras.regularizers.l2(0.01),  # Increase L2 regularization

# Apply data augmentation to the training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))  # Augment training data

model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),  # Normalize pixel values to [0, 1]
    layers.Conv2D(32, 3, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),  # First convolutional layer with stronger L2 regularization
    layers.BatchNormalization(),  # Batch normalization to stabilize learning
    layers.MaxPooling2D(),  # First max pooling layer
    layers.Dropout(0.3),  # Increased dropout rate
    layers.Conv2D(64, 3, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),  # Second convolutional layer with stronger L2 regularization
    layers.BatchNormalization(),  # Batch normalization to stabilize learning
    layers.MaxPooling2D(),  # Second max pooling layer
    layers.Dropout(0.3),  # Increased dropout rate
    layers.Conv2D(128, 3, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),  # Third convolutional layer for increased complexity
    layers.BatchNormalization(),  # Batch normalization to stabilize learning
    layers.MaxPooling2D(),  # Third max pooling layer
    layers.Dropout(0.4),  # Increased dropout rate
    layers.Flatten(),  # Flatten the feature maps into a 1D vector
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.01)),  # Fully connected layer with stronger L2 regularization
    layers.Dropout(0.5),  # Increased dropout before the final layer
    layers.Dense(1, activation="sigmoid")  # Output layer for binary classification
])

# --- Compile the model ---
# Use Adam optimizer with a learning rate scheduler
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00005,
    decay_steps=10000,  # Decay less frequently
    decay_rate=0.9  # Reduce learning rate less aggressively
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)  # Adam optimizer with learning rate scheduler
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])  # Only use accuracy as a metric

# --- Train the model ---
# Use class weights to handle class imbalance
class_weights = {0: 1.0, 1: 4.0}  # Adjust weights based on class distribution
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=10,  # Stop if no improvement for 5 epochs
    restore_best_weights=True  # Restore the best model weights
)
history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS, callbacks=[early_stopping])

# --- Plot Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')  # Plot training accuracy
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Plot validation accuracy
plt.xlabel('Epoch')  # Label for x-axis
plt.ylabel('Accuracy')  # Label for y-axis
plt.title('Model Accuracy')  # Title of the plot
plt.legend()  # Add legend
plt.grid(True)  # Add grid for better readability
plt.show()

# --- Plot Loss ---
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')  # Plot training loss
plt.plot(history.history['val_loss'], label='Validation Loss')  # Plot validation loss
plt.xlabel('Epoch')  # Label for x-axis
plt.ylabel('Loss')  # Label for y-axis
plt.title('Model Loss')  # Title of the plot
plt.legend()  # Add legend
plt.grid(True)  # Add grid for better readability
plt.show()

# --- Evaluate the model ---
test_loss, test_accuracy = model.evaluate(test_ds)  # Evaluate the model on the test dataset
print(f"Test Loss: {test_loss:.4f}")  # Print test loss
print(f"Test Accuracy: {test_accuracy:.4f}")  # Print test accuracy

# --- Visualize sample predictions ---
# Get a batch of test images and labels
for images, labels in test_ds.take(1):  # Take one batch from the test dataset
    predictions = model.predict(images)  # Predict on the batch
    predicted_labels = (predictions > 0.5).astype(int)  # Convert probabilities to binary labels

    num_images = len(images)
    grid_size = int(np.ceil(np.sqrt(num_images)))  # Dynamically calculate grid size

    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        ax = plt.subplot(grid_size, grid_size, i + 1)  # Adjust grid size dynamically
        plt.imshow(images[i].numpy().astype("uint8"))  # Display the image
        plt.title(f"True: {labels[i].numpy()}, Pred: {predicted_labels[i][0]}")  # Show true and predicted labels
        plt.axis("off")  # Remove axis for better visualization
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    break  # Only visualize one batch