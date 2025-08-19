import tensorflow as tf
import os

# Set the directory for your dataset
data_dir = 'SIZE_120_rescaled_max_area_1024'

# Define image size and batch size
IMG_SIZE = (120, 120)
BATCH_SIZE = 32

# Split dataset into training and testing using validation_split
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,  # 80% for training, 20% for testing
    subset="training",  # Use this for training data
    seed=123,  # Random seed for reproducibility
)

# Load validation data using the same parameters, but for validation
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.4,  # Ensure the same split
    subset="validation",  # Use this for validation data
    seed=123,  # Same seed for reproducibility
)

# Normalize images to [0, 1] range
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

# Print dataset sizes
train_size = tf.data.experimental.cardinality(train_dataset).numpy()
val_size = tf.data.experimental.cardinality(val_dataset).numpy()

print(f"Training dataset size: {train_size}")
print(f"Validation dataset size: {val_size}")

# Optional: You can shuffle the datasets if you want
train_dataset = train_dataset.shuffle(buffer_size=1000)
val_dataset = val_dataset.shuffle(buffer_size=1000)