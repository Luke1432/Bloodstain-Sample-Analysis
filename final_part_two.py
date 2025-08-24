# -------------------------------
# BLOODSTAIN CLASSIFIER - ENHANCED CNN (anti-collapse v2)
# -------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import losses  # kept for reference

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
try:
    _in = input("Enter number of epochs (default 100): ").strip()
    EPOCHS = int(_in) if _in else 100
except Exception:
    EPOCHS = 100

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
images, labels = [], []
for idx, category in enumerate(CATEGORIES):
    folder_path = os.path.join(DATA_DIR, category)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(idx)

images = np.array(images)
labels = np.array(labels, dtype='int32')

print(f"Total images: {len(images)}")
print(f"Class distribution: {np.bincount(labels)}")

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)
print("Train dist:", np.bincount(y_train))
print("Val dist:  ", np.bincount(y_val))

# -------------------------------
# AUGMENTATION (realistic/tame)
# -------------------------------
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    horizontal_flip=True,
    shear_range=0.05,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train, y_train.astype('int32'), batch_size=BATCH_SIZE, shuffle=True
)
val_generator = val_datagen.flow(
    X_val, y_val.astype('int32'), batch_size=BATCH_SIZE, shuffle=False
)

# -------------------------------
# FOCAL LOSS (sparse, class-balanced)
# -------------------------------
# alpha vector derived from train distribution (normalized)
train_counts = np.bincount(y_train)
alpha_vec = (train_counts.sum() / (2.0 * np.maximum(train_counts, 1))).astype(np.float32)
alpha_vec = alpha_vec / alpha_vec.sum()  # normalize to sum=1

def sparse_focal_loss(gamma=2.0, alpha=None):
    alpha = tf.constant(alpha, dtype=tf.float32) if alpha is not None else None
    def loss_fn(y_true, y_pred):
        # y_true: [N], y_pred: [N, C] (softmax probabilities)
        y_true = tf.reshape(tf.cast(y_true, tf.int32), (-1,))
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        # gather p_t for the true class
        idx = tf.stack([tf.range(tf.shape(y_pred)[0]), y_true], axis=1)
        p_t = tf.gather_nd(y_pred, idx)
        if alpha is not None:
            a_t = tf.gather(alpha, y_true)
        else:
            a_t = 1.0
        loss = -a_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return loss_fn

# -------------------------------
# MODEL (same style; remove first BN, seed priors in final bias)
# -------------------------------
# bias init to log-priors so initial softmax ≈ class priors
priors = (train_counts / train_counts.sum()).astype(np.float32)
bias_init = tf.keras.initializers.Constant(np.log(np.maximum(priors, 1e-7)))

def create_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same',
               input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        # BatchNormalization(),  # removed first BN to reduce small-batch noise
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.10),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.10),

        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.20),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.30),
        Dense(2, activation='softmax', bias_initializer=bias_init)
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=sparse_focal_loss(gamma=2.0, alpha=alpha_vec),
        metrics=['accuracy']
    )
    return model

# -------------------------------
# CALLBACKS (monitor by val_loss)
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss', patience=12,
    restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4,
    min_lr=1e-6, verbose=1
)
checkpoint = ModelCheckpoint(
    'best_cnn_model.weights.h5',   # Keras 3 weight-only suffix
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# -------------------------------
# TRAIN  (no class_weight; α in focal loss handles imbalance)
# -------------------------------
model = create_cnn()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop, checkpoint],
    verbose=1,
    # workers=8, use_multiprocessing=True, max_queue_size=64  # faster pipeline
)

# -------------------------------
# EVALUATE & VISUALIZE
# -------------------------------
model.load_weights('best_cnn_model.weights.h5')
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
plt.title('Accuracy over epochs'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', alpha=0.3)
plt.plot(history.history['val_loss'], label='Val Loss', alpha=0.3)
plt.plot(smooth_curve(history.history['loss']), label='Train Loss (smoothed)')
plt.plot(smooth_curve(history.history['val_loss']), label='Val Loss (smoothed)')
plt.title('Loss over epochs'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

plt.tight_layout(); plt.savefig('training_history.png'); plt.show()

# -------------------------------
# ADDITIONAL DIAGNOSTICS
# -------------------------------
y_pred_probs = model.predict(val_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
uniq, counts = np.unique(y_pred, return_counts=True)
print("Validation prediction distribution:", dict(zip(uniq.tolist(), counts.tolist())))

try:
    from sklearn.metrics import classification_report, confusion_matrix
    print("Confusion matrix:\n", confusion_matrix(y_val, y_pred))
    print("Classification report:\n",
          classification_report(y_val, y_pred, target_names=CATEGORIES))
except Exception as e:
    print("Could not compute classification report:", e)

if len(images) < 100:
    print("\nWARNING: Dataset is very small. Consider: more data, transfer learning, or simpler models.")
class_counts_all = np.bincount(labels)
if class_counts_all.min() > 0 and class_counts_all.max() / class_counts_all.min() > 5:
    print("\nWARNING: Severe class imbalance detected — consider stronger rebalancing.")
