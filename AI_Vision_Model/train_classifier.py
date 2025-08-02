import matplotlib
# matplotlib.use('Agg')  # Disable interactive plots for headless training

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import pathlib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import time
from tensorflow.keras import mixed_precision
from tqdm import tqdm
 # Removed keras_focal_loss import since package is unavailable
from tensorflow.keras.applications import MobileNetV2
mixed_precision.set_global_policy('mixed_float16')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to hide INFO and WARNING messages
from PIL import Image

# --- Custom Focal Loss ---
def binary_focal_loss(gamma=2., alpha_bad=0.7, alpha_good=0.3):
    """Custom focal loss giving more weight to 'bad_bags' (label 0)."""
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha = tf.where(tf.equal(y_true, 0), alpha_bad, alpha_good)
        return tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * bce)
    return loss

tf.get_logger().setLevel('ERROR')

# Path to your data folders
data_dir = pathlib.Path("./dataset")

# --- Manual stratified split into train/val/test subdirectories ---
import shutil
from sklearn.model_selection import train_test_split

# Automatically remove .DS_Store files

for root, _, files in os.walk(data_dir):
    for f in files:
        if f == ".DS_Store":
            os.remove(os.path.join(root, f))

bad_files = []
for root, _, files in os.walk(data_dir):
    for f in files:
        path = os.path.join(root, f)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception:
            bad_files.append(path)

print("Bad files:", bad_files)

# Automatically delete other non-image files
valid_exts = (".jpg", ".jpeg", ".png")
for root, _, files in os.walk(data_dir):
    for f in files:
        if not f.lower().endswith(valid_exts):
            print(f"Removing non-image file: {f}")
            os.remove(os.path.join(root, f))

total_images = sum(len(files) for _, _, files in os.walk(data_dir))
print(f"Total images found: {total_images}")

# --- Create stratified train/val/test split ---
def create_stratified_split(base_dir, output_dir, test_size=0.1, val_size=0.1):
    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for cls in classes:
        images = [os.path.join(base_dir, cls, f) for f in os.listdir(os.path.join(base_dir, cls)) if f.lower().endswith(valid_exts)]
        if len(images) == 0:
            continue
        # Split into test and remaining
        train_files, test_files = train_test_split(images, test_size=test_size, stratify=[cls]*len(images), random_state=42)
        # Split train_files into train and val
        if len(train_files) > 0:
            train_files, val_files = train_test_split(train_files, test_size=val_size/(1 - test_size), stratify=[cls]*len(train_files), random_state=42)
        else:
            val_files = []
        for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            split_dir = os.path.join(output_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)
            for file in files:
                shutil.copy(file, os.path.join(split_dir, os.path.basename(file)))

# Clean and create directories
split_dir = "./dataset_split"
if os.path.exists(split_dir):
    shutil.rmtree(split_dir)
os.makedirs(split_dir, exist_ok=True)
create_stratified_split("./dataset", split_dir)

batch_size = 32
img_height = 128
img_width = 128

# --- Load new train/val/test splits ---
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(split_dir, "train"),
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels="inferred",
    label_mode="int"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(split_dir, "val"),
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels="inferred",
    label_mode="int"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(split_dir, "test"),
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels="inferred",
    label_mode="int"
)

class_names = train_ds.class_names
print("Detected classes:", class_names)

# --- Improved Data Augmentation ---
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomContrast(0.1),
])

bad_bag_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.3),
    layers.GaussianNoise(0.05),
])

# --- Oversample bad_bags with balanced batch sampling ---
def oversample_dataset(dataset):
    # Unbatch first to make individual samples
    dataset = dataset.unbatch()

    bad_ds = dataset.filter(lambda x, y: tf.equal(y, 0))
    good_ds = dataset.filter(lambda x, y: tf.equal(y, 1))

    bad_count = tf.data.experimental.cardinality(bad_ds).numpy()
    good_count = tf.data.experimental.cardinality(good_ds).numpy()

    multiplier_bad = max(1, (good_count // max(1, bad_count)) * 2)
    multiplier_good = 1
    if bad_count > good_count:
        multiplier_good = max(1, (bad_count // max(1, good_count)) * 2)

    bad_ds_oversampled = bad_ds.repeat(multiplier_bad).map(
        lambda x, y: (tf.cast(bad_bag_augmentation(x), tf.float32), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    good_ds_oversampled = good_ds.repeat(multiplier_good)

    # Combine datasets to get near 50/50 balance
    combined = tf.data.Dataset.zip((good_ds_oversampled, bad_ds_oversampled)).flat_map(
        lambda good, bad: tf.data.Dataset.from_tensors(good).concatenate(tf.data.Dataset.from_tensors(bad))
    )

    combined = combined.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return combined

train_ds = oversample_dataset(train_ds)

# --- Display sample images ---
plt.figure(figsize=(8, 8))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.suptitle("Sample Images from Training Set")
plt.savefig("sample_images.png")

# --- Class Distribution ---
print("Calculating class distribution...")
counts = {0: 0, 1: 0}
for _, labels in train_ds.unbatch():
    lbl = int(labels)
    counts[lbl] = counts.get(lbl, 0) + 1
plt.bar(class_names, counts.values())
plt.title("Class Distribution")
plt.savefig("class_distribution.png")

# --- Compute Class Weights ---
labels_list = []
for _, labels in train_ds.unbatch():
    labels_list.append(int(labels))
labels_list = np.array(labels_list)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_list),
    y=labels_list
)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}
# Increase class weight for bad bags by multiplying by 3 (less aggressive)
if 0 in class_weights_dict:
    class_weights_dict[0] *= 3
print("Class Weights:", class_weights_dict)

# --- Plot and save class distribution
plt.figure(figsize=(6,4))
plt.bar(class_names, counts.values())
plt.title("Class Distribution in Oversampled Training Data")
plt.ylabel("Number of Samples")
plt.savefig("oversampled_class_distribution.png")
plt.close()

# --- Balanced Model Architecture ---
def build_balanced_model():
    return models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])

# --- Transfer Learning Model ---
def build_transfer_learning_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    return model

# --- Data Augmentation ---
# data_augmentation = keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomTranslation(0.1, 0.1)
# ])

# Save augmentation preview
for images, _ in train_ds.take(1):
    augmented_images = data_augmentation(images)
    plt.figure(figsize=(8, 8))
    for i in range(min(9, len(images))):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.suptitle("Augmented Image Samples")
    plt.savefig("augmented_samples.png")
    break

# --- CNN Architectures ---
def build_baseline():
    """Simple CNN with 3 convolution layers"""
    return models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(64, activation='relu'),
        layers.Dropout(0.3), layers.Dense(1, activation='sigmoid')
    ])

def build_deeper():
    """Deeper CNN with batch normalization and 4 convolution layers"""
    return models.Sequential([
        data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(256, 3, activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), layers.Dense(1, activation='sigmoid')
    ])

def build_alt_pooling():
    """CNN with average pooling and global average pooling"""
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

models_list = {
    "Baseline": build_baseline(),
    "Deeper": build_deeper(),
    "AltPooling": build_alt_pooling(),
    "TransferLearning": build_transfer_learning_model()
}

# Add balanced model to list
models_list["Balanced"] = build_balanced_model()

print("Models to be trained:")
for name in models_list.keys():
    print(f" - {name}")

# Compute good_count and bad_count for use in training loop steps_per_epoch
# Use train_ds unbatched for class counts
unbatched_train = train_ds.unbatch()
bad_count = 0
good_count = 0
for _, label in unbatched_train:
    if int(label) == 0:
        bad_count += 1
    else:
        good_count += 1

results = []
for name, model in tqdm(models_list.items(), desc="Training Models"):
    if name == "Balanced":
        loss_fn = 'binary_crossentropy'
    else:
        loss_fn = binary_focal_loss(gamma=2)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    start = time.time()
    
    steps_per_epoch = int(np.ceil(240 / batch_size))  # Adjust based on train image count
    if name == "Balanced":
        steps_per_epoch = int(np.ceil((good_count + bad_count) / batch_size))
    
    history = model.fit(
        train_ds.repeat(),
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        epochs=30,
        callbacks=[early_stop],
        verbose=1,
        class_weight=class_weights_dict
    )
    
    duration = time.time() - start
    loss, acc = model.evaluate(test_ds)
    results.append([name, acc, model.count_params(), duration])

    # Accuracy and loss plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{name} Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{name} Loss')
    plt.legend()
    plt.show()

    # Precision-Recall Curve and Best Threshold
    from sklearn.metrics import precision_recall_curve
    y_true, y_scores = [], []
    for images, labels in test_ds:
        scores = model.predict(images).flatten()
        y_true.extend(labels.numpy())
        y_scores.extend(scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Best threshold for {name}: {best_threshold:.2f}, Precision={precisions[best_idx]:.2f}, Recall={recalls[best_idx]:.2f}, F1={f1_scores[best_idx]:.2f}")

    # --- Plot and save precision-recall curve
    plt.figure(figsize=(6,4))
    plt.plot(recalls, precisions, marker='.')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

    y_pred = (np.array(y_scores) > best_threshold).astype("int32")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    # --- Display misclassified images ---
    misclassified_indices = [i for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(misclassified_indices[:9]):
        image = test_ds.unbatch().skip(idx).take(1)
        for img, lbl in image:
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(img.numpy().astype("uint8"))
            plt.title(f"True: {class_names[lbl]}, Pred: {class_names[y_pred[idx]]}")
            plt.axis("off")
    plt.suptitle(f"Misclassified Images - {name}")
    plt.show()

    # --- F1 Score vs Threshold plot
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, f1_scores[:-1], marker='o')
    plt.title(f'F1 Score vs Threshold - {name}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.show()

    print(f"Classification Report for {name}:\n", classification_report(y_true, y_pred, labels=[0, 1], target_names=class_names))
    model.save(f"{name}_model.keras")

print("All models have been trained and evaluated successfully.")

df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Parameters", "Training Time (s)"])
print(df_results)
df_results.to_csv("model_comparison.csv", index=False)