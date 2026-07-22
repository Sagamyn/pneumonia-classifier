import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics import confusion_matrix, classification_report


IMG_SIZE = 150
BATCH_SIZE = 16
IMAGES_PER_EPOCH = 800
EPOCHS = 50
OUTPUT_DIR = "./results"
DECISION_THRESHOLD = 0.5  # tune this after running find_best_threshold()


def build_file_dataframe(split_dir):
    """
    Walks a split folder (train/NORMAL, train/PNEUMONIA, etc.) and builds a
    dataframe of [filepath, class] rows -- used so we can oversample NORMAL
    before handing it to flow_from_dataframe.
    """
    rows = []
    for category in ["NORMAL", "PNEUMONIA"]:
        category_path = os.path.join(split_dir, category)
        if not os.path.exists(category_path):
            continue
        for fname in os.listdir(category_path):
            rows.append({"filepath": os.path.join(category_path, fname), "class": category})
    return pd.DataFrame(rows)


def oversample_minority_class(df):
    """
    Duplicates rows of the minority class (NORMAL, typically ~3x rarer than
    PNEUMONIA in this dataset) until both classes have equal counts. Since
    training uses ImageDataGenerator augmentation, duplicated rows still get
    randomly rotated/shifted/zoomed differently each epoch -- they aren't
    literally identical images every time.
    """
    counts = df["class"].value_counts()
    print(f"Before oversampling: {counts.to_dict()}")

    majority_count = counts.max()
    balanced_frames = []
    for cls, count in counts.items():
        cls_df = df[df["class"] == cls]
        if count < majority_count:
            extra = cls_df.sample(majority_count - count, replace=True, random_state=42)
            cls_df = pd.concat([cls_df, extra], ignore_index=True)
        balanced_frames.append(cls_df)

    balanced_df = pd.concat(balanced_frames, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After oversampling: {balanced_df['class'].value_counts().to_dict()}")
    return balanced_df


def build_generators(train_dir, test_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_df = build_file_dataframe(train_dir)
    train_df = oversample_minority_class(train_df)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="class",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="binary",
        classes=["NORMAL", "PNEUMONIA"]
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="binary",
        classes=["NORMAL", "PNEUMONIA"],
        shuffle=False
    )

    return train_generator, test_generator


def create_cnn_model(img_size=IMG_SIZE):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 1)),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0005), metrics=["accuracy"])
    return model


def train_model(model, train_generator, test_generator,
                 epochs=EPOCHS, batch_size=BATCH_SIZE, images_per_epoch=IMAGES_PER_EPOCH):
    """
    Note: class_weight is no longer needed here since oversampling already
    balances the classes at the data level -- using both at once would
    over-correct and bias the model the other way (toward NORMAL).
    """
    steps_per_epoch = max(1, images_per_epoch // batch_size)
    validation_steps = max(1, test_generator.samples // batch_size)

    print(f"\nEach epoch will run {steps_per_epoch} steps x {batch_size} batch_size "
          f"= {steps_per_epoch * batch_size} images per epoch")
    print(f"Training CNN for up to {epochs} epochs (with early stopping on val_loss)...")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=validation_steps,
        callbacks=[early_stop],
        verbose=1
    )
    return history


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Test Accuracy", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy over Training")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Test Loss", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss over Training")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curve to {save_path}")
    plt.show()


def get_test_predictions(model, test_generator):
    """Runs the model over the full test set once and returns raw probabilities + true labels."""
    test_generator.reset()
    steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
    probabilities = model.predict(test_generator, steps=steps, verbose=0).ravel()
    probabilities = probabilities[:test_generator.samples]
    true_labels = test_generator.classes[:test_generator.samples]
    return probabilities, true_labels


def evaluate_with_confusion_matrix(model, test_generator, threshold=DECISION_THRESHOLD):
    """
    Prints a real confusion matrix and per-class precision/recall/F1 --
    the actual diagnostic tool for spotting class-specific bias, instead of
    relying on a single overall accuracy number.
    """
    probabilities, true_labels = get_test_predictions(model, test_generator)
    predictions = (probabilities > threshold).astype(int)

    cm = confusion_matrix(true_labels, predictions)
    print(f"\nConfusion Matrix (threshold={threshold}):")
    print(f"                 Predicted NORMAL   Predicted PNEUMONIA")
    print(f"Actual NORMAL    {cm[0][0]:^17d}   {cm[0][1]:^19d}")
    print(f"Actual PNEUMONIA {cm[1][0]:^17d}   {cm[1][1]:^19d}")

    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["NORMAL", "PNEUMONIA"]))

    return cm


def find_best_threshold(model, test_generator, thresholds=None):
    """
    Sweeps decision thresholds and reports NORMAL vs PNEUMONIA recall at each,
    so you can pick the threshold that best balances the two classes instead
    of blindly using 0.5.
    """
    if thresholds is None:
        thresholds = np.arange(0.3, 0.75, 0.05)

    probabilities, true_labels = get_test_predictions(model, test_generator)

    print("\nThreshold sweep (looking for the best NORMAL/PNEUMONIA balance):")
    print(f"{'Threshold':>10} | {'NORMAL Recall':>14} | {'PNEUMONIA Recall':>17} | {'Balanced Acc':>13}")

    best_threshold = 0.5
    best_balanced_acc = 0
    for t in thresholds:
        predictions = (probabilities > t).astype(int)
        cm = confusion_matrix(true_labels, predictions)
        normal_recall = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
        pneumonia_recall = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
        balanced_acc = (normal_recall + pneumonia_recall) / 2

        print(f"{t:>10.2f} | {normal_recall:>14.3f} | {pneumonia_recall:>17.3f} | {balanced_acc:>13.3f}")

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = t

    print(f"\nBest threshold found: {best_threshold:.2f} (balanced accuracy: {best_balanced_acc:.3f})")
    return best_threshold


def visualize_prediction(image_path, model, img_size=IMG_SIZE, true_label=None,
                          threshold=DECISION_THRESHOLD, save_dir=OUTPUT_DIR):
    categories = ["NORMAL", "PNEUMONIA"]

    img = keras_image.load_img(image_path, target_size=(img_size, img_size), color_mode="grayscale")
    x = keras_image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    probability = model.predict(x, verbose=0)[0][0]

    if probability > threshold:
        guess = "PNEUMONIA"
        confidence = probability * 100
    else:
        guess = "NORMAL"
        confidence = (1.0 - probability) * 100

    verdict_color = "#2ecc71" if guess == "NORMAL" else "#e74c3c"

    display_img = keras_image.load_img(image_path)
    fig, ax = plt.subplots(figsize=(5, 5.5))
    ax.imshow(display_img, cmap="gray")
    ax.axis("off")

    title = f"Prediction: {guess}  ({confidence:.1f}%)"
    if true_label is not None:
        title += f"\nActual: {true_label}"
    ax.set_title(title, color=verdict_color, fontsize=13, fontweight="bold")

    bar_text = f"NORMAL {(1 - probability)*100:.1f}%   |   PNEUMONIA {probability*100:.1f}%"
    fig.text(0.5, 0.03, bar_text, ha="center", fontsize=10)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    out_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_cnn_prediction.png"
    out_path = os.path.join(save_dir, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved prediction image: {out_path}")
    return out_path


def visualize_batch_predictions(image_paths, model, img_size=IMG_SIZE, true_label=None,
                                 threshold=DECISION_THRESHOLD, window_title=None, save_dir=OUTPUT_DIR):
    n = len(image_paths)
    if n == 0:
        print("No images to visualize.")
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]

    if window_title:
        fig.canvas.manager.set_window_title(window_title)

    for ax, image_path in zip(axes, image_paths):
        img = keras_image.load_img(image_path, target_size=(img_size, img_size), color_mode="grayscale")
        x = keras_image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        probability = model.predict(x, verbose=0)[0][0]

        if probability > threshold:
            guess = "PNEUMONIA"
            confidence = probability * 100
        else:
            guess = "NORMAL"
            confidence = (1.0 - probability) * 100

        verdict_color = "#2ecc71" if guess == "NORMAL" else "#e74c3c"

        display_img = keras_image.load_img(image_path)
        ax.imshow(display_img, cmap="gray")
        ax.axis("off")

        title = f"{guess} ({confidence:.1f}%)"
        if true_label is not None:
            title += f"\nActual: {true_label}"
        ax.set_title(title, color=verdict_color, fontsize=11, fontweight="bold")

    plt.tight_layout()

    if window_title:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{window_title.lower().replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved batch prediction grid: {out_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    DATASET_ROOT = "./datasets/chest_xray"
    train_dir = os.path.join(DATASET_ROOT, "train")
    test_dir = os.path.join(DATASET_ROOT, "test")

    # 1. Data generators -- NORMAL is now oversampled to match PNEUMONIA count
    train_generator, test_generator = build_generators(train_dir, test_dir)

    # 2. Build + train CNN (class_weight removed -- oversampling handles balance now)
    model = create_cnn_model()
    history = train_model(model, train_generator, test_generator)

    # 3. Overall test accuracy (still useful, just not the whole picture)
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f"\nFinal Model Test Accuracy: {test_acc * 100:.2f}%")

    plot_training_history(history, save_path=os.path.join(OUTPUT_DIR, "cnn_training_curve.png"))

    # 4. Real per-class diagnostics
    evaluate_with_confusion_matrix(model, test_generator, threshold=DECISION_THRESHOLD)
    best_threshold = find_best_threshold(model, test_generator)

    print(f"\n--- Re-evaluating at best threshold ({best_threshold:.2f}) ---")
    evaluate_with_confusion_matrix(model, test_generator, threshold=best_threshold)

    # 5. Visual (non-terminal) predictions on your val folder, using the best threshold
    SAMPLES_PER_CLASS = 4
    val_normal_dir = os.path.join(DATASET_ROOT, "val/NORMAL")
    val_pneumonia_dir = os.path.join(DATASET_ROOT, "val/PNEUMONIA")

    if os.path.exists(val_normal_dir) and os.listdir(val_normal_dir):
        normal_files = os.listdir(val_normal_dir)
        sample_normals = random.sample(normal_files, min(SAMPLES_PER_CLASS, len(normal_files)))
        sample_normal_paths = [os.path.join(val_normal_dir, f) for f in sample_normals]
        visualize_batch_predictions(sample_normal_paths, model, true_label="NORMAL",
                                     threshold=best_threshold, window_title="Normal Predictions")

    if os.path.exists(val_pneumonia_dir) and os.listdir(val_pneumonia_dir):
        pneumonia_files = os.listdir(val_pneumonia_dir)
        sample_pneumonias = random.sample(pneumonia_files, min(SAMPLES_PER_CLASS, len(pneumonia_files)))
        sample_pneumonia_paths = [os.path.join(val_pneumonia_dir, f) for f in sample_pneumonias]
        visualize_batch_predictions(sample_pneumonia_paths, model, true_label="PNEUMONIA",
                                     threshold=best_threshold, window_title="Pneumonia Predictions")