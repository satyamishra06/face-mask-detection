"""
Trains the mask classifier using MobileNetV2 transfer learning.
Applies class weights to compensate for the dataset's imbalance
(mask: 3232, no_mask: 717, improper_mask: 123).
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model

CLASSES = ["mask", "no_mask", "improper_mask"]


def load_split(name):
    d = np.load(f"data/processed/{name}.npz")
    return d["X"], d["y"]


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    # compute class weights so the model doesn't just predict "mask" every time
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    print("Class weights:", {CLASSES[k]: round(v, 2) for k, v in class_weights.items()})

    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1
    )

    model = build_model()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint("models/best_model.h5", save_best_only=True, monitor="val_accuracy")
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=25,
        class_weight=class_weights,
        callbacks=callbacks
    )

    print("\nTraining complete. Best model saved to models/best_model.h5")


if __name__ == "__main__":
    main()