"""
Trains the mask classifier using MobileNetV2 transfer learning.
Uses OVERSAMPLING (not just class weights) to fix severe imbalance -
v1/v2 used class weights alone and the model collapsed to predicting
only the majority class, because random batches rarely contained any
improper_mask examples at all (only 86 in training set vs 2260 for mask).
"""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import build_model

CLASSES = ["mask", "no_mask", "improper_mask"]
TARGET_PER_CLASS = 700  # oversample each minority class up to roughly this many


def load_split(name):
    d = np.load(f"data/processed/{name}.npz")
    return d["X"], d["y"]


def oversample(X, y):
    # build up index numbers first (cheap, just integers) instead of
    # duplicating full image arrays multiple times - avoids memory error
    all_indices = [np.arange(len(y))]
    for cls_idx in np.unique(y):
        cls_count = np.sum(y == cls_idx)
        if cls_count >= TARGET_PER_CLASS:
            continue
        needed = TARGET_PER_CLASS - cls_count
        cls_indices = np.where(y == cls_idx)[0]
        extra_indices = np.random.choice(cls_indices, size=needed, replace=True)
        all_indices.append(extra_indices)

    final_indices = np.concatenate(all_indices)
    np.random.shuffle(final_indices)

    # only ONE copy of the final array is created here
    return X[final_indices], y[final_indices]


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print("Before oversampling:", np.bincount(y_train))
    X_train, y_train = oversample(X_train, y_train)
    print("After oversampling: ", np.bincount(y_train))

    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    model = build_model()
    

    callbacks = [
        EarlyStopping(patience=7, restore_best_weights=True, monitor="val_accuracy"),
        ModelCheckpoint("models/best_model.h5", save_best_only=True, monitor="val_accuracy"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=30,
        callbacks=callbacks
    )

    print("\nTraining complete. Best model saved to models/best_model.h5")


if __name__ == "__main__":
    main()