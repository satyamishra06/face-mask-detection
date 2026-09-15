"""
Loads cropped face images from data/raw/<class>/, resizes and normalizes them,
splits into train/val/test sets, and saves as .npz files in data/processed/.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
CLASSES = ["mask", "no_mask", "improper_mask"]  # index order matters - used as labels
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"


def load_images():
    X, y = [], []
    for label_idx, cls in enumerate(CLASSES):
        folder = os.path.join(RAW_DIR, cls)
        files = os.listdir(folder)
        print(f"Loading {len(files)} images from class '{cls}'...")

        for fname in files:
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0

            X.append(img)
            y.append(label_idx)

    return np.array(X), np.array(y)


def split_and_save(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(OUT_DIR, "val.npz"), X=X_val, y=y_val)
    np.savez(os.path.join(OUT_DIR, "test.npz"), X=X_test, y=y_test)

    print("\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")


if __name__ == "__main__":
    X, y = load_images()
    print(f"\nTotal images loaded: {len(X)}")
    split_and_save(X, y)
    print("\nSaved to data/processed/ (train.npz, val.npz, test.npz)")