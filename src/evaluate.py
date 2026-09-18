"""
Evaluates the trained model on the test set: per-class precision/recall/F1
and a confusion matrix. This will reveal if the model is just predicting
the majority class ("mask") for everything.
"""

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import focal_loss

CLASSES = ["mask", "no_mask", "improper_mask"]

d = np.load("data/processed/test.npz")
X_test, y_test = d["X"], d["y"]

model = load_model("models/best_model.h5", custom_objects={"loss_fn": focal_loss()})
y_pred = np.argmax(model.predict(X_test), axis=1)

print("Predicted class distribution:", np.bincount(y_pred, minlength=3))
print("Actual class distribution:   ", np.bincount(y_test, minlength=3))
print()
print(classification_report(y_test, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("reports/figures/confusion_matrix.png")
print("\nConfusion matrix saved to reports/figures/confusion_matrix.png")