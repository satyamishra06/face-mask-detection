"""
Defines the CNN classifier using MobileNetV2 transfer learning.
Base layers are frozen initially - only the new classification head is trained.
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
NUM_CLASSES = 3  # mask, no_mask, improper_mask


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE, freeze_base=True):
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = not freeze_base

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    # quick sanity check - build the model and print its summary
    model = build_model()
    model.summary()