"""
MobileNetV2 transfer learning with Focal Loss - the standard technique
for severe class imbalance (our improper_mask class has only 86 training
images vs 2262 for mask). Focal Loss automatically down-weights easy/majority
examples and focuses gradient updates on hard/rare examples, which is more
stable than the manual class weights and oversampling we tried before.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
NUM_CLASSES = 3


def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=NUM_CLASSES)
        y_true_oh = tf.reshape(y_true_oh, tf.shape(y_pred))
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true_oh * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        fl = weight * ce
        return tf.reduce_sum(fl, axis=-1)
    return loss_fn


def build_model(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base.trainable = False  # frozen - stable, reliable features

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()