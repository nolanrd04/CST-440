from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from project4.config import GestureConfig


@dataclass
class TrainResult:
    model: tf.keras.Model
    history: tf.keras.callbacks.History


def train_classifier(
    model: tf.keras.Model,
    x_train,
    y_train,
    x_val,
    y_val,
    cfg: GestureConfig,
) -> TrainResult:
    """Step 6: Train with regularization and validation monitoring."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    return TrainResult(model=model, history=history)
