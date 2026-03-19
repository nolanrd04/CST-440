from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


def export_float32_tflite(model: tf.keras.Model, output_path: Path) -> Path:
    """Step 8a: Convert Keras model to float32 TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return output_path


def export_int8_tflite(model: tf.keras.Model, representative_data: np.ndarray, output_path: Path) -> Path:
    """Step 8b/c/d: Int8 quantization and export with representative calibration."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_dataset() -> Iterable[list[np.ndarray]]:
        for sample in representative_data[: min(200, len(representative_data))]:
            yield [np.expand_dims(sample.astype(np.float32), axis=0)]

    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    return output_path


def model_file_size_bytes(model_path: Path) -> int:
    return model_path.stat().st_size
