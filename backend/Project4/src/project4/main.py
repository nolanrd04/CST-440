from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from project4.config import GestureConfig, ensure_dirs
from project4.data.dataset import SyntheticIMUDatasetAdapter, split_dataset
from project4.deployment.arduino_export import estimate_tensor_arena_bytes, tflite_to_c_header
from project4.evaluation.metrics import evaluate_classifier, save_evaluation
from project4.features.imu_features import extract_imu_features
from project4.models.architectures import build_imu_mlp
from project4.optimization.tflite_export import export_float32_tflite, export_int8_tflite, model_file_size_bytes
from project4.runtime.state_machine import GestureStateMachine
from project4.training.trainer import train_classifier


def run_train_imu(cfg: GestureConfig) -> None:
    ensure_dirs(cfg)

    x_windows, y = SyntheticIMUDatasetAdapter(cfg).load_windows()
    x_features = extract_imu_features(x_windows)
    split = split_dataset(x_features, y, cfg)

    model = build_imu_mlp(input_dim=x_features.shape[1], num_classes=cfg.num_classes)
    result = train_classifier(model, split.x_train, split.y_train, split.x_val, split.y_val, cfg)

    eval_summary = evaluate_classifier(result.model, split.x_test, split.y_test, cfg.class_names)
    save_evaluation(eval_summary, cfg.artifacts_dir / "evaluation_summary.json")

    model_path = cfg.artifacts_dir / "imu_gesture_model.keras"
    result.model.save(model_path)
    print(f"Saved model: {model_path}")
    print(f"Test accuracy: {eval_summary['accuracy']:.4f}")


def run_export(cfg: GestureConfig) -> None:
    ensure_dirs(cfg)
    model_path = cfg.artifacts_dir / "imu_gesture_model.keras"
    if not model_path.exists():
        raise FileNotFoundError("Train first: artifacts/imu_gesture_model.keras not found.")

    model = tf.keras.models.load_model(model_path)

    # Build representative feature set from synthetic adapter for int8 calibration.
    x_windows, _ = SyntheticIMUDatasetAdapter(cfg, samples_per_class=80).load_windows()
    x_features = extract_imu_features(x_windows)

    float32_path = export_float32_tflite(model, cfg.artifacts_dir / "imu_gesture_model_float32.tflite")
    int8_path = export_int8_tflite(model, x_features, cfg.artifacts_dir / "imu_gesture_model_int8.tflite")
    header_path = tflite_to_c_header(int8_path, cfg.artifacts_dir / "imu_gesture_model_int8.h", var_name="kws_model")

    int8_size = model_file_size_bytes(int8_path)
    print(f"Float32 TFLite: {float32_path}")
    print(f"Int8 TFLite: {int8_path} ({int8_size} bytes)")
    print(f"C header: {header_path}")
    print(f"Suggested tensor arena: ~{estimate_tensor_arena_bytes(int8_size)} bytes")


def run_state_machine_demo(cfg: GestureConfig) -> None:
    sm = GestureStateMachine(threshold=cfg.detection_threshold, cooldown_seconds=cfg.cooldown_seconds)

    # Simulated probabilities over time.
    timeline = [
        np.array([0.10, 0.10, 0.10, 0.10, 0.60]),
        np.array([0.05, 0.82, 0.04, 0.04, 0.05]),
        np.array([0.06, 0.80, 0.05, 0.04, 0.05]),
        np.array([0.10, 0.10, 0.76, 0.02, 0.02]),
    ]

    now = 0.0
    for idx, probs in enumerate(timeline):
        detected = sm.step(probs, now)
        print(json.dumps({"tick": idx, "time_s": now, "state": sm.state, "detected_class": detected}))
        now += 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project 4 TinyML gesture pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train-imu", help="Train/evaluate compact IMU model (synthetic adapter).")
    sub.add_parser("export", help="Export trained model to TFLite and Arduino C header.")
    sub.add_parser("demo-state-machine", help="Run threshold/cooldown runtime logic demo.")

    return parser.parse_args()


def main() -> None:
    cfg = GestureConfig()
    args = parse_args()

    if args.command == "train-imu":
        run_train_imu(cfg)
    elif args.command == "export":
        run_export(cfg)
    elif args.command == "demo-state-machine":
        run_state_machine_demo(cfg)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
