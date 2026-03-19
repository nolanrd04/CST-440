# Project 4: Modular TinyML Gesture Pipeline

This project is a clean, modular starter for gesture detection on edge devices (TinyML/Arduino), aligned to your 11-step workflow.

Current scope:

- End-to-end pipeline skeleton is implemented.
- Training runs today with a synthetic IMU adapter.
- Real dataset integration points are ready.
- TFLite + C header export path is included.

## 1) Setup

```bash
cd Project4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Quick Run (Synthetic Data)

Train + evaluate IMU baseline:

```bash
PYTHONPATH=src python -m project4.main train-imu
```

Export TensorFlow model to float32/int8 TFLite and Arduino header:

```bash
PYTHONPATH=src python -m project4.main export
```

Run state-machine demo (threshold + cooldown logic):

```bash
PYTHONPATH=src python -m project4.main demo-state-machine
```

## 3) Key Output Artifacts

Generated in `Project4/artifacts/`:

- `imu_gesture_model.keras`
- `imu_gesture_model_float32.tflite`
- `imu_gesture_model_int8.tflite`
- `imu_gesture_model_int8.h`
- `evaluation_summary.json`

## 4) Integrating Real Dataset Next

Replace synthetic adapter logic in `src/project4/data/dataset.py`:

- Implement `RealIMUDatasetAdapter.load_raw_streams()`
- Keep downstream modules unchanged (windowing, features, train/eval/export)

## 5) Project Structure

```text
Project4/
  README.md
  REPORT.md
  requirements.txt
  src/project4/
    main.py
    config.py
    data/
    features/
    models/
    training/
    evaluation/
    optimization/
    deployment/
    runtime/
```
