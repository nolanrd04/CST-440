# Project 4 Brief Report

## Goal

Build a high-modularity TinyML gesture pipeline for IMU/camera workflows, with clear handoff from training to Arduino deployment.

## IMU (What It Is)

IMU stands for Inertial Measurement Unit. It is a sensor package (typically accelerometer + gyroscope) that measures motion and rotation over time. For gesture detection, IMU streams become time-series windows used by the model to classify actions such as swipe, circle, shake, or no gesture.

## Steps and What Each Step Does

1. Problem definition: sets target gestures, sensor choice, and hardware limits so the model is feasible on device.
2. Data collection: captures realistic raw sensor streams across users and conditions, including no-gesture data.
3. Labeling and segmentation: assigns class labels and splits continuous streams into fixed windows for training input.
4. Feature engineering: converts windows into useful representations (statistics/frequency for IMU, resize/grayscale for vision).
5. Model architecture design: chooses compact networks that fit memory/latency constraints.
6. Model training: learns model weights with train/validation/test splits and regularization to reduce overfitting.
7. Evaluation and error analysis: measures accuracy, confusion matrix, and class-level performance to find failure modes.
8. TinyML optimization: converts to TFLite, applies quantization, and checks accuracy-memory tradeoffs.
9. Embedded deployment: converts model to C header and prepares interpreter/tensor arena for Arduino inference.
10. Runtime state machine integration: adds thresholding and cooldown to prevent noisy repeated triggers.
11. Real-world iteration: tests latency, robustness, and power impact, then refines data/model/deployment settings.

## Current Project Status

1. Modular codebase is implemented with clear separation: data, features, models, training, evaluation, optimization, deployment, and runtime.
2. Synthetic IMU adapter is active now so development can continue before final dataset selection.
3. Real dataset hook is ready for next phase (adapter implementation).
4. End-to-end path is in place: train -> evaluate -> export TFLite -> generate C header.
5. Runtime gesture state machine is included for threshold/cooldown behavior.

## Architecture

### Organization and Coordination

**Init files** (`__init__.py`) mark directories as importable Python packages. They currently contain docstrings only but can re-export common functions for convenience later.

**Config** (`config.py` defines `GestureConfig`):

- Single source of truth for problem definition (gesture classes, sensor specs, hardware limits).
- Encapsulates hyperparameters (batch size, epochs, dropout) and paths (artifacts directory).
- Computed properties (window_size, num_classes) eliminate redundant calculations.
- `ensure_dirs()` centralizes folder setup, called once per pipeline run.

**Main** (`main.py`) orchestrates the full pipeline:

1. Creates one `GestureConfig` instance passed to all modules.
2. Three command routes handle train → evaluate → export → deploy workflows:
   - `train-imu`: data load → feature extract → split → build model → train → evaluate → save.
   - `export`: load model → quantize float32 → quantize int8 → convert to C header.
   - `demo-state-machine`: simulate runtime threshold and cooldown logic.

Each module pulls one concern (data, features, models, training, evaluation, optimization, deployment, runtime) and communicates via config and simple return types. This design keeps dataset/model changes isolated and allows easy addition of new modalities (e.g., vision gate coming in a future phase).

### Key Technical Choices

#### Why Grayscale (Not RGB)

- Gesture motion is captured in spatial patterns, not color detail.
- Grayscale reduces input from 3 channels to 1, cutting memory and compute by 2/3.
- Smaller model footprint fits better on resource-constrained Arduino boards.
- Conversion happens in `vision_features.py` via TensorFlow's `rgb_to_grayscale()`.

#### Why Float32 → Int8 (Quantization)

- Float32: full 32-bit precision, ~4 bytes per weight. Accurate but large, slow on embedded.
- Int8: 8-bit integer (−128 to +127), ~1 byte per weight. *4× smaller model, faster inference, less RAM.*
- Trade-off: slight accuracy loss (typically 1–2% on gesture tasks), but acceptable for thresholded detection.
- How it works: representative dataset calibration finds min/max ranges per layer, then scales floats to int8 range while keeping relative differences.
- When to use: always for Arduino/TinyML; always quantize int8 after float32 works well.

#### Hardware-Aware Model Design

- IMU MLP uses small dense layers (64→32 neurons) instead of large networks.
- Vision CNN uses GlobalAveragePooling instead of Flatten to reduce parameters by ~90%.
- Dropout regularizes without increasing model size; early stopping prevents overfitting.
- Result: ~15k–30k parameters fits in Arduino flash memory with room for other code.

#### Feature Engineering Strategy

- IMU: statistical (mean, variance, RMS) + frequency (FFT energy) capture gesture dynamics compactly.
- Normalization (z-score) makes training stable across different sensor hardware.
- No per-sample preprocessing during inference keeps Arduino runtime fast.

#### Training Safeguards

- Stratified train/val/test splits ensure balanced class representation.
- Early stopping avoids overfitting; learning rate reduction adapts to plateaus.
- Synthetic data adapter allows development before real dataset finalization; real adapter hook ready for plug-in.

#### Runtime State Machine

- Thresholding (confidence > 0.70) prevents false positives from noisy inference.
- Cooldown (1.0 second) prevents repeated triggers of the same gesture.
- Decouples model logic from deployment behavior, allowing tuning without retraining.

## Design Notes

- Modules separated by responsibility: dataset changes do not ripple to model/export layers.
- Pipeline runs with synthetic IMU data now; real data integration is straightforward via adapter.
- Int8 export includes representative dataset calibration and memory footprint estimation.

## Next Step

Implement `RealIMUDatasetAdapter.load_raw_streams()` with your selected gesture dataset and tune model/feature choices based on validation metrics.
