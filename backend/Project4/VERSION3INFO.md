# Project 4 — Version 3 Changes

## Overview

This update addresses three major problems with the gesture detection pipeline:
the training data source, the model/output organization, and a domain mismatch
between training data and live inference. The result is a fully script-driven
workflow that no longer requires manual file editing between runs.

---

## Problem 1: Collecting Training Data Required a Webcam

The original collection script required sitting in front of a webcam and manually
performing each gesture hundreds of times. This was time-consuming and produced
inconsistent results depending on lighting, distance, and hand position that day.

### Solution: HaGRID Dataset Collector

A new script pulls training images directly from the HaGRID dataset (already
downloaded locally), which contains thousands of labeled hand gesture photos
taken across diverse people, lighting conditions, and backgrounds.

**The key challenge was matching the training data to what the Arduino camera
actually sees.** Several approaches were tried before landing on the correct one:

- **Tight bbox crop** — cropped exactly around the hand. The model trained well
  on this data but failed on the Arduino because the Arduino sees the hand as a
  small part of a larger scene, not filling the entire frame.

- **Full frame resize** — resized the entire photo to 96×96. The hand ended up
  only a few pixels wide and the model achieved ~20% accuracy (random chance
  with 5 classes) because there was no meaningful gesture signal to learn from.

- **Context crop (current approach)** — crops a square centered on the hand bbox
  with 30% padding on each side, so the hand fills ~77% of the frame. The bbox
  format in HaGRID is `[x_min, y_min, width, height]`, and the crop window is
  shifted (not just clamped) when it extends outside the image boundary, so the
  hand stays centered even when near the edge of the photo.

### How to Run

```sh
python collect_gesture_data_hagrid.py
```

You will be prompted for:
- **Dataset name** — a folder name for this collection run (e.g. `hagrid_v1`)
- **Gestures to include** — comma-separated; any gesture from the full HaGRID
  list is valid (call, dislike, fist, four, like, mute, ok, one, palm, peace, etc.)
- **Max samples per gesture** — leave blank to use all available images

A preview grid of sample crops is shown at the end so you can verify the hand
is centered and well-framed before training.

Output is saved to `data/processed/<dataset_name>/`.

---

## Problem 2: The Pipeline Required Manual File Editing Between Runs

Every time you trained a new model you had to open `train_cnn_model.py` and
manually change `DATASET_NAME`. All output files (model, TFLite, plots, C header)
were written to the project root with fixed filenames, meaning each new training
run silently overwrote the previous one. There was also no easy way to test a
specific model without editing hardcoded paths.

### Solution: Prompted Inputs and Per-Model Output Directories

The training script now asks for the dataset name and model name at startup.
All outputs for a run are saved together under `models/<model_name>/`:

```
models/
  my_model_v1/
    my_model_v1.keras
    my_model_v1.tflite
    model_data.h
    training_history.png
    confusion_matrix.png
```

This keeps every training run isolated and makes it easy to compare results or
go back to a previous model. When you are ready to deploy a model to the Arduino,
the script tells you exactly which `model_data.h` to copy and where.

### How to Run Training

```sh
python train_cnn_model.py
```

You will be prompted for:
- **Dataset name** — must match a folder in `data/processed/`
- **Model name** — used for output folder and file names (e.g. `gesture_v1`)

---

## Problem 3: High Training Accuracy but Poor Live Performance

The model trained at 90%+ accuracy on HaGRID data but gave nearly random
predictions on the webcam. The cause was a domain mismatch — HaGRID images
look different from a live webcam feed in terms of background, lighting, and
perspective.

### Solution: Webcam Sample Collector + Matched Preprocessing

Two changes were made:

**1. Webcam sample collector (`collect_webcam_samples.py`)** — a new script
that captures samples directly from your webcam using the exact same
center-crop → 96×96 grayscale preprocessing that inference uses. It can
append to an existing dataset, so you can mix HaGRID samples with a smaller
number of real webcam samples to bridge the domain gap.

**2. Matched inference preprocessing** — the webcam test script was updated to
take a center square crop of the frame before resizing to 96×96, instead of
resizing the full frame. This ensures the model sees the same framing at test
time as it was trained on.

### How to Collect Webcam Samples

```sh
python collect_webcam_samples.py
```

You will be prompted for:
- **Dataset name** — can be an existing dataset (samples are appended) or a new one
- **Gestures to record** — comma-separated; new gestures are added to the label map automatically
- **Target samples per gesture** — default 300

Controls during recording:
- **SPACE** — start / pause auto-capture (holds SPACE without toggling repeatedly)
- **q** — finish current gesture and move to the next
- **ESC** — save everything and quit immediately

A live inset in the top-right corner shows the exact 96×96 frame being captured
so you can see what the model will see.

### How to Test on Webcam

```sh
python helper_files/test_float32_webcam.py
```

You will be prompted for:
- **Model name** — must match a folder in `models/`
- **Dataset name** — used to load the correct label map from `data/processed/`

Press `c` to classify. A second window shows the exact 96×96 input the model
classified, so you can verify the framing looks correct.

### Deploying to Arduino

After training, copy `model_data.h` from `models/<model_name>/` into
`gesture_detector_arduino/` and re-flash the sketch.

---

## Full Workflow

```
1. Collect HaGRID data (bulk, offline)
   python collect_gesture_data_hagrid.py

2. (Optional) Supplement with webcam samples (append to same dataset)
   python collect_webcam_samples.py

3. Train
   python train_cnn_model.py

4. Test on webcam
   python helper_files/test_float32_webcam.py

5. Deploy
   Copy models/<model_name>/model_data.h → gesture_detector_arduino/
   Flash gesture_detector_arduino.ino via Arduino IDE
```