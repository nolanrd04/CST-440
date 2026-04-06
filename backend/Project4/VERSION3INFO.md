# Project 4 — Version 3 Changes

## Overview

This update addresses two major problems with the gesture detection pipeline:
the training data source and the model/output organization. The result is a
fully script-driven workflow that no longer requires manual file editing between
runs.

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
actually sees.** Two approaches were tried and rejected before landing on the
correct one:

- **Tight bbox crop** — cropped exactly around the hand. The model trained well
  on this data but failed on the Arduino because the Arduino sees the hand as a
  small part of a larger scene, not filling the entire frame.

- **Full frame resize** — resized the entire photo to 96×96. The hand ended up
  only a few pixels wide and the model achieved ~20% accuracy (random chance
  with 5 classes) because there was no meaningful gesture signal to learn from.

- **Context crop (current approach)** — crops a square centered on the hand at
  3× the hand's size, then resizes to 96×96. The hand fills roughly one-third of
  the frame with natural background context around it. This better matches what
  the Arduino camera captures when someone holds their hand up close to it.

### How to Run

```sh
python collect_gesture_data_hagrid.py
```

You will be prompted for:
- **Dataset name** — a folder name for this collection run (e.g. `hagrid_v1`)
- **Gestures to include** — comma-separated from the full HaGRID gesture list
- **Max samples per gesture** — leave blank to use all available images

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

The webcam test script follows the same pattern — it asks which model and dataset
to use instead of having a hardcoded path.

### How to Run Training

```sh
python train_cnn_model.py
```

You will be prompted for:
- **Dataset name** — must match a folder in `data/processed/`
- **Model name** — used for output folder and file names (e.g. `gesture_v1`)

### How to Test on Webcam

```sh
python helper_files/test_float32_webcam.py
```

You will be prompted for:
- **Model name** — must match a folder in `models/`
- **Dataset name** — used to load the correct label map from `data/processed/`

Press `c` to classify the current frame, `q` to quit.

### Deploying to Arduino

After training, copy `model_data.h` from `models/<model_name>/` into
`gesture_detector_arduino/` and re-flash the sketch.

---

## Full Workflow

```
1. Collect data
   python collect_gesture_data_hagrid.py

2. Train
   python train_cnn_model.py

3. Test on webcam
   python helper_files/test_float32_webcam.py

4. Deploy
   Copy models/<model_name>/model_data.h → gesture_detector_arduino/
   Flash gesture_detector_arduino.ino via Arduino IDE
```