# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a CST-440 course repository containing projects and lab assignments focused on machine learning for embedded systems. The repository is organized with backend projects (Python-based data science/ML work) and lab assignment documentation.

## Course Context

### Topic 1: Introduction to Machine Learning on Microcontrollers (Jan 5-25, 2026)

This topic covers the transition from machine learning on large computing devices (laptops, smartphones) to constrained embedded devices (microcontrollers). The focus is on designing, training, and deploying machine learning models that can run on resource-limited hardware.

**Learning Objectives:**
- Design a deep learning workflow for an embedded device
- Build and train a machine learning model
- Deploy a machine learning application on a microcontroller

**Key Project: CLC – Machine Learning on a Microcontroller**

The main collaborative project involves designing and training a machine learning model to compute trigonometric functions, then deploying it to a microcontroller. The workflow follows these steps:

1. **Model Design** - Create an architecture that fits the task and resource constraints
2. **Model Building** - Implement the model using frameworks like TensorFlow
3. **Model Training** - Teach the model using collected/preprocessed data
4. **Application Building** - Package the model into a deployable application
5. **Testing** - Validate accuracy and efficiency of the model
6. **Deployment** - Deploy to microcontroller (e.g., Arduino) and capture output

**Lab Questions Focus:**
- Training data design for mathematical functions
- Neural network implementation using TensorFlow
- Deployment workflows and diagrams
- Differences between microcontroller vs. desktop deployment

## Project Structure

```
CST-440/
├── backend/
│   ├── Project1/              # Trigonometric function approximation (MLP)
│   │   ├── create_float32_model.py
│   │   ├── convert_tflite_to_c.py
│   │   ├── Report.ipynb
│   │   ├── trig_test_float32/
│   │   └── requirements.txt
│   ├── Project2/              # Keyword Spotting (CNN on audio)
│   │   ├── DataImporter.py    # Load Speech Commands dataset
│   │   ├── DataPreprocessor.py # MFCC feature extraction
│   │   ├── train_cnn_model.py
│   │   ├── Report.ipynb
│   │   ├── keyword_spotting_arduino/  # Arduino deployment files
│   │   └── requirements.txt
│   ├── Project3/              # Facial Detection (CNN on images)
│   │   ├── DataImporter.py    # Load Labeled Faces in the Wild
│   │   ├── DataPreprocessor.py # Grayscale 64x64 preprocessing
│   │   ├── train_cnn_model.py
│   │   ├── Report.ipynb
│   │   ├── face_detector_arduino/  # Arduino deployment files
│   │   └── requirements.txt
│   └── requirements.txt        # Root-level shared dependencies
├── Lab Questions/             # Lab assignment documentation
└── README.md
```

## Development Setup

Each backend project is self-contained with its own virtual environment and `requirements.txt`. Setup is identical across all projects:

### Setup for Any Project

1. Navigate to the project directory:
   ```sh
   cd backend/ProjectX  # Replace X with 1, 2, or 3
   ```

2. Create and activate virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Dependencies

All projects use:
- **Data manipulation**: numpy, pandas
- **Machine learning**: tensorflow (2.x), scikit-learn
- **Audio processing** (Project2): librosa (MFCC features)
- **Image processing** (Project2, Project3): opencv-python
- **Visualization**: matplotlib, seaborn
- **Jupyter environment**: jupyter, ipython, jupyterlab
- **Scientific computing**: scipy

See individual `backend/ProjectX/requirements.txt` for exact versions.

## Common Development Tasks

All projects follow a standard ML workflow: **Data → Preprocess → Train → Convert → Deploy**

### Project1: Trigonometric Functions

```sh
cd backend/Project1
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train the model
python create_float32_model.py

# Convert to TFLite and generate C header
python convert_tflite_to_c.py
```

**Output files:**
- `trig_model_all.tflite` - Compressed model
- `trig_model_all.h` - C header for Arduino

### Project2: Keyword Spotting

```sh
cd backend/Project2
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Step 1: Import raw audio data from Speech Commands dataset
python -c "from DataImporter import load_file_lists; print('Importing data...')"

# Step 2: Preprocess data (extract MFCC features, split train/val/test)
python -c "from DataPreprocessor import preprocess; preprocess()"

# Step 3: Train the CNN model and convert to TFLite
python train_cnn_model.py

# Step 4: View results and confusion matrix
# Open Report.ipynb in Jupyter
jupyter notebook Report.ipynb
```

**Output files:**
- `kws_model.keras` - Trained model
- `kws_model.tflite` - TensorFlow Lite model
- `keyword_spotting_arduino/` - Arduino deployment files (C headers, sketch)
- `data/processed/` - Preprocessed training/validation/test data

### Project3: Facial Detection

```sh
cd backend/Project3
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Step 1: Import Labeled Faces in the Wild dataset
python -c "from DataImporter import load_file_lists; print('Importing data...')"

# Step 2: Preprocess images (grayscale, 64x64, split train/val/test)
python -c "from DataPreprocessor import preprocess; preprocess()"

# Step 3: Train the CNN model and convert to TFLite
python train_cnn_model.py

# Step 4: View results and analysis
jupyter notebook Report.ipynb
```

**Output files:**
- `face_model.keras` - Trained model
- `face_model.tflite` - TensorFlow Lite model
- `face_detector_arduino/` - Arduino deployment files
- `data/processed/` - Preprocessed training/validation/test data

### Typical Workflow Pattern

Each project follows this structure:

1. **DataImporter.py** - Loads raw dataset and organizes files
2. **DataPreprocessor.py** - Extracts features (MFCC, normalization, resizing), creates train/val/test splits
3. **train_cnn_model.py** - Builds, trains, evaluates model; converts to TFLite; generates C headers
4. **Report.ipynb** - Jupyter notebook for visualization, confusion matrices, and analysis
5. **`*_arduino/` directory** - Generated C headers and Arduino sketch templates

## Architecture Notes

### Backend Structure

All three projects follow an identical architectural pattern:

**Data Pipeline:**
- `DataImporter.py` - Loads raw dataset, maps labels, handles train/val/test file lists
- `DataPreprocessor.py` - Feature extraction (MFCC, image resizing, normalization), saves as `.npy` arrays
- Training data stored in `data/processed/` as numpy arrays (X_train, y_train, X_val, y_val, X_test, y_test)

**Model Pipeline:**
- `train_cnn_model.py` - Single script that:
  - Loads preprocessed data
  - Builds CNN/RNN architecture
  - Trains with validation monitoring
  - Evaluates on test set with per-class metrics
  - Converts to float32 TensorFlow Lite
  - Generates C header file for Arduino

**Deployment Artifacts:**
- `.keras` file - Full trained model for local inference testing
- `.tflite` file - Compressed model for microcontroller deployment
- `*_arduino/` directory:
  - `model_data.h` - C byte array of quantized model weights
  - `*.ino` - Arduino sketch template for inference
  - Supporting header files for TensorFlow Lite Micro

**Model Specifications:**
- **Project1**: MLP (256→128→3) on normalized angles [0, 2π] → sin/cos/tan values
- **Project2**: 1D CNN (Conv→MaxPool→Dense) on MFCC features (49 frames × 26 coefficients) → 8 keyword classes
- **Project3**: 2D CNN (Conv→MaxPool→Dense) on 64×64 grayscale images → binary face/no-face classification

### Key Implementation Details

- **GPU Handling**: All training scripts force CPU-only (`CUDA_VISIBLE_DEVICES='-1'`) to ensure portability
- **Random Seeds**: Set for reproducibility (SEED=42 in all scripts)
- **Quantization**: Projects 2 and 3 generate float32 TFLite models; Project1 uses dedicated quantization script
- **Data Format**: NumPy arrays (.npy) for efficient preprocessing caching

### Lab Questions & Documentation

The `Lab Questions/` directory contains conceptual documentation for course assignments, separate from hands-on implementation. Topics cover:
- ML workflow design for embedded systems
- Neural network architecture choices for resource constraints
- Data preparation strategies for microcontrollers
- Comparison between desktop vs. microcontroller deployment
- Model compression techniques (quantization, pruning)

## Development Guidelines

### Model Training & Testing

- **Training is CPU-only**: All scripts disable GPU (`CUDA_VISIBLE_DEVICES='-1'`) for portability. For GPU acceleration during development, remove this line, but do not commit GPU-dependent code.
- **Data preprocessing is one-time**: Run preprocessing scripts once per dataset; the resulting `.npy` files in `data/processed/` can be reused across training runs.
- **Jupyter notebooks (Report.ipynb)** are for visualization and analysis only—never commit trained model outputs to these files.

### Arduino Deployment

- Each project's `*_arduino/` directory contains:
  - Auto-generated `.h` file with quantized model weights (do not edit manually)
  - `.ino` sketch template (requires TensorFlow Lite Micro library in Arduino IDE)
- Model size targets: <100KB for Project1 (1MB flash), <200KB for Projects 2-3 (varies by board)
- Verify model accuracy on desktop test set before deploying to microcontroller

### Debugging Tips

- **Project1 (Trigonometric)**: Use `trig_test_float32/analyze_results.py` to compare model output vs. ground truth
- **Project2 (Keyword Spotting)**: Check confusion matrix in Report.ipynb for misclassified classes; verify MFCC extraction matches training
- **Project3 (Facial Detection)**: Inspect preprocessed images in `data/processed/` to ensure proper grayscale conversion and normalization

## Git Workflow

The main branch is `main`. The repository excludes:
- Virtual environments (`venv/`, `*/venv/`, `.venv/`)
- Preprocessed data (`backend/Project*/data/processed/` is partially ignored)
- Personal projects (`PersonalProjects/`)
- Temporary files (`**/__pycache__/`, `**.zip`)
