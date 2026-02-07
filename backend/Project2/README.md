# Keyword Spotting Project

This project implements a keyword spotting system using machine learning models. It includes scripts for data preprocessing, training, and exporting models for deployment on Arduino devices.

## Project Structure

- `DataImporter.py`: Script to import and organize raw audio data.
- `DataPreprocessor.py`: Script to preprocess audio data and generate training, validation, and test datasets.
- `train_gru_model.py`: Script to train a GRU-based model for keyword spotting.
- `kws_model.keras`: Trained model in Keras format.
- `kws_model.tflite`: Trained model in TensorFlow Lite format for deployment.
- `data/`: Contains raw and processed audio data.
  - `archive/`: Raw audio data organized by keyword.
  - `processed/`: Preprocessed data and metadata files.
- `keyword_spotting_arduino/`: Contains Arduino deployment files.
  - `keyword_spotting_arduino.ino`: Arduino sketch for keyword spotting.
  - `kws_model_data.h`: Model data for Arduino.
- `Report.ipynb`: Jupyter notebook for generating reports and visualizations.
- `requirements.txt`: Python dependencies for the project.

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python installed. Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data

Run the `DataImporter.py` and `DataPreprocessor.py` scripts to prepare the data:

```bash
python DataImporter.py
python DataPreprocessor.py
```

### 3. Train the Model

Train the GRU-based model using the `train_gru_model.py` script:

```bash
python train_gru_model.py
```

This will generate the `kws_model.keras` and `kws_model.tflite` files.

### 4. Test the Model

Use the `Report.ipynb` notebook to evaluate the model's performance. Open the notebook in Jupyter or VS Code and run the cells.

### 5. Deploy to Arduino

#### Option 1: Using Arduino IDE

. Open Arduino IDE
2. Go to **Tools > Board > Boards Manager**
3. Search for "Arduino Mbed OS Nano Boards"
4. Install "Arduino Mbed OS Nano Boards" by Arduino

### 3. Install Required Libraries
1. **PDM** (by Arduino):
   - Go to **Tools > Manage Libraries**.
   - Search for "PDM" and install it.

2. **Chirale_TensorFlowLite**:
   - If not found in the Library Manager, manually install:
     1. Download from: https://github.com/ChiraleBrandon/Chirale_TensorFlowLite
     2. Go to **Sketch > Include Library > Add .ZIP Library**.
     3. Select the downloaded ZIP file.

3. **TensorFlow Lite Micro Libraries**:
   - These are required for TensorFlow Lite inference.
   - Follow the TensorFlow Lite Micro setup guide to include the necessary files in your Arduino project.

4. **CMSIS-DSP**:
   - Go to **Tools > Manage Libraries**.
   - Search for "CMSIS-DSP" or "CMSIS" and install it.
   - Ensure the `arm_math.h` file is available in the library folder.

### 4. Configure Board Settings
1. Connect your Arduino Nano 33 BLE to your computer via USB
2. Go to **Tools > Board** and select **Arduino Nano 33 BLE**
3. Go to **Tools > Port** and select the port showing your Arduino
   - On Mac: looks like `/dev/cu.usbmodem14201` or similar
   - On Windows: looks like `COM3` or similar
   - On Linux: looks like `/dev/ttyACM0` or similar

### 5. Open the Project
1. Open `keyword_spotting_arduino.ino` in Arduino IDE
2. The IDE should automatically open `kws_model_data.h` as a tab

### 6. Compile and Upload
1. Click the **Verify** button (✓) to compile the code
   - First compilation will take 2-5 minutes due to TensorFlow Lite library size
   - Subsequent compilations will be faster
2. Click the **Upload** button (→) to upload to the board
3. Open **Tools > Serial Monitor** to see debug output (set to 115200 baud)

#### Option 2: Using PlatformIO in VS Code

1. Install the [PlatformIO extension for VS Code](https://marketplace.visualstudio.com/items?itemName=platformio.platformio-ide).
2. Open the `keyword_spotting_arduino/` folder in VS Code.
3. Configure the Arduino board and port in PlatformIO:
   - Open the `platformio.ini` file and set the `board` to your Arduino board model.
   - Connect your Arduino device to your computer. In the PlatformIO interface, click on the "Devices" icon in the left-hand toolbar to view connected devices. Identify your Arduino's port and ensure it matches the `upload_port` value in the `platformio.ini` file. 
   *should be under the form of /dev/cu.usbmodemXXXX     Nano 33 BLE*
4. Build and upload the project:
   - Click the "Build" button (a checkmark icon) in the PlatformIO toolbar at the bottom of the VS Code window. If you don't see the toolbar, ensure the PlatformIO extension is installed and active. The toolbar typically appears when you open a project with a `platformio.ini` file. The Build button compiles the sketch and checks for any errors in the code.
   - Click the "Upload" button (a right-facing arrow icon) in the PlatformIO toolbar to flash the compiled sketch to the Arduino. Ensure your Arduino is connected to your computer during this step.

### 6. Test Voice Commands

After deploying the model to the Arduino, test the voice commands by speaking the keywords near the microphone connected to the Arduino. The system should recognize the keywords and perform the corresponding actions.

## Notes

- Ensure your Arduino board has sufficient memory to handle the model.
- Use the `data/_background_noise_/` folder to add background noise for better model robustness.
- Refer to the `data/archive/README.md` for details about the raw dataset.