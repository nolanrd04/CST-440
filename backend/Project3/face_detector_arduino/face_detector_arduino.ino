/*
  Face Detection with Arducam OV2640 and TensorFlow Lite
  Captures 320x240 frames, resizes to 48x48 grayscale, runs inference
*/

#include <Wire.h>
#include <SPI.h>
#include <ArduCAM.h>

#undef swap
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "face_model_data.h"

// Version
const char* FIRMWARE_VERSION = "v2.7";

// Camera configuration
#define CS_PIN 10
ArduCAM cam(OV2640, CS_PIN);

// TFLite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 150 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Debug: store grayscale image (48x48)
uint8_t debug_image[48 * 48];

void captureAndFillTensor() {
  uint8_t row_buf[2200]; // Large enough for variable row sizes

  // Enhanced FIFO clearing - flush multiple times to ensure clean state
  for (int i = 0; i < 3; i++) {
    cam.flush_fifo();
    cam.clear_fifo_flag();
    delay(10);
  }
  
  cam.start_capture();
  while (!cam.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) { delay(1); }

  uint32_t fifo_len = cam.read_fifo_length();
  Serial.print("FIFO length: "); Serial.println(fifo_len);

  // Warn if FIFO size is abnormal (expected ~153600 for 320x240 RGB565)
  if (fifo_len < 150000 || fifo_len > 160000) {
    Serial.print("WARNING: Abnormal FIFO size! Expected ~153600, got ");
    Serial.println(fifo_len);
  }

  // Calculate bytes per row based on actual FIFO size (320x240 image)
  uint32_t bytes_per_row = fifo_len / 240;
  Serial.print("Bytes per row: "); Serial.println(bytes_per_row);

  cam.set_fifo_burst();

  for (int src_row = 0; src_row < 240; src_row++) {
    for (uint32_t i = 0; i < bytes_per_row; i++) row_buf[i] = cam.read_fifo();

    if (src_row == 0) {
      Serial.print("Row 0 sample bytes: ");
      Serial.print(row_buf[0], HEX); Serial.print(" ");
      Serial.print(row_buf[1], HEX); Serial.print(" ");
      Serial.print(row_buf[2], HEX); Serial.print(" ");
      Serial.println(row_buf[3], HEX);
    }

    if (src_row % 5 != 0) continue;        // skip non-sampled rows
    int out_row = src_row / 5;

    // Calculate actual pixel width from bytes per row
    uint16_t pixels_per_row = bytes_per_row / 2;

    for (int out_col = 0; out_col < 48; out_col++) {
      int src_col = (int)(out_col * (float)pixels_per_row / 48.0f + 0.5f);
      if (src_col >= pixels_per_row) src_col = pixels_per_row - 1;

      uint16_t px = row_buf[src_col * 2] | (row_buf[src_col * 2 + 1] << 8);

      // RGB565 → 8-bit components
      uint8_t r = ((px >> 11) & 0x1F) << 3;
      uint8_t g = ((px >> 5)  & 0x3F) << 2;
      uint8_t b = (px         & 0x1F) << 3;

      // Grayscale (BT.601 luminance, integer arithmetic)
      float gray = (77 * r + 150 * g + 29 * b) / 256.0f;

      // Store debug image (0-255 uint8)
      debug_image[out_row * kImageSize + out_col] = (uint8_t)gray;

      // Normalize to match training
      input->data.f[out_row * kImageSize + out_col] =
          (gray / 255.0f - kPixelMean) / kPixelStd;
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }
  Serial.print("Firmware version: ");
  Serial.println(FIRMWARE_VERSION);

  // --- Camera init ---
  Wire.begin();
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  // Verify SPI communication
  cam.write_reg(ARDUCHIP_TEST1, 0x55);
  if (cam.read_reg(ARDUCHIP_TEST1) != 0x55) {
    Serial.println("ArduCAM SPI error!"); while (1);
  }

  // Verify OV2640 chip ID (CHIPID_HIGH=0x0A, CHIPID_LOW=0x0B)
  uint8_t vid, pid;
  cam.wrSensorReg8_8(0xff, 0x01);
  cam.rdSensorReg8_8(0x0A, &vid);
  cam.rdSensorReg8_8(0x0B, &pid);
  if (vid != 0x26) { Serial.println("OV2640 not found!"); while (1); }
  Serial.print("OV2640 chip ID: 0x"); Serial.println((vid << 8) | pid, HEX);

  cam.set_format(BMP);
cam.InitCAM();

cam.clear_fifo_flag();

// Select DSP register bank
cam.wrSensorReg8_8(0xFF, 0x00);

// Disable JPEG compression engine
cam.wrSensorReg8_8(0xE0, 0x04);   // reset JPEG
cam.wrSensorReg8_8(0xE0, 0x00);   // disable JPEG

// Set RGB565 output
cam.wrSensorReg8_8(0xDA, 0x08);

Serial.println("Camera ready (320x240 RGB565)");

  // --- TFLite init ---
  tflite::InitializeTarget();
  model = tflite::GetModel(face_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!"); while (1);
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!"); while (1);
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("TFLite ready. Send 'c' to capture.");
}

void loop() {
  if (!Serial.available()) return;
  char cmd = Serial.read();
  if (cmd != 'c') return;

  Serial.println("Capturing...");
  captureAndFillTensor();

  // Debug: print first few input tensor values
  Serial.print("Input tensor sample [0]: "); Serial.println(input->data.f[0], 4);
  Serial.print("Input tensor sample [100]: "); Serial.println(input->data.f[100], 4);
  Serial.print("Input tensor sample [1000]: "); Serial.println(input->data.f[1000], 4);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke() failed!"); return;
  }

  float face     = output->data.f[0];  // Output indices are swapped in TFLite
  float non_face = output->data.f[1];
  Serial.print("Non-face score: "); Serial.println(non_face, 4);
  Serial.print("Face score:     "); Serial.println(face, 4);
  Serial.println(face > non_face ? "FACE DETECTED" : "NO FACE");

  // Send debug image as hex (48x48 grayscale)
  Serial.println("\nDEBUG_IMAGE_START");
  for (int i = 0; i < 48 * 48; i++) {
    if (debug_image[i] < 16) Serial.print("0");
    Serial.print(debug_image[i], HEX);
    if ((i + 1) % 48 == 0) Serial.println();  // newline after each row
    else Serial.print(" ");
  }
  Serial.println("DEBUG_IMAGE_END\n");
}
