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
const char* FIRMWARE_VERSION = "v3.1";

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

// Debug: store RGB565 color image (48x48)
uint16_t debug_image_rgb565[48 * 48];

void captureAndFillTensor() {
  uint8_t row_buf[2200]; // Large enough for one row

  cam.flush_fifo();
  cam.clear_fifo_flag();

  cam.start_capture();
  while (!cam.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) { delay(1); }

  uint32_t fifo_len = cam.read_fifo_length();
  Serial.print("FIFO length: "); Serial.println(fifo_len);

  if (fifo_len < 150000 || fifo_len > 160000) {
    Serial.print("WARNING: Abnormal FIFO size! Expected ~153600, got ");
    Serial.println(fifo_len);
  }

  uint32_t bytes_per_row = fifo_len / 240;
  Serial.print("Bytes per row: "); Serial.println(bytes_per_row);
  uint16_t pixels_per_row = bytes_per_row / 2;

  cam.set_fifo_burst();

  // Process image using nearest-neighbor interpolation
  for (int src_row = 0; src_row < 240; src_row++) {
    for (uint32_t i = 0; i < bytes_per_row; i++) {
      row_buf[i] = cam.read_fifo();
    }

    // Determine which output row this source row maps to
    int out_row = (src_row * 48 + 120) / 240;  // nearest output row with rounding
    if (out_row >= 48) out_row = 47;

    if (src_row == 0) {
      Serial.print("Row 0 sample bytes: ");
      Serial.print(row_buf[0], HEX); Serial.print(" ");
      Serial.print(row_buf[1], HEX); Serial.print(" ");
      Serial.print(row_buf[2], HEX); Serial.print(" ");
      Serial.println(row_buf[3], HEX);
    }

    // Downsample columns with nearest-neighbor interpolation
    for (int out_col = 0; out_col < 48; out_col++) {
      // Find nearest source column
      int src_col = (out_col * 320 + 160) / 48;  // nearest column with rounding
      if (src_col >= pixels_per_row) src_col = pixels_per_row - 1;

      uint16_t px = (row_buf[src_col * 2] << 8) | row_buf[src_col * 2 + 1];

      // Store RGB565 color
      debug_image_rgb565[out_row * 48 + out_col] = px;

      // RGB565 → grayscale
      uint8_t r = ((px >> 11) & 0x1F) << 3;
      uint8_t g = ((px >> 5)  & 0x3F) << 2;
      uint8_t b = (px         & 0x1F) << 3;
      uint16_t gray = (77 * r + 150 * g + 29 * b) >> 8;

      // Store debug image
      debug_image[out_row * kImageSize + out_col] = gray;

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

// Enable auto exposure and auto gain control
cam.wrSensorReg8_8(0xFF, 0x01);  // Select sensor register bank
cam.wrSensorReg8_8(0x13, 0x05);  // COM8: Enable AEC (bit 0) and AGC (bit 2)
cam.wrSensorReg8_8(0x14, 0x28);  // COM9: AGC gain ceiling = 16x

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

  captureAndFillTensor();

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke() failed!"); return;
  }

  float non_face = output->data.f[0];
  float face     = output->data.f[1];
  Serial.println(face > non_face ? "FACE DETECTED" : "NO FACE");
}
