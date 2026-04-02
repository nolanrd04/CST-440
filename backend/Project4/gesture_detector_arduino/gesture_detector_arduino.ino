/*
  Gesture Detection — Project 4
  Captures 320x240 RGB565 frames from an ArduCAM OV2640, downsamples to
  96x96 grayscale, and runs int8 TFLite Micro inference to classify one of
  5 hand gestures (or report no gesture if confidence is too low).

  Classes (index order matches training label_map.json):
    0: call   1: dislike   2: like   3: mute   4: ok

  Hardware:
    - Arduino Nano 33 BLE Sense
    - ArduCAM OV2640 Mini 2MP  (SPI, CS pin -> D10)

  Libraries (install via Arduino Library Manager):
    - ArduCAM
    - Chirale_TensorFlowLite

  Usage:
    Open Serial Monitor at 115200 baud.
    Send the character 'c' to trigger a capture + inference.
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

#include "model_data.h"

// Version
const char* FIRMWARE_VERSION = "v1.0";

// Camera configuration
#define CS_PIN 10
ArduCAM cam(OV2640, CS_PIN);

// Model dimensions
#define IMG_ROWS   96
#define IMG_COLS   96
#define CROP_ROWS  192  // Intermediate crop size (then downsampled to 96x96)
#define CROP_COLS  192
#define NUM_CLASSES 5

// Minimum dequantized softmax score to report a detection
const float CONFIDENCE_THRESHOLD = 0.60f;

const char* CLASS_NAMES[NUM_CLASSES] = {
  "call", "dislike", "like", "mute", "ok"
};

// TFLite globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 150 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Debug: store grayscale image (96x96)
uint8_t debug_image[IMG_ROWS * IMG_COLS];

void captureAndFillTensor() {
  uint8_t row_buf[2200]; // large enough for one row at 320px * 2 bytes

  cam.flush_fifo();
  cam.clear_fifo_flag();

  cam.start_capture();
  while (!cam.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) { delay(1); }

  uint32_t fifo_len = cam.read_fifo_length();
  Serial.print("FIFO length: "); Serial.println(fifo_len);

  uint32_t bytes_per_row = fifo_len / 240;
  uint16_t pixels_per_row = bytes_per_row / 2;

  // Cache quantization params — same for every pixel in the input tensor
  float   in_scale = input->params.scale;
  int32_t in_zp    = input->params.zero_point;

  // Centered 192x192 crop offsets within the 320x240 frame
  const int CROP_ROW_OFFSET = (240 - CROP_ROWS) / 2;  // 24: use rows 24-215
  const int CROP_COL_OFFSET = (320 - CROP_COLS) / 2;  // 64: use cols 64-255

  cam.set_fifo_burst();

  for (int src_row = 0; src_row < 240; src_row++) {
    for (uint32_t i = 0; i < bytes_per_row; i++) {
      row_buf[i] = cam.read_fifo();
    }

    // Determine which crop row this source row maps to
    int crop_row = src_row - CROP_ROW_OFFSET;
    if (crop_row < 0 || crop_row >= CROP_ROWS) continue;

    // Sample every 2nd row from the cropped 192x192 to get 96x96
    if (crop_row % 2 != 0) continue;  // Skip odd rows
    int out_row = crop_row / 2;

    for (int out_col = 0; out_col < IMG_COLS; out_col++) {
      // Sample every 2nd column from the cropped region
      int crop_col = out_col * 2;
      int src_col = crop_col + CROP_COL_OFFSET;
      if (src_col >= (int)pixels_per_row) src_col = pixels_per_row - 1;

      // Unpack RGB565 big-endian
      uint16_t px = ((uint16_t)row_buf[src_col * 2] << 8) | row_buf[src_col * 2 + 1];

      // Expand to 8-bit channels
      uint8_t r = ((px >> 11) & 0x1F) << 3;
      uint8_t g = ((px >>  5) & 0x3F) << 2;
      uint8_t b = ( px        & 0x1F) << 3;

      // BT.601 luminance
      uint8_t gray = (uint8_t)((77 * (uint16_t)r + 150 * (uint16_t)g + 29 * (uint16_t)b) >> 8);

      // Store debug image
      debug_image[out_row * IMG_COLS + out_col] = gray;

      // float [0,1] -> int8: q = round(gray/255 / scale) + zero_point
      int32_t q = (int32_t)roundf((float)gray / (255.0f * in_scale)) + in_zp;
      if (q < -128) q = -128;
      if (q >  127) q =  127;
      input->data.int8[out_row * IMG_COLS + out_col] = (int8_t)q;
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

  // Verify OV2640 chip ID
  uint8_t vid, pid;
  cam.wrSensorReg8_8(0xff, 0x01);
  cam.rdSensorReg8_8(0x0A, &vid);
  cam.rdSensorReg8_8(0x0B, &pid);
  if (vid != 0x26) { Serial.println("OV2640 not found!"); while (1); }
  Serial.print("OV2640 chip ID: 0x"); Serial.println(((uint16_t)vid << 8) | pid, HEX);

  cam.set_format(BMP);
  cam.InitCAM();
  cam.clear_fifo_flag();

  // Select DSP register bank
  cam.wrSensorReg8_8(0xFF, 0x00);

  // Disable JPEG compression engine
  cam.wrSensorReg8_8(0xE0, 0x04);
  cam.wrSensorReg8_8(0xE0, 0x00);

  // Set RGB565 output
  cam.wrSensorReg8_8(0xDA, 0x08);

  // Disable auto exposure and auto gain control (manual mode)
  cam.wrSensorReg8_8(0xFF, 0x01);
  cam.wrSensorReg8_8(0x13, 0x00);  // COM8: Disable AEC + AGC
  cam.wrSensorReg8_8(0x14, 0x48);  // COM9: AGC gain ceiling = 32x

  Serial.println("Camera ready (320x240 RGB565)");

  // --- TFLite init ---
  tflite::InitializeTarget();
  model = tflite::GetModel(gesture_model_tflite);
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

  float   out_scale = output->params.scale;
  int32_t out_zp    = output->params.zero_point;

  int   best_idx   = 0;
  float best_score = -1.0f;

  Serial.println("Scores:");
  for (int i = 0; i < NUM_CLASSES; i++) {
    float score = (output->data.int8[i] - out_zp) * out_scale;
    if (score < 0.0f) score = 0.0f;
    Serial.print("  "); Serial.print(CLASS_NAMES[i]);
    Serial.print(": "); Serial.println(score, 4);
    if (score > best_score) { best_score = score; best_idx = i; }
  }

  Serial.println();
  if (best_score >= CONFIDENCE_THRESHOLD) {
    Serial.print("GESTURE DETECTED: ");
    Serial.print(CLASS_NAMES[best_idx]);
    Serial.print("  ("); Serial.print((int)(best_score * 100)); Serial.println("%)");
  } else {
    Serial.print("NO GESTURE DETECTED");
    Serial.print("  (best: "); Serial.print(CLASS_NAMES[best_idx]);
    Serial.print(" @ "); Serial.print((int)(best_score * 100)); Serial.println("%)");
  }

  // Send debug image as hex (96x96 grayscale)
  Serial.println("\nDEBUG_IMAGE_START");
  for (int i = 0; i < IMG_ROWS * IMG_COLS; i++) {
    if (debug_image[i] < 16) Serial.print("0");
    Serial.print(debug_image[i], HEX);
    if ((i + 1) % IMG_COLS == 0) Serial.println();  // newline after each row
    else Serial.print(" ");
  }
  Serial.println("DEBUG_IMAGE_END\n");
}
