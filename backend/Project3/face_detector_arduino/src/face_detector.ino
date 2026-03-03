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
  uint8_t row_buf[640]; // one row of 320 RGB565 pixels

  cam.flush_fifo();
  cam.clear_fifo_flag();
  cam.start_capture();
  while (!cam.get_bit(ARDUCHIP_TRIG, CAP_DONE_MASK)) { delay(1); }

  uint32_t fifo_len = cam.read_fifo_length();
  Serial.print("FIFO length: "); Serial.println(fifo_len);

  cam.CS_LOW();
  SPI.transfer(BURST_FIFO_READ);

  for (int src_row = 0; src_row < 240; src_row++) {
    for (int i = 0; i < 640; i++) row_buf[i] = SPI.transfer(0x00);

    if (src_row == 0) {
      Serial.print("Row 0 sample bytes: ");
      Serial.print(row_buf[0], HEX); Serial.print(" ");
      Serial.print(row_buf[1], HEX); Serial.print(" ");
      Serial.print(row_buf[2], HEX); Serial.print(" ");
      Serial.println(row_buf[3], HEX);
    }

    if (src_row % 5 != 0) continue;        // skip non-sampled rows
    int out_row = src_row / 5;

    for (int out_col = 0; out_col < 48; out_col++) {
      int src_col = (int)(out_col * 320.0f / 48.0f + 0.5f);
      if (src_col >= 320) src_col = 319;

      uint16_t px = ((uint16_t)row_buf[src_col * 2] << 8) | row_buf[src_col * 2 + 1];

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
  cam.CS_HIGH();
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

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
  cam.set_bit(ARDUCHIP_TIM, VSYNC_LEVEL_MASK);
  cam.clear_fifo_flag();

  // Override to RGB565 raw output (bypass JPEG compression)
  // Register 0xFF selects register bank (0x00 = DSP, 0x01 = sensor)
  // Register 0xDA (IMAGE_MODE) controls output format:
  //   Bit[4]=0 (non-compressed), Bit[3:2]=10 (RGB565)
  cam.wrSensorReg8_8(0xFF, 0x00);  // Select DSP register bank
  cam.wrSensorReg8_8(0xDA, 0x08);  // IMAGE_MODE: non-compressed RGB565

  // Enable auto exposure and auto gain control
  // Register 0xFF = 0x01 to access sensor registers
  // Register 0x13 (COM8): Bit[0]=AEC enable, Bit[2]=AGC enable
  // Register 0x14 (COM9): Bits[7:5]=AGC gain ceiling (100=32x)
  cam.wrSensorReg8_8(0xFF, 0x01);  // Select sensor register bank
  cam.wrSensorReg8_8(0x13, 0x05);  // COM8: Enable AEC (bit 0) and AGC (bit 2)
  cam.wrSensorReg8_8(0x14, 0x48);  // COM9: AGC gain ceiling = 32x

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
