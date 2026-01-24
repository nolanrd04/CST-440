/*
 * Float32 Trig Model Test
 * Tests if the issue is with int8 quantization or TFLite Micro itself
 */

#include <ArduTFLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "trig_model_float32.h"

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 50 * 1024;
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("=== Float32 Trig Model Test ===");
  Serial.println("Testing if int8 quantization is the issue\n");

  tflite::InitializeTarget();

  model = tflite::GetModel(trig_model_float32);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Schema mismatch! Model: ");
    Serial.print(model->version());
    Serial.print(", Expected: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while(1);
  }

  Serial.print("Model size: ");
  Serial.print(trig_model_float32_len);
  Serial.println(" bytes");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while(1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.print("Input type: ");
  Serial.print(input->type);
  if (input->type == kTfLiteFloat32) Serial.println(" (FLOAT32)");
  else if (input->type == kTfLiteInt8) Serial.println(" (INT8)");
  else Serial.println(" (OTHER)");

  Serial.print("Output type: ");
  Serial.print(output->type);
  if (output->type == kTfLiteFloat32) Serial.println(" (FLOAT32)");
  else if (output->type == kTfLiteInt8) Serial.println(" (INT8)");
  else Serial.println(" (OTHER)");

  Serial.println("\nStarting tests...\n");
  delay(1000);
}

void loop() {
  // 22 test angles within trained range [-3.14, 3.14]
  const int num_tests = 22;

  Serial.println("DATA_START");

  for (int i = 0; i < num_tests; i++) {
    // Generate angle: -3.14 to 3.14 in equal steps
    float x_val = -3.14f + (6.28f * i / (num_tests - 1));
    float x_norm = (x_val + 3.14f) / (2.0f * 3.14f);

    // Compute SIN
    input->data.f[0] = x_norm;
    input->data.f[1] = 1.0f;
    input->data.f[2] = 0.0f;
    interpreter->Invoke();
    float sin_pred = output->data.f[0];

    // Compute COS
    input->data.f[0] = x_norm;
    input->data.f[1] = 0.0f;
    input->data.f[2] = 1.0f;
    interpreter->Invoke();
    float cos_pred = output->data.f[0];

    // Derive TAN = SIN / COS
    float tan_pred = (fabs(cos_pred) > 0.01f) ? sin_pred / cos_pred : 0.0f;

    // Output CSV format: angle,sin_pred,cos_pred,tan_pred
    Serial.print(x_val, 6);
    Serial.print(",");
    Serial.print(sin_pred, 6);
    Serial.print(",");
    Serial.print(cos_pred, 6);
    Serial.print(",");
    Serial.println(tan_pred, 6);
  }

  Serial.println("DATA_END");

  // Wait before next run
  delay(60000);
}