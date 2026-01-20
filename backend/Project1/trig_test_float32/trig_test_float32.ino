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
  // Test angles
  float test_angles[] = {-3.14, -1.571, 0.0, 1.571, 3.14};
  const char* labels[] = {"-pi", "-pi/2", "0", "pi/2", "pi"};

  Serial.println("=== SIN Tests ===");
  for (int i = 0; i < 5; i++) {
    float x_val = test_angles[i];
    float x_norm = (x_val + 3.14f) / (2.0f * 3.14f);

    // Float32 model - direct float input, no quantization
    input->data.f[0] = x_norm;
    input->data.f[1] = 1.0f;  // is_sin
    input->data.f[2] = 0.0f;  // is_cos

    interpreter->Invoke();

    float pred = output->data.f[0];
    float actual = sin(x_val);
    float error = fabs(pred - actual);

    Serial.print("sin(");
    Serial.print(labels[i]);
    Serial.print("): pred=");
    Serial.print(pred, 4);
    Serial.print(", actual=");
    Serial.print(actual, 4);
    Serial.print(", error=");
    Serial.println(error, 4);
  }

  Serial.println("\n=== COS Tests ===");
  for (int i = 0; i < 5; i++) {
    float x_val = test_angles[i];
    float x_norm = (x_val + 3.14f) / (2.0f * 3.14f);

    input->data.f[0] = x_norm;
    input->data.f[1] = 0.0f;  // is_sin
    input->data.f[2] = 1.0f;  // is_cos

    interpreter->Invoke();

    float pred = output->data.f[0];
    float actual = cos(x_val);
    float error = fabs(pred - actual);

    Serial.print("cos(");
    Serial.print(labels[i]);
    Serial.print("): pred=");
    Serial.print(pred, 4);
    Serial.print(", actual=");
    Serial.print(actual, 4);
    Serial.print(", error=");
    Serial.println(error, 4);
  }

  Serial.println("\n=== Test Complete ===");
  Serial.println("If errors are small (<0.05), float32 works.");
  Serial.println("This means int8 quantization handling is broken.\n");
  Serial.println("Waiting 30s...\n");
  delay(30000);
}