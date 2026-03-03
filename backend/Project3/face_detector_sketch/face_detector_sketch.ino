/*
  Face Detector - Mimicking Project2's successful structure
*/

#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "face_model_data.h"

// Model setup
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 50 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 50; i++) {
    Serial.write('.');
    delay(100);
  }

  Serial.println("\n\nLoading model...");

  model = tflite::GetModel(face_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch");
    while(1) delay(1000);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: Tensor allocation failed");
    while(1) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("SUCCESS: Model loaded!");
  Serial.println("Ready.");
}

void loop() {
  delay(1000);
  Serial.println("RUNNING");
}
