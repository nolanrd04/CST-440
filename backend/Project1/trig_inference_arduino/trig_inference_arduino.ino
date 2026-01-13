/*
 * Trigonometric Model Inference on Arduino Nano 33 BLE
 * CST-440 - Machine Learning on Microcontrollers
 *
 * This sketch runs inference on a TensorFlow Lite model that computes
 * trigonometric functions (sin, cos, tan) for a given input angle.
 */

#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "trig_model_all.h"

// TensorFlow Lite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Memory allocation for TensorFlow Lite
  // Float32 models need more memory than int8 (4x per value)
  constexpr int kTensorArenaSize = 100 * 1024;  // 100KB for float32 model
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }

  Serial.println("========================================");
  Serial.println("TensorFlow Lite Trig Model - Arduino");
  Serial.println("CST-440 Project");
  Serial.println("========================================");

  // Initialize TensorFlow Lite
  tflite::InitializeTarget();

  // Load the TFLite model
  model = tflite::GetModel(trig_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version: ");
    Serial.println(model->version());
    Serial.print("Expected schema version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    Serial.println("ERROR: Model schema mismatch!");
    while(1);
  }

  Serial.println("Model loaded successfully!");
  Serial.print("Model size: ");
  Serial.print(sizeof(trig_model));
  Serial.println(" bytes");

  // Set up the operations resolver
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    while(1);
  }

  Serial.println("Tensors allocated successfully!");

  // Get pointers to the model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Print input tensor info
  Serial.println("\nInput Tensor Info:");
  Serial.print("  Dimensions: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();
  Serial.print("  Type: ");
  Serial.println(input->type);

  // Print output tensor info
  Serial.println("\nOutput Tensor Info:");
  Serial.print("  Dimensions: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();
  Serial.print("  Type: ");
  Serial.println(output->type);

  // Print quantization parameters if quantized
  if (input->type == kTfLiteInt8) {
    Serial.println("\nQuantization Info:");
    Serial.print("  Input scale: ");
    Serial.println(input->params.scale, 6);
    Serial.print("  Input zero_point: ");
    Serial.println(input->params.zero_point);
    Serial.print("  Output scale: ");
    Serial.println(output->params.scale, 6);
    Serial.print("  Output zero_point: ");
    Serial.println(output->params.zero_point);
  }

  Serial.println("\n========================================");
  Serial.println("Setup complete! Starting inference...");
  Serial.println("========================================\n");

  delay(1000);
}

void loop() {
  // Test values: angles from 0 to 2π in steps of π/6
  const int num_tests = 13;
  const float test_angles[] = {
    0.0, 0.523599, 1.047198, 1.570796, 2.094395, 2.617994,
    3.141593, 3.665191, 4.18879, 4.712389, 5.235988, 5.759587, 6.283185
  };
  const char* angle_labels[] = {
    "0", "π/6", "π/3", "π/2", "2π/3", "5π/6",
    "π", "7π/6", "4π/3", "3π/2", "5π/3", "11π/6", "2π"
  };

  for (int i = 0; i < num_tests; i++) {
    float angle = test_angles[i];

    // Normalize angle to [0, 1] range as done in training
    // Formula: (x + π) / (2π)
    float x_normalized = (angle + 3.14159265359) / (2.0 * 3.14159265359);

    // Helper function to set input based on tensor type
    // Match Python's quantization: truncate, no rounding
    auto setInput = [](TfLiteTensor* tensor, int idx, float value) {
      if (tensor->type == kTfLiteInt8) {
        // Quantize exactly like Python: (value / scale + zero_point).astype(int8)
        int32_t quantized_value = (int32_t)(value / tensor->params.scale + tensor->params.zero_point);
        // Clamp to int8 range
        if (quantized_value > 127) quantized_value = 127;
        if (quantized_value < -128) quantized_value = -128;
        tensor->data.int8[idx] = (int8_t)quantized_value;
      } else {
        tensor->data.f[idx] = value;
      }
    };

    // Helper function to get output based on tensor type
    // Match Python's dequantization
    auto getOutput = [](TfLiteTensor* tensor, int idx) -> float {
      if (tensor->type == kTfLiteInt8) {
        // Dequantize exactly like Python: (int8 - zero_point) * scale
        return (static_cast<float>(tensor->data.int8[idx]) - static_cast<float>(tensor->params.zero_point)) * tensor->params.scale;
      } else {
        return tensor->data.f[idx];
      }
    };

    // Run inference for SIN: input = [x_norm, 1, 0, 0]
    setInput(input, 0, x_normalized);
    setInput(input, 1, 1.0);  // is_sin
    setInput(input, 2, 0.0);  // is_cos
    setInput(input, 3, 0.0);  // is_tan

    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("ERROR: Sin Invoke() failed!");
      continue;
    }
    float sin_output = getOutput(output, 0);

    // Run inference for COS: input = [x_norm, 0, 1, 0]
    setInput(input, 0, x_normalized);
    setInput(input, 1, 0.0);  // is_sin
    setInput(input, 2, 1.0);  // is_cos
    setInput(input, 3, 0.0);  // is_tan

    invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("ERROR: Cos Invoke() failed!");
      continue;
    }
    float cos_output = getOutput(output, 0);

    // Run inference for TAN: input = [x_norm, 0, 0, 1]
    setInput(input, 0, x_normalized);
    setInput(input, 1, 0.0);  // is_sin
    setInput(input, 2, 0.0);  // is_cos
    setInput(input, 3, 1.0);  // is_tan

    invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("ERROR: Tan Invoke() failed!");
      continue;
    }
    float tan_output = getOutput(output, 0);

    // Calculate actual values for comparison
    float actual_sin = sin(angle);
    float actual_cos = cos(angle);
    float actual_tan = tan(angle);

    // Helper function to print both absolute and relative error
    auto printError = [](float predicted, float actual) {
      float abs_error = abs(predicted - actual);
      Serial.print("abs=");
      Serial.print(abs_error, 4);
      // Calculate relative error with threshold to avoid division by zero
      float rel_error = (abs_error / max(abs(actual), 0.01f)) * 100.0;
      Serial.print(", rel=");
      Serial.print(rel_error, 2);
      Serial.print("%");
    };

    // Print results
    Serial.print("Angle: ");
    Serial.print(angle_labels[i]);
    Serial.print(" (");
    Serial.print(angle, 6);
    Serial.println(" rad)");

    Serial.print("  Sin - Pred: ");
    Serial.print(sin_output, 4);
    Serial.print(", Actual: ");
    Serial.print(actual_sin, 4);
    Serial.print(", Err: ");
    printError(sin_output, actual_sin);
    Serial.println();

    Serial.print("  Cos - Pred: ");
    Serial.print(cos_output, 4);
    Serial.print(", Actual: ");
    Serial.print(actual_cos, 4);
    Serial.print(", Err: ");
    printError(cos_output, actual_cos);
    Serial.println();

    Serial.print("  Tan - Pred: ");
    Serial.print(tan_output, 4);
    Serial.print(", Actual: ");
    Serial.print(actual_tan, 4);
    Serial.print(", Err: ");
    printError(tan_output, actual_tan);
    Serial.println();

    Serial.println();
    delay(500);  // Delay between tests
  }

  Serial.println("========================================");
  Serial.println("All tests complete! Restarting in 5s...");
  Serial.println("========================================\n");
  delay(5000);
}
