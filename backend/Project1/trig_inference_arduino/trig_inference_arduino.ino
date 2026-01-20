#include <Adafruit_TFLite.h>

/*
 * Trigonometric Model Inference on Arduino Nano 33 BLE
 * CST-440 - Machine Learning on Microcontrollers
 *
 * This sketch runs inference on a TensorFlow Lite model that computes
 * trigonometric functions (sin, cos, tan) for a given input angle.
 *
 * Model uses 3 inputs: [x_normalized, is_sin, is_cos]
 * tan(x) is derived as sin(x)/cos(x)
 */

#include <Adafruit_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "trig_model_int8_data.h"

// TensorFlow Lite globals
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  // Memory allocation for TensorFlow Lite
  constexpr int kTensorArenaSize = 50 * 1024;  // 50KB for int8 model
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;
  }

  Serial.println("========================================");
  Serial.println("TensorFlow Lite Trig Model - Arduino");
  Serial.println("CST-440 Project");
  Serial.println("Model: INT8 Quantized (3 inputs)");
  Serial.println("========================================");

  tflite::InitializeTarget();

  model = tflite::GetModel(trig_model_int8_tflite);
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
  Serial.print(trig_model_int8_tflite_len);
  Serial.println(" bytes");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    while(1);
  }

  Serial.println("Tensors allocated successfully!");

  input = interpreter->input(0);
  output = interpreter->output(0);

  // Print input tensor info - CRITICAL FOR DEBUGGING
  Serial.println("\n--- INPUT TENSOR INFO ---");
  Serial.print("  Dimensions: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();
  Serial.print("  Type code: ");
  Serial.print(input->type);
  Serial.print(" (");
  if (input->type == kTfLiteInt8) Serial.print("INT8");
  else if (input->type == kTfLiteFloat32) Serial.print("FLOAT32");
  else Serial.print("OTHER");
  Serial.println(")");

  // Print output tensor info
  Serial.println("\n--- OUTPUT TENSOR INFO ---");
  Serial.print("  Dimensions: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print(" x ");
  }
  Serial.println();
  Serial.print("  Type code: ");
  Serial.print(output->type);
  Serial.print(" (");
  if (output->type == kTfLiteInt8) Serial.print("INT8");
  else if (output->type == kTfLiteFloat32) Serial.print("FLOAT32");
  else Serial.print("OTHER");
  Serial.println(")");

  // Print quantization parameters
  Serial.println("\n--- QUANTIZATION PARAMETERS ---");
  Serial.print("  Input scale: ");
  Serial.println(input->params.scale, 8);
  Serial.print("  Input zero_point: ");
  Serial.println(input->params.zero_point);
  Serial.print("  Output scale: ");
  Serial.println(output->params.scale, 8);
  Serial.print("  Output zero_point: ");
  Serial.println(output->params.zero_point);

  Serial.println("\n========================================");
  Serial.println("Setup complete! Starting inference...");
  Serial.println("========================================\n");

  delay(1000);
}

void loop() {
  // Test using EXACT same range as Python training: [-3.14, 3.14]
  // This matches: x_base = np.linspace(-3.14, 3.14, num_samples)
  const int num_tests = 13;
  const float test_angles[] = {
    -3.14, -2.618, -2.094, -1.571, -1.047, -0.524,
    0.0, 0.524, 1.047, 1.571, 2.094, 2.618, 3.14
  };
  const char* angle_labels[] = {
    "-pi", "-5pi/6", "-2pi/3", "-pi/2", "-pi/3", "-pi/6",
    "0", "pi/6", "pi/3", "pi/2", "2pi/3", "5pi/6", "pi"
  };

  // Accumulators for accuracy computation
  float sin_total_error = 0;
  float cos_total_error = 0;
  float tan_total_error = 0;
  int sin_correct = 0;  // Within 0.05 threshold
  int cos_correct = 0;
  int tan_correct = 0;
  int tan_count = 0;  // tan has fewer valid samples

  const float TOLERANCE = 0.05;

  for (int i = 0; i < num_tests; i++) {
    float x_val = test_angles[i];

    // Normalize EXACTLY like Python: (x + 3.14) / (2 * 3.14)
    float x_normalized = (x_val + 3.14f) / (2.0f * 3.14f);

    // Debug: print normalized value and quantized inputs for first angle
    if (i == 0) {
      Serial.print("DEBUG: x_val=");
      Serial.print(x_val, 4);
      Serial.print(", x_normalized=");
      Serial.println(x_normalized, 6);

      // Show what int8 values we're sending
      float q0 = x_normalized / input->params.scale + input->params.zero_point;
      float q1 = 1.0f / input->params.scale + input->params.zero_point;  // is_sin=1
      float q2 = 0.0f / input->params.scale + input->params.zero_point;  // is_cos=0
      Serial.print("DEBUG quantized inputs: [");
      Serial.print((int8_t)(int32_t)q0);
      Serial.print(", ");
      Serial.print((int8_t)(int32_t)q1);
      Serial.print(", ");
      Serial.print((int8_t)(int32_t)q2);
      Serial.println("]");
    }

    // Helper to set input (handles both int8 and float32)
    auto setInput = [](TfLiteTensor* tensor, int idx, float value) {
      if (tensor->type == kTfLiteInt8) {
        // Quantize: q = value / scale + zero_point
        float q = value / tensor->params.scale + tensor->params.zero_point;
        int32_t quantized = (int32_t)q;
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        tensor->data.int8[idx] = (int8_t)quantized;
      } else {
        tensor->data.f[idx] = value;
      }
    };

    // Helper to get output
    auto getOutput = [](TfLiteTensor* tensor, int idx) -> float {
      if (tensor->type == kTfLiteInt8) {
        // Dequantize: value = (q - zero_point) * scale
        return (float(tensor->data.int8[idx]) - float(tensor->params.zero_point)) * tensor->params.scale;
      } else {
        return tensor->data.f[idx];
      }
    };

    // Run inference for SIN: input = [x_norm, 1, 0]
    setInput(input, 0, x_normalized);
    setInput(input, 1, 1.0f);
    setInput(input, 2, 0.0f);

    interpreter->Invoke();
    float sin_pred = getOutput(output, 0);

    // Debug: show raw int8 output for first angle
    if (i == 0) {
      Serial.print("DEBUG sin raw int8 output: ");
      Serial.print(output->data.int8[0]);
      Serial.print(" -> dequantized: ");
      Serial.println(sin_pred, 4);
    }

    // Run inference for COS: input = [x_norm, 0, 1]
    setInput(input, 0, x_normalized);
    setInput(input, 1, 0.0f);
    setInput(input, 2, 1.0f);

    interpreter->Invoke();
    float cos_pred = getOutput(output, 0);

    // Derive TAN = SIN / COS
    float tan_pred = 0.0f;
    bool tan_valid = false;
    if (fabs(cos_pred) > 0.01f) {
      tan_pred = sin_pred / cos_pred;
      tan_valid = true;
    }

    // Actual values
    float sin_actual = sin(x_val);
    float cos_actual = cos(x_val);
    float tan_actual = tan(x_val);

    // Calculate errors
    float sin_err = fabs(sin_pred - sin_actual);
    float cos_err = fabs(cos_pred - cos_actual);
    float tan_err = fabs(tan_pred - tan_actual);

    // Accumulate for accuracy
    sin_total_error += sin_err;
    cos_total_error += cos_err;
    if (sin_err < TOLERANCE) sin_correct++;
    if (cos_err < TOLERANCE) cos_correct++;

    if (tan_valid && fabs(tan_actual) < 3.0f) {
      tan_total_error += tan_err;
      tan_count++;
      if (tan_err < TOLERANCE) tan_correct++;
    }

    // Print results
    Serial.print("Angle: ");
    Serial.print(angle_labels[i]);
    Serial.print(" (");
    Serial.print(x_val, 4);
    Serial.println(" rad)");

    Serial.print("  Sin: pred=");
    Serial.print(sin_pred, 4);
    Serial.print(" actual=");
    Serial.print(sin_actual, 4);
    Serial.print(" err=");
    Serial.println(sin_err, 4);

    Serial.print("  Cos: pred=");
    Serial.print(cos_pred, 4);
    Serial.print(" actual=");
    Serial.print(cos_actual, 4);
    Serial.print(" err=");
    Serial.println(cos_err, 4);

    if (tan_valid && fabs(tan_actual) < 10.0f) {
      Serial.print("  Tan: pred=");
      Serial.print(tan_pred, 4);
      Serial.print(" actual=");
      Serial.print(tan_actual, 4);
      Serial.print(" err=");
      Serial.println(tan_err, 4);
    } else {
      Serial.println("  Tan: skipped (near asymptote)");
    }

    Serial.println();
    delay(100);
  }

  // Print accuracy summary
  Serial.println("========================================");
  Serial.println("ACCURACY SUMMARY");
  Serial.println("========================================");
  Serial.print("Tolerance threshold: ");
  Serial.println(TOLERANCE, 2);
  Serial.println();

  Serial.print("SIN - Accuracy: ");
  Serial.print((float)sin_correct / num_tests * 100.0f, 1);
  Serial.print("% (");
  Serial.print(sin_correct);
  Serial.print("/");
  Serial.print(num_tests);
  Serial.print("), MAE: ");
  Serial.println(sin_total_error / num_tests, 6);

  Serial.print("COS - Accuracy: ");
  Serial.print((float)cos_correct / num_tests * 100.0f, 1);
  Serial.print("% (");
  Serial.print(cos_correct);
  Serial.print("/");
  Serial.print(num_tests);
  Serial.print("), MAE: ");
  Serial.println(cos_total_error / num_tests, 6);

  if (tan_count > 0) {
    Serial.print("TAN - Accuracy: ");
    Serial.print((float)tan_correct / tan_count * 100.0f, 1);
    Serial.print("% (");
    Serial.print(tan_correct);
    Serial.print("/");
    Serial.print(tan_count);
    Serial.print("), MAE: ");
    Serial.println(tan_total_error / tan_count, 6);
  }

  Serial.println("\n========================================");
  Serial.println("Test complete! Restarting in 30s...");
  Serial.println("========================================\n");
  delay(30000);
}