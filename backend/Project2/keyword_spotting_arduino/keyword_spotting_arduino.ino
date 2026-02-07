#include <Chirale_TensorFlowLite.h>

/*
 * Keyword Spotting with Wake Word on Arduino Nano 33 BLE
 * CST-440 - Machine Learning on Microcontrollers
 *
 * State machine:
 *   WAITING  --[hears "sheila"]--> LISTENING --[25s timeout]--> WAITING
 *
 * In WAITING mode, only responds to "sheila" (wake word). Outputs 0.
 * In LISTENING mode, detects keywords (down/off/on/up/wow) and outputs 1
 * when a keyword is detected. Returns to WAITING after 25 seconds.
 *
 * Audio: PDM microphone at 16kHz, 1-second buffer
 * Features: 13 MFCCs + 13 delta MFCCs = 26 features, 49 frames (30ms window, 20ms stride)
 * Model: GRU(48) -> GRU(48) -> 8-class softmax (float32 TFLite)
 */

#include <PDM.h>
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <arm_math.h>

#include "kws_model_data.h"

// ============================================================
// Audio configuration
// ============================================================
static const int kSampleRate = 16000;
static const int kAudioBufferSize = 16000;  // 1 second of audio
static int16_t g_audio_buffer[kAudioBufferSize];
static volatile int g_audio_write_index = 0;
static volatile bool g_audio_ready = false;

// ============================================================
// MFCC configuration
// ============================================================
static const int kNumFrames = 49;
static const int kNumMfcc = 13;
static const int kNumFeatures = 26;     // 13 MFCCs + 13 delta MFCCs
static const int kFftSize = 512;        // Next power of 2 >= 480
static const int kWindowSize = 480;     // 30ms at 16kHz
static const int kHopLength = 320;      // 20ms stride at 16kHz
static const int kNumMelBins = 40;
static const float kSampleRateF = 16000.0f;
static const int kDeltaHalfWidth = 4;   // librosa default: width=9, half=4

// Mel filterbank (precomputed for 40 bins, 0-8000 Hz, 512-point FFT)
static float g_mel_filterbank[kNumMelBins][kFftSize / 2 + 1];

// Hann window
static float g_hann_window[kWindowSize];

// DCT-II matrix for MFCC extraction
static float g_dct_matrix[kNumMfcc][kNumMelBins];

// MFCC output buffer (13 MFCCs only, before delta computation)
static float g_mfcc_raw[kNumFrames][kNumMfcc];

// Full feature buffer: 13 MFCCs + 13 delta MFCCs = 26 features per frame
static float g_mfcc_features[kNumFrames][kNumFeatures];

// FFT instance
static arm_rfft_fast_instance_f32 g_fft_instance;

// ============================================================
// TFLite configuration
// ============================================================
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* output_tensor = nullptr;

  constexpr int kTensorArenaSize = 60 * 1024;  // 60 KB
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// ============================================================
// State machine
// ============================================================
enum State {
  STATE_WAITING,
  STATE_LISTENING
};

static State g_state = STATE_WAITING;
static unsigned long g_listen_start_ms = 0;
static const unsigned long kListenTimeoutMs = 25000;  // 25 seconds
static const float kConfidenceThreshold = 0.6f;

// Class indices (must match label_map.json)
static const int kIdxDown    = 0;
static const int kIdxOff     = 1;
static const int kIdxOn      = 2;
static const int kIdxSheila  = 3;
static const int kIdxSilence = 4;
static const int kIdxUnknown = 5;
static const int kIdxUp      = 6;
static const int kIdxWow     = 7;

// Track when a keyword detection started (for 1-second output window)
static unsigned long g_keyword_detected_ms = 0;
static const unsigned long kKeywordOutputMs = 1000;  // Output 1 for 1 second

// ============================================================
// Helper: Hz to Mel and Mel to Hz
// ============================================================
static float hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

static float mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

// ============================================================
// Initialize mel filterbank
// ============================================================
static void init_mel_filterbank() {
  float mel_low = hz_to_mel(0.0f);
  float mel_high = hz_to_mel(kSampleRateF / 2.0f);

  // Compute mel-spaced center frequencies
  float mel_points[kNumMelBins + 2];
  for (int i = 0; i < kNumMelBins + 2; i++) {
    mel_points[i] = mel_low + (mel_high - mel_low) * i / (kNumMelBins + 1);
  }

  // Convert to frequency bin indices
  float bin_points[kNumMelBins + 2];
  for (int i = 0; i < kNumMelBins + 2; i++) {
    float hz = mel_to_hz(mel_points[i]);
    bin_points[i] = hz * kFftSize / kSampleRateF;
  }

  // Build triangular filters
  int num_fft_bins = kFftSize / 2 + 1;
  for (int m = 0; m < kNumMelBins; m++) {
    for (int k = 0; k < num_fft_bins; k++) {
      float fk = (float)k;
      if (fk < bin_points[m]) {
        g_mel_filterbank[m][k] = 0.0f;
      } else if (fk <= bin_points[m + 1]) {
        g_mel_filterbank[m][k] = (fk - bin_points[m]) / (bin_points[m + 1] - bin_points[m]);
      } else if (fk <= bin_points[m + 2]) {
        g_mel_filterbank[m][k] = (bin_points[m + 2] - fk) / (bin_points[m + 2] - bin_points[m + 1]);
      } else {
        g_mel_filterbank[m][k] = 0.0f;
      }
    }
  }
}

// ============================================================
// Initialize Hann window
// ============================================================
static void init_hann_window() {
  for (int i = 0; i < kWindowSize; i++) {
    g_hann_window[i] = 0.5f * (1.0f - cosf(2.0f * PI * i / kWindowSize));
  }
}

// ============================================================
// Initialize DCT-II matrix
// ============================================================
static void init_dct_matrix() {
  for (int i = 0; i < kNumMfcc; i++) {
    for (int j = 0; j < kNumMelBins; j++) {
      g_dct_matrix[i][j] = cosf(PI * i * (j + 0.5f) / kNumMelBins);
    }
  }
}

// ============================================================
// Extract MFCCs from audio buffer
// ============================================================
static void extract_mfcc(const int16_t* audio, int audio_len) {
  float fft_input[kFftSize];
  float fft_output[kFftSize];
  float power_spectrum[kFftSize / 2 + 1];
  float mel_energies[kNumMelBins];
  int num_fft_bins = kFftSize / 2 + 1;

  for (int frame = 0; frame < kNumFrames; frame++) {
    int start = frame * kHopLength;

    // Apply Hann window and convert int16 to float, zero-pad to kFftSize
    for (int i = 0; i < kFftSize; i++) {
      if (i < kWindowSize && (start + i) < audio_len) {
        fft_input[i] = (float)audio[start + i] / 32768.0f * g_hann_window[i];
      } else {
        fft_input[i] = 0.0f;
      }
    }

    // FFT
    arm_rfft_fast_f32(&g_fft_instance, fft_input, fft_output, 0);

    // Compute power spectrum
    // First bin (DC): fft_output[0]^2
    power_spectrum[0] = fft_output[0] * fft_output[0];
    // Nyquist bin: fft_output[1]^2 (packed format)
    power_spectrum[num_fft_bins - 1] = fft_output[1] * fft_output[1];
    // Remaining bins: real^2 + imag^2
    for (int i = 1; i < num_fft_bins - 1; i++) {
      float real = fft_output[2 * i];
      float imag = fft_output[2 * i + 1];
      power_spectrum[i] = real * real + imag * imag;
    }

    // Apply mel filterbank
    for (int m = 0; m < kNumMelBins; m++) {
      mel_energies[m] = 0.0f;
      for (int k = 0; k < num_fft_bins; k++) {
        mel_energies[m] += power_spectrum[k] * g_mel_filterbank[m][k];
      }
      // Log (with floor to avoid log(0))
      mel_energies[m] = logf(mel_energies[m] + 1e-10f);
    }

    // DCT-II to get MFCCs
    for (int i = 0; i < kNumMfcc; i++) {
      float sum = 0.0f;
      for (int j = 0; j < kNumMelBins; j++) {
        sum += g_dct_matrix[i][j] * mel_energies[j];
      }
      g_mfcc_raw[frame][i] = sum;
    }
  }

  // Compute delta MFCCs using Savitzky-Golay first-derivative filter (width=9)
  // delta[t] = sum(n * mfcc[t+n] for n in -4..4) / sum(n^2 for n in -4..4)
  // Denominator for half_width=4: 2*(1+4+9+16) = 60
  float denom = 0.0f;
  for (int n = 1; n <= kDeltaHalfWidth; n++) {
    denom += (float)(n * n);
  }
  denom *= 2.0f;  // = 60

  for (int frame = 0; frame < kNumFrames; frame++) {
    // Copy raw MFCCs into first 13 features
    for (int i = 0; i < kNumMfcc; i++) {
      g_mfcc_features[frame][i] = g_mfcc_raw[frame][i];
    }

    // Compute delta for each MFCC coefficient
    for (int i = 0; i < kNumMfcc; i++) {
      float delta = 0.0f;
      for (int n = -kDeltaHalfWidth; n <= kDeltaHalfWidth; n++) {
        // Edge padding: clamp index to valid range (matches librosa 'edge' mode)
        int idx = frame + n;
        if (idx < 0) idx = 0;
        if (idx >= kNumFrames) idx = kNumFrames - 1;
        delta += (float)n * g_mfcc_raw[idx][i];
      }
      g_mfcc_features[frame][kNumMfcc + i] = delta / denom;
    }
  }

  // Normalize all 26 features using training statistics
  for (int frame = 0; frame < kNumFrames; frame++) {
    for (int i = 0; i < kNumFeatures; i++) {
      g_mfcc_features[frame][i] =
          (g_mfcc_features[frame][i] - kMfccMean[i]) / kMfccStd[i];
    }
  }
}

// ============================================================
// PDM callback - fills audio buffer
// ============================================================
void onPDMdata() {
  int bytes_available = PDM.available();
  int samples_available = bytes_available / 2;

  int16_t temp_buffer[512];
  int to_read = min(samples_available, 512);
  PDM.read(temp_buffer, to_read * 2);

  for (int i = 0; i < to_read; i++) {
    g_audio_buffer[g_audio_write_index] = temp_buffer[i];
    g_audio_write_index++;
    if (g_audio_write_index >= kAudioBufferSize) {
      g_audio_write_index = 0;
      g_audio_ready = true;
    }
  }
}

// ============================================================
// Setup
// ============================================================
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  Serial.println("========================================");
  Serial.println("Keyword Spotting with Wake Word");
  Serial.println("CST-440 Project 2");
  Serial.println("========================================");

  // Initialize signal processing
  init_hann_window();
  init_mel_filterbank();
  init_dct_matrix();
  arm_rfft_fast_init_f32(&g_fft_instance, kFftSize);
  Serial.println("Signal processing initialized.");

  // Initialize TFLite
  tflite::InitializeTarget();

  model = tflite::GetModel(kws_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch!");
    while (1) { ; }
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    while (1) { ; }
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  Serial.print("Model loaded. Input shape: ");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    if (i < input_tensor->dims->size - 1) Serial.print("x");
  }
  Serial.print(", Output shape: ");
  for (int i = 0; i < output_tensor->dims->size; i++) {
    Serial.print(output_tensor->dims->data[i]);
    if (i < output_tensor->dims->size - 1) Serial.print("x");
  }
  Serial.println();

  // Start PDM microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, kSampleRate)) {
    Serial.println("ERROR: Failed to start PDM microphone!");
    while (1) { ; }
  }
  Serial.println("PDM microphone started.");

  Serial.println("\nState: WAITING for wake word 'sheila'...");
  Serial.println("========================================\n");
}

// ============================================================
// Run inference and return predicted class index + confidence
// ============================================================
static int run_inference(float* confidence) {
  // Copy MFCC + delta features into input tensor
  for (int frame = 0; frame < kNumFrames; frame++) {
    for (int coeff = 0; coeff < kNumFeatures; coeff++) {
      input_tensor->data.f[frame * kNumFeatures + coeff] = g_mfcc_features[frame][coeff];
    }
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Invoke() failed!");
    return -1;
  }

  // Find class with highest probability
  int best_class = 0;
  float best_prob = output_tensor->data.f[0];
  for (int i = 1; i < kNumClasses; i++) {
    if (output_tensor->data.f[i] > best_prob) {
      best_prob = output_tensor->data.f[i];
      best_class = i;
    }
  }

  *confidence = best_prob;
  return best_class;
}

// ============================================================
// Check if class is a keyword (not wake word, silence, or unknown)
// ============================================================
static bool is_keyword(int class_idx) {
  return class_idx == kIdxDown ||
         class_idx == kIdxOff ||
         class_idx == kIdxOn ||
         class_idx == kIdxUp ||
         class_idx == kIdxWow;
}

// ============================================================
// Main loop
// ============================================================
void loop() {
  if (!g_audio_ready) {
    return;
  }
  g_audio_ready = false;

  // Extract MFCC features from the audio buffer
  extract_mfcc(g_audio_buffer, kAudioBufferSize);

  // Run inference
  float confidence = 0.0f;
  int predicted_class = run_inference(&confidence);
  if (predicted_class < 0) return;

  unsigned long now = millis();

  switch (g_state) {
    case STATE_WAITING:
      if (predicted_class == kIdxSheila && confidence >= kConfidenceThreshold) {
        g_state = STATE_LISTENING;
        g_listen_start_ms = now;
        g_keyword_detected_ms = 0;
        Serial.println("[WAKE] 'sheila' detected! Listening for commands...");
      }
      // Output 0 in waiting mode
      Serial.println("0");
      break;

    case STATE_LISTENING:
      // Check timeout
      if (now - g_listen_start_ms >= kListenTimeoutMs) {
        g_state = STATE_WAITING;
        g_keyword_detected_ms = 0;
        Serial.println("[TIMEOUT] Returning to WAITING state.");
        Serial.println("0");
        break;
      }

      if (is_keyword(predicted_class) && confidence >= kConfidenceThreshold) {
        g_keyword_detected_ms = now;
        Serial.print("[KEYWORD] Detected: ");
        Serial.print(kLabelNames[predicted_class]);
        Serial.print(" (confidence: ");
        Serial.print(confidence, 2);
        Serial.println(")");
      }

      // Output 1 for 1 second after keyword detection, else 0
      if (g_keyword_detected_ms > 0 && (now - g_keyword_detected_ms) < kKeywordOutputMs) {
        Serial.println("1");
      } else {
        Serial.println("0");
      }
      break;
  }
}
