#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <math.h>  // Added math.h for expf, sqrt, roundf

// TFLM includes 
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tiny_seizure_model_data.h"  // 

// ----- ADS1115 -----
Adafruit_ADS1115 ads;
const int ADC_CHANNEL = 0;             // A0 (single channel input)
const float ADS1115_LSB_UV = 7.8125F;  // microvolts per bit at GAIN_SIXTEEN

// ----- Sampling / Buffer -----
const int SAMPLE_RATE_HZ = 250;                         // target sampling rate
const int SAMPLE_PERIOD_US = 1000000 / SAMPLE_RATE_HZ;  // microseconds per sample
const int WINDOW_SIZE = 256;                            // model input length

float eeg_buffer[WINDOW_SIZE];  // holds microvolt values

// ----- Model / TFLM config -----
constexpr int kTensorArenaSize = 48 * 1024;  // adjust if needed
static uint8_t tensor_arena[kTensorArenaSize];

// alert system
int buzzPin = 12;
int redPin = 33;
int greenPin = 32;

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = nullptr;

// FIX: Global pointers for the TFLite components
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;


void printTensorInfo(TfLiteTensor* t, const char* name) {
  Serial.printf("%s dims: [", name);
  for (int i = 0; i < t->dims->size; i++) {
    Serial.print(t->dims->data[i]);
    if (i < t->dims->size - 1) Serial.print(",");
  }
  Serial.println("]");

  Serial.printf("  type=%d scale=%f zero_point=%d bytes=%d\n",
                t->type, t->params.scale, t->params.zero_point, t->bytes);
}

// These must match the values printed earlier on your board
const float MODEL_INPUT_SCALE = 1.101878F;  // from earlier serial output
const int MODEL_INPUT_ZERO_POINT = -8;      // from earlier serial output

// Helper: compute mean and stddev
void compute_mean_std(const float* arr, int n, float& mean, float& stddev) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += arr[i];
  mean = (float)(s / n);
  double sq = 0.0;
  for (int i = 0; i < n; ++i) {
    double d = arr[i] - mean;
    sq += d * d;
  }
  stddev = (float)sqrt(sq / n);
  if (stddev < 1e-6) stddev = 1.0f;  // avoid divide by zero
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 Seizure Detection (Realtime) ===");

  // Initialize ADS1115
  if (!ads.begin()) {
    Serial.println("ERROR: ADS1115 not found. Check wiring.");
    while (1) delay(1000);
  }
  Serial.println("ADS1115 found. Using GAIN_SIXTEEN, 250SPS");
  ads.setGain(GAIN_SIXTEEN);
  ads.setDataRate(RATE_ADS1115_250SPS);

  // Initialize TFLM **ONCE** (Memory Fix)
  error_reporter = &micro_error_reporter;
  const tflite::Model* model = tflite::GetModel(tiny_seizure_model_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    while (1) delay(1000);
  }
  Serial.println("Model loaded.");

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  // Assign to the global pointer
  interpreter = &static_interpreter;  // FIX: Use global pointer

  // Allocate Tensors **ONCE**
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed in setup()");
    while (1) delay(1000);
  }
  Serial.println("Tensors allocated.");

  // Assign input/output tensors to global pointers
  input_tensor = interpreter->input(0);    // FIX: Use global pointer
  output_tensor = interpreter->output(0);  // FIX: Use global pointer

  printTensorInfo(input_tensor, "Model Input");
  printTensorInfo(output_tensor, "Model Output");

  // Warm up buffer with zeros
  for (int i = 0; i < WINDOW_SIZE; ++i) eeg_buffer[i] = 0.0f;

  Serial.println("Setup complete. Starting acquisition...");
  delay(200);

  // alert
  pinMode(buzzPin, OUTPUT);
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
}

// Run one inference on the buffer (preprocessed + quantized)
bool run_inference_and_print() {
  // FIX: Removed all TFLite local declarations and the redundant AllocateTensors() call.
  // The global 'interpreter', 'input_tensor', and 'output_tensor' are used automatically.

  // Preprocessing: mean removal and std normalization
  float mean = 0.0f, stddev = 1.0f;
  compute_mean_std(eeg_buffer, WINDOW_SIZE, mean, stddev);

  // Debug output for data integrity
  Serial.printf("Buffer Stats: Mean=%.6f, StdDev=%.6f\n", mean, stddev);

  // Fill input tensor with quantized int8 values
  if (input_tensor->type != kTfLiteInt8) {  // FIX: Use global input_tensor
    Serial.println("ERROR: Model input not int8 as expected.");
    return false;
  }

  int8_t* in_data = input_tensor->data.int8;  // FIX: Use global input_tensor
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    // normalized value (unitless)
    float normalized = (eeg_buffer[i] - mean) / stddev;

    // quantize according to model params: q = round(normalized / scale) + zero_point
    float q = roundf(normalized / MODEL_INPUT_SCALE) + (float)MODEL_INPUT_ZERO_POINT;

    // clamp to int8 range
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    in_data[i] = (int8_t)((int)q);
  }

  // Run the inference
  if (interpreter->Invoke() != kTfLiteOk) {  // FIX: Use global interpreter
    Serial.println("ERROR: Invoke failed");
    return false;
  }

  // Read output (assume output int8 with two classes)
  if (output_tensor->type != kTfLiteInt8) {  // FIX: Use global output_tensor
    Serial.println("ERROR: Output not int8");
    return false;
  }

  int out_elems = 1;
  for (int i = 0; i < output_tensor->dims->size; ++i) out_elems *= output_tensor->dims->data[i];

  // Convert to float probabilities using quantization params
  float out_scale = output_tensor->params.scale;    // FIX: Use global output_tensor
  int out_zero = output_tensor->params.zero_point;  // FIX: Use global output_tensor

  float softmax_input[8];  // support up to 8 classes if needed
  for (int i = 0; i < out_elems; ++i) {
    int8_t q = output_tensor->data.int8[i];  // FIX: Use global output_tensor
    float real = (q - out_zero) * out_scale;
    softmax_input[i] = real;
    Serial.printf("raw_out[%d] = %d -> %f\n", i, q, real);
  }
  Serial.println("---");

  // Simple softmax compute
  float maxv = softmax_input[0];
  for (int i = 1; i < out_elems; ++i)
    if (softmax_input[i] > maxv) maxv = softmax_input[i];
  float sumexp = 0.0f;
  for (int i = 0; i < out_elems; ++i) {
    softmax_input[i] = expf(softmax_input[i] - maxv);
    sumexp += softmax_input[i];
  }
  for (int i = 0; i < out_elems; ++i) softmax_input[i] /= sumexp;

  // Decide class (index of max probability)
  int best = 0;
  float bestp = softmax_input[0];
  for (int i = 1; i < out_elems; ++i) {
    if (softmax_input[i] > bestp) {
      bestp = softmax_input[i];
      best = i;
    }
  }

// normal or seizure detection: 1, 0;
  Serial.printf("Prediction -> class=%d prob=%.3f\n", best, bestp);
  if (best == 0) { // 1
    Serial.println("--- Normal ---");
    digitalWrite(greenPin, HIGH);
    delay(1000);
    digitalWrite(greenPin, LOW);
    delay(1000);
  } else {
    Serial.println("*** SEIZURE DETECTED ***");
    digitalWrite(buzzPin, HIGH);
    digitalWrite(redPin, HIGH);
    delay(1000);
    digitalWrite(buzzPin, LOW);
  }

  // alert

  return true;
}

void loop() {
  // Acquire WINDOW_SIZE samples at SAMPLE_RATE_HZ from ADS1115 A0
  unsigned long start_us = micros();
  for (int i = 0; i < WINDOW_SIZE; ++i) {
    unsigned long t0 = micros();

    int16_t raw = ads.readADC_SingleEnded(ADC_CHANNEL);  // raw ADC counts

    // Debug: Print Raw ADC count once per window (at start)
    if (i == 0) {
      Serial.printf("Raw ADC Start: %d\n", raw);
    }

    // convert to microvolts
    float uV = raw * ADS1115_LSB_UV;
    eeg_buffer[i] = uV;

    // Sleep until next sample
    unsigned long dt = micros() - t0;
    if (dt < SAMPLE_PERIOD_US) {
      unsigned long sleep_us = SAMPLE_PERIOD_US - dt;
      delayMicroseconds(sleep_us);
    } else {
      // sampling overrun; we still continue (timing not exact)
    }
  }
  unsigned long total_us = micros() - start_us;
  Serial.printf("Collected %d samples in %lu ms\n", WINDOW_SIZE, total_us / 1000UL);

  // Run inference on buffer
  bool ok = run_inference_and_print();
  if (!ok) {
    Serial.println("Inference error.");
  }

  // For continuous monitoring we can slide window or re-sample
  // Here we wait briefly before next window (no overlap). You can implement sliding windows later.
  delay(50);
}