# ESP32 Seizure Detection: Edge AI on a Wearable Microcontroller

Real-time EEG seizure detection running fully on-device on an ESP32. No cloud, no Wi-Fi, no phone.

## Why this exists
Most seizure detection and prediction research needs a server or a GPU. I wanted to show it can run on a chip costing a few dollars, with no network, so it can work where connectivity and money are limited. Because nothing leaves the device, there is also no server to breach.

## How it works
1. Sensing: dry EEG electrodes on a headband, with reference and ground on the earlobes.
2. Acquisition: an ADS1115 16-bit ADC (gain x16, 250 SPS) sends samples to the ESP32 over I2C (SDA GPIO21, SCL GPIO22).
3. Windowing: the firmware buffers 256 samples (1 second) and normalizes each window.
4. Inference: an int8-quantized 1D-CNN runs on TensorFlow Lite Micro and classifies each window as seizure or non-seizure.
5. Alert: LEDs (red = possible seizure, green = normal) and a buzzer.

## Model
- 1D-CNN trained on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
- Input: 1 x 256 x 1 (one second at 256 Hz, one channel)
- Converted to TensorFlow Lite int8 and embedded in firmware as a C array

## Results
| Metric | Before quantization | After int8 on device |
|---|---|---|
| Accuracy | 92.5% | 68% |
| Precision | 95% | 64% |
| Recall | 85% | 63% |
| F1 | 90% | 63% |

| Resource | Result |
|---|---|
| Inference latency | 20-40 ms per 1-second window |
| Flash | 87.14 KB |
| SRAM | ~80 KB (48 KB tensor arena) |
| External memory | None |

The accuracy drop after quantization is the main thing I'm working to fix.

## Limitations (honest ones)
- This is a proof of concept, validated only on the public CHB-MIT dataset. No clinical trials.
- The firmware reads one EEG channel.
- Power consumption was not measured with instruments, only observed to be stable on a power bank.
- The EEG signal is digitized without a dedicated amplifier stage.

## Roadmap to fix
- Seizure prediction: train on pre-seizure EEG so the device warns seconds to minutes in advance
- Recover the accuracy lost in quantization (better calibration data, quantization-aware training)
- Encrypted removable memory chip that logs every prediction so doctors can read a patient's history without any cloud
- On-device personalization
- On-board amplification and filtering, measured power consumption, and a smaller PCB

## Files
| File | What it is |
|---|---|
| `thesis.ino` | ESP32 firmware (sampling, windowing, inference, alerts) |
| `tiny_seizure_model_int8.tflite` | Quantized model |
| `tiny_seizure_model_data.cc / .h` | Model as a C array for the firmware |

## Build and run
- Arduino IDE with ESP32 board support
- Libraries: `Adafruit ADS1X15`, `TensorFlowLite_ESP32`
- Open `thesis.ino`, select your ESP32 board, upload, and open the Serial Monitor at 115200 baud

## Author
Abdulkadir Umar Barambu: abdulkadirbarambu1@gmail.com
