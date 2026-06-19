# CNN CIFAR-10 — Fixed-Point Golden Model

End-to-end pipeline from PyTorch training to fixed-point inference,
designed to match the Verilog hardware implementation bit-for-bit.

## Repository contents

| File | Description |
|------|-------------|
| `cnn_pytorch.py` | Trains the CNN on CIFAR-10 in floating point. Exports `cnn_cifar10.pth` and `cnn_weights_float.json`. |
| `cnn_cifar10.pth` | Best PyTorch checkpoint (68.3% test accuracy). |
| `cnn_weights_float.json` | Trained weights as plain float, one entry per named parameter. |
| `quantize_weights.py` | Converts float weights to Q4.6 fixed-point (SCALE=64, 11-bit signed). Outputs `cnn_weights_q4_6.json`. |
| `cnn_weights_q4_6.json` | Quantized weights ready for the golden model and Verilog. |
| `golden_model.py` | Full fixed-point inference in Python. Matches the Verilog implementation. Validates against PyTorch (95% agreement on 20 test images). |
| `data/` | CIFAR-10 dataset downloaded automatically by PyTorch. |

## Architecture

```
Input  32x32x3  (Q1.6, 8-bit signed per channel)
Conv1  3x3,  8 filters, ReLU  →  32x32x8
Pool1  2x2                    →  16x16x8
Conv2  3x3, 16 filters, ReLU  →  16x16x16
Pool2  2x2                    →   8x8x16
Conv3  3x3, 32 filters, ReLU  →   8x8x32
Pool3  2x2                    →   4x4x32
Flatten                       →  512
FC1    512 → 64, ReLU
FC2     64 → 10  (logits)
```

~100K parameters. Trained on CIFAR-10 (10 classes, 32×32 RGB images).

## Fixed-point format

Matches `parameters.svh` in the Verilog design exactly.

| Parameter | Value |
|-----------|-------|
| `FRAC_BITS` | 6 |
| `BITS_Q4_6` | 11 (signed) |
| `SCALE` | 64 (= 2^6) |
| Integer range | [−1024, 1023] |
| Float range | [−16.0, 15.984375] |
| Resolution | 0.015625 (= 1/64) |

## How to run

### 1 — Train (only needed if retraining from scratch)
```bash
pip install torch torchvision
python3 cnn_pytorch.py
```
Outputs: `cnn_cifar10.pth`, `cnn_weights_float.json`

### 2 — Quantize
```bash
python3 quantize_weights.py
```
Outputs: `cnn_weights_q4_6.json`

### 3 — Run golden model
```bash
python3 golden_model.py
```
Compares fixed-point inference against PyTorch on 20 test images and prints
layer-by-layer activation ranges for image 0.

## Golden model vs Verilog mapping

| Python (`golden_model.py`) | Verilog |
|----------------------------|---------|
| `conv_core()` | `conv_core.sv` — accumulate all products, single shift, round-to-nearest, ReLU |
| `conv2d_q()` | `conv_control.sv` + `conv_layer.sv` — sliding window FSM, 8/16/32 parallel kernels |
| `maxpool2d()` | `max_pooling_core.sv` + `max_pooling_control.sv` — signed 2×2 max |
| `linear_q()` | FC layers (not yet implemented in RTL) |

## Verilog implementation status

| Block | Status |
|-------|--------|
| Conv1 (8 kernels) | Done — `conv_layer.sv`, `conv_control.sv`, `conv_core.sv` |
| MaxPool1 | Done — `max_pooling_core.sv`, `max_pooling_control.sv` |
| Conv1 + Pool1 top | Done — `top_conv1_pool1.sv` |
| Conv2, Conv3 | Pending |
| MaxPool2, MaxPool3 | Pending |
| FC1, FC2 | Pending |