# CNN Accelerator for Image Classification

Fixed-point CNN accelerator implemented in RTL, targeting image classification tasks on GF180MCU.

**Team:** [NAME] — Universidad Nacional de Colombia  
**Process:** GF180MCU  

## Overview

This project implements the inference stage of an end-to-end image processing pipeline in silicon. The full pipeline is:

```
[Preprocessing ASIC] RGB → Grayscale → Sobel → Threshold → [CNN Accelerator] → Classification output
```

The preprocessing stage is implemented in separate ASICs that feed this CNN accelerator. The accelerator covers convolutional layers, max-pooling, and fully-connected layers using fixed-point arithmetic.

### Related repositories

| Repo | Description |
|------|-------------|
| [ttsky_grayscale_sobel](https://github.com/DianaNatali/ttsky_grayscale_sobel) | RGB-to-grayscale + Sobel ASIC (TT06, TTsky25a) |
| [tt_um_sobel_threshold](https://github.com/jharamirezma/tt_um_sobel_threshold) | Sobel + threshold detector ASIC (TTsky26b) |

### Tapeout history

| Shuttle | Process | Design | Status |
|---------|---------|--------|--------|
| TT06 | SKY130 | RGB-to-grayscale + Sobel edge detection | Fabricated |
| TTsky25a | SKY130 | Same design, bug fix | Fabricated |
| TTsky26b | SKY130B | RGB-to-grayscale + Sobel + threshold detector | Submitted |


## Architecture

```
Input  32×32×1  (grayscale)
  ↓
Conv1  3×3, 8 filters, ReLU  →  32×32×8
Pool1  2×2                   →  16×16×8
Conv2  3×3, 16 filters, ReLU →  16×16×16
Pool2  2×2                   →   8×8×16
Conv3  3×3, 32 filters, ReLU →   8×8×32
Pool3  2×2                   →   4×4×32
  ↓
FC1    512 → 64, ReLU
FC2     64 → 10  (logits)
```

## RTL Status

| Block | Status |
|-------|--------|
| Conv1 | ✅ Complete |
| MaxPool1 | ✅ Complete |
| top_conv1_pool1 | ✅ Complete |
| Conv2, Conv3 | 🔄 In progress |
| MaxPool2, MaxPool3 | 🔄 In progress |
| FC1, FC2 | 🔄 In progress |

## Python Golden Model

See [`cnn_py/README.md`](cnn_py/README.md).

## Running the Testbench

```bash
cd test
pip install -r requirements.txt
make
```

## Tools

- RTL: SystemVerilog / Verilog
- Synthesis: OpenLane / OpenROAD / Yosys
- Simulation: cocotb
- Python: PyTorch, NumPy