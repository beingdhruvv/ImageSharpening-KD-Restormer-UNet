# `models/` — Student Network Architectures

This folder contains the **Mini-UNet** architecture used for the student model in our knowledge distillation framework. The design prioritizes lightweight performance while maintaining high output quality.

---

## Contents

| File | Description |
|------|-------------|
| `student_model_unet.py` | Full PyTorch implementation of the Mini-UNet architecture (student model) |
| `README.md` | Documentation for the model folder (you are here) |

---

## Student Model: Mini-UNet

We use a custom U-Net architecture as the student model. It is a lightweight encoder-decoder network with skip connections, modular design, and supports optional features like dropout, batch normalization, and configurable depth.

### U-Net Flowchart (3-Level Depth)

```
[Input Image (3xHxW)]
        │
        ▼
+---------------------+
|   Encoder Block 1   |  →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→+
+---------------------+                                │
        │                                              ↓
        ▼                                    +---------------------+
+---------------------+                     |  Decoder Block 1    |
|   Encoder Block 2   |    ───── skip ───▶ | (UpSample + Conv)   |
+---------------------+                     +---------------------+
        │                                             ↓
        ▼                                   +---------------------+
+---------------------+                     |  Decoder Block 2    |
|   Bottleneck Block  |     ───── skip ───▶| (UpSample + Conv)    |
+---------------------+                     +---------------------+
        │                                             ↓
        ▼                                   +---------------------+
+---------------------+                     |   Output Conv 1x1   |
|  Sharpened Output   | ◀───────────────── |      RGB Output      |
+---------------------+                     +---------------------+
```

---

### Key Characteristics

- **Input**: RGB patches (`3×512×512`)
- **Base Filters**: 64 (scalable)
- **Depth**: 3 or 4 levels (configurable)
- **Output**: Reconstructed RGB image (sharpened)
- **Skip Connections**: Between encoder and decoder layers
- **Flexibility**:
  - Toggle dropout via flag
  - Easily adjust number of encoder-decoder levels
  - Optional residual blocks (can be extended)
  
---

## Implementation Highlights

- **Modular ConvBlock** (Conv + BatchNorm + ReLU)
- **Downsampling**: MaxPooling
- **Upsampling**: Transposed Convolutions
- **Final Layer**: 1×1 convolution to predict RGB image

---

## Model Variants

| Version | Description |
|---------|-------------|
| `student_model_v1.pth` | L1 + Knowledge Distillation only |
| `student_model_v2.pth` | L1 + KD + VGG perceptual loss |

Both of these models were trained using the `student_model_unet.py` architecture with different loss compositions.

---

## Integration

This model is used in:

- `training/train_student_kd.py`
- Evaluation and inference code inside `ISKD - RESTORMER.ipynb`
- SSIM scoring and visualization workflows

---

## Customization Guide

| Parameter | Change Location | Options |
|-----------|------------------|---------|
| `depth` | Constructor argument | 3 (default), 4 |
| `use_dropout` | Constructor argument | `True` / `False` |
| `base_filters` | Constructor argument | 32, 64, 128... |
| Layers | `student_model_unet.py` | Edit encoder/decoder blocks as needed |

---

## Reference

Inspired by the original U-Net design but heavily optimized for knowledge distillation and fast inference. All components are implemented in native **PyTorch** for maximum compatibility.

---

📁 **Path**: [`models/student_model_unet.py`](./student_model_unet.py)

🧠 Designed and tuned for this project: **Image Sharpening using Knowledge Distillation (Restormer + UNet)**


