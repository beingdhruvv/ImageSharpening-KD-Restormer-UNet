# `losses/` — VGG-Based Perceptual Loss Module

This folder contains custom loss functions used during student model training in the knowledge distillation pipeline.

---

## Files Included

| Filename       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `vgg_loss.py`  | Implements perceptual loss using VGG16 features (used with λ_vgg = 0.1)     |
| `README.md`    | This documentation file                                                     |

---

## vgg_loss.py — Perceptual Loss via VGG16

The VGG loss enhances perceptual sharpness by comparing deep feature representations from a pre-trained VGG16 network.

### Loss Formula:

```python
Loss_total = L1(Student, GT) + λ * Perceptual(Student, GT)
```

Where:
- **L1(Student, GT)**: pixel-wise L1 loss between student output and ground truth
- **Perceptual(Student, GT)**: VGG feature distance (usually `relu2_2` features)
- **λ_vgg**: weight (set to `0.1` in this project)

---

## Why Perceptual Loss?

| Benefit                  | Explanation                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Visual Sharpness      | Captures high-level textures missed by pixel-wise losses                    |
| Perceptual Alignment  | Encourages the output to resemble human perception more closely              |
| Complementary to KD   | Enhances detail reconstruction beyond mimicking teacher features             |

---

## Integration

This loss is integrated into the student training script:

- File: [`training/train_student_kd.py`](../training/train_student_kd.py)
- Argument: `lambda_vgg = 0.1`
- Imports the class: `from losses.vgg_loss import VGGPerceptualLoss`

---

## Notes

- VGG is loaded from `torchvision.models.vgg16(pretrained=True)`
- It only uses early convolutional layers (e.g., `relu2_2`) for fast inference
- VGG parameters are frozen (`requires_grad = False`)
- Input images are interpolated to 224×224 before feeding into VGG

---

## Usage Context

This loss is used only during **student training**, not during inference or testing.
It significantly improved the SSIM score in our project and helped the student model reach **~0.90 SSIM**.

---

