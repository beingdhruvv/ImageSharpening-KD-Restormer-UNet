# Image Sharpening via Knowledge Distillation using Restormer and Mini-UNet

This repository showcases a complete pipeline for high-quality **Image Sharpening** using **Knowledge Distillation (KD)**. A pretrained **Restormer model** acts as the high-capacity **teacher**, while a lightweight **Mini-UNet model** is trained as the **student** to match the teacher’s performance.

The pipeline is designed for **efficient deployment** in real-world settings, where high-resolution image restoration is needed but **hardware resources are limited**.

---

## Project Overview

Traditional motion deblurring models either provide **high accuracy but are computationally heavy** (like Restormer), or are **lightweight but lose quality** (like basic U-Nets). Our solution combines the best of both:

- A **Restormer teacher model** generates high-quality sharp outputs from blurry images.
- A compact **Mini-UNet student** learns not only from ground truth (sharp images), but also from the teacher’s output — using:
  - **L1 Loss** for pixel accuracy  
  - **Distillation Loss** to mimic teacher behavior  
  - **VGG Perceptual Loss** to preserve visual quality

The student is trained to match the teacher’s performance, while being up to **10× smaller** and much faster — making it ideal for **mobile and edge devices**.

---

## Key Highlights

- Uses **Restormer** as a powerful pretrained teacher model
- Student model is a **custom Mini-UNet**, 3–4 layers deep, <30 MB
- Incorporates **multi-loss training** (L1 + KD + Perceptual)
- Based on **DIV2K dataset** with blurry/sharp pairs
- Full training, inference, and SSIM evaluation done in **Google Colab**
- **Resumable training** with checkpoint support
- Achieves strong SSIM (~0.90+) with high visual fidelity

---

### Dataset Used — DIV2K 

We use the **DIV2K dataset** (high-resolution image dataset) as the base for training and evaluation. The dataset is organized into paired blurry–sharp images for both training and benchmarking, with a strict triplet alignment to support knowledge distillation:

---

### Folder Structure
```
data/
├── whole_dataset/                  # Original DIV2K_train_HR images
├── blurry/
│   ├── train/
│   │   ├── train/                  # 80% of training images (blurry patches)
│   │   └── test/                   # 20% of training images (blurry patches)
│   └── benchmark/                  # 10% benchmark blurry images
├── sharp/
│   ├── train/
│   │   ├── train/                  # 80% of training images (sharp patches)
│   │   └── test/                   # 20% of training images (sharp patches)
│   └── benchmark/                  # 10% benchmark sharp images
```

### Key Details
- **Patch Size**: 512×512 (non-overlapping, center-cropped if needed)
- **Total Training Images (Patches)**: ~20,000 triplets
- **Benchmark Pairs**: ~100 full-size images for final SSIM evaluation
- **Sharpness Degradation**: Blur is synthetically added using downscale-upscale + motion blur for realism
- **Triplet Matching**: All blurry/sharp/teacher images are aligned by filename

---

📎 **More Details:** Refer to the [`data/README.md`](./data/README.md) for exact dataset preparation steps, patching logic, and source download references.

---

## Methodology – Knowledge Distillation + U-Net + VGG Loss

This project focuses on a **lightweight student network (Mini U-Net)** trained using a **knowledge distillation (KD)** framework. We leverage a powerful **Restormer** model as the teacher to guide and supervise the training of the student. The student learns not just from the ground truth (sharp images), but also from the intermediate guidance of the teacher's outputs. This results in improved sharpness, structural accuracy, and generalization—all while reducing model size and inference cost.

---

### Components of the Method

| Component            | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Teacher Model**    | Pretrained **Restormer** (Motion Deblurring model)                         |
| **Student Model**    | Mini **U-Net** (2–3 encoding–decoding blocks with skip connections)         |
| **Distillation Type**| Output-based regression (pixel-wise teacher output)                         |
| **Input Patch Size** | 512×512 patches                                                             |
| **Training Loss**    | `Total Loss = L1 + λ_kd * Distillation Loss + λ_vgg * Perceptual Loss`      |

---

### Loss Functions Used

| Loss Type             | Description                                                                                      | Weight |
|-----------------------|--------------------------------------------------------------------------------------------------|--------|
| **L1 Loss**           | Measures pixel-wise difference between student output and ground truth sharp image              | 1.0    |
| **Distillation Loss** | Measures difference between student output and teacher (Restormer) output                        | 1.0    |
| **VGG Perceptual Loss** | Computes high-level feature distance between student output and ground truth (using VGG16)   | 0.1    |

> ℹAll losses are computed on full-resolution patches and combined during training.

---

### Why Knowledge Distillation?

- **Performance**: Mimicking a strong teacher helps the student learn refined deblurring patterns  
- **Efficiency**: Enables real-time image sharpening on resource-constrained devices (e.g., mobile)  
- **Generalization**: The student learns smoother and more perceptually aligned reconstructions  

---

📎 See: [`training/train_student_kd.py`](./training/train_student_kd.py) for full implementation of the KD + VGG training pipeline.

---


