# pretrained_models/ — Pretrained Teacher Model (Restormer)

This folder contains the reference and structure for the **Restormer Teacher Model** used in this project for knowledge distillation.

---

## Teacher Model Overview

- **Model**: Restormer (Restoration Transformer)
- **Purpose**: Acts as a strong teacher for guiding the lightweight student model (Mini-UNet)
- **Task**: Motion Deblurring on DIV2K patches (512×512)
- **Pretrained Weights**: Used official pretrained weights for motion deblurring

---

## Files in This Folder

| File Name                 | Description                                              |
|---------------------------|----------------------------------------------------------|
| `motion_deblurring.pth`   | Pretrained Restormer weights (Motion Deblurring)       |
| `restormer_arch.py`       | Model architecture definition (from official repo)       |
| `README.md`               | You are here — description of pretrained teacher model   |

> **Note**: These files are large, I have provided via [Google Drive](https://drive.google.com/drive/folders/12h8aeCktRI54Byuaau4rwP9NKjgxfph-?usp=drive_link) or you can take reference from the official GitHub.

---

## Official Model Reference

- **Source Repo**: [Restormer - Official GitHub](https://github.com/swz30/Restormer)
- **Pretrained Model (Motion Deblurring)**:
  ```
  https://github.com/swz30/Restormer#pre-trained-models
  ```

---

## How It's Used

1. Inference is performed on the blurry patches using the pretrained `motion_deblurring.pth`.
2. The outputs are stored in:
   ```
   /outputs/teacher_output/train/train/
   /outputs/teacher_output/train/test/
   /outputs/teacher_output/benchmark/
   ```
3. These soft targets are used to train the student model using L1 + KD + VGG loss.

---

## Note

- No fine-tuning was performed on the teacher model.
- It serves purely as a **frozen teacher** for guiding the student model through knowledge distillation.
- Please cite or reference the original Restormer paper if using this work for academic research.

---

## Download

You can download the pretrained weights if not present in the repo:

📁 [Download motion_deblurring.pth from Official GitHub Repo](https://github.com/swz30/Restormer/tree/main/Motion_Deblurring)

---
