# Checkpoints – Student Model (U-Net)

This folder contains **trained checkpoints** of the student model (Mini-UNet), saved during and after the knowledge distillation training process.

Due to GitHub file size limitations, the actual `.pth` model files are hosted on **Google Drive** and linked below. This README explains what each file is, how it’s used, and how to resume training or perform inference.

---

## Contents of This Folder

| Filename                  | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `student_checkpoint.pth`  | Intermediate checkpoint (auto-saved during training) — contains model weights **and** optimizer state. <br> Used for **resuming training** |
| `student_model_v1.pth`    | Early version of the student model trained with **L1 + KD loss only** (no VGG loss) <br> Trained for 9 epochs on the full dataset |
| `student_model_v2.pth`    | **Final trained student model** using **L1 + KD + VGG** loss <br> Best performance, ready for inference and evaluation |

---

## Download from Google Drive

Since these files are large, they are hosted on Google Drive. You can download them from the link below:

🔗 [**Access Checkpoints Folder on Google Drive**](https://drive.google.com/drive/folders/1zEMz_28E1gpPhai-fWeuQNrK_zG5T0V4?usp=drive_link)

---

## Resuming Training from Checkpoint

If you want to resume training after a session ends (e.g. Colab GPU timeout), simply ensure:

- `student_checkpoint.pth` is present
- Your training script loads this file (already handled in `train_student_kd.py`)

> The checkpoint includes:
> - Epoch number  
> - Model weights  
> - Optimizer state

**Training will resume from the last saved epoch.**

---

## Using Final Models for Inference

Use `student_model_v2.pth` for **final evaluation** and **inference**.

Example snippet:
```python
model = UNet(base_filters=64, use_dropout=False, depth=4)
model.load_state_dict(torch.load("student_model_v2.pth", map_location="cpu"))
model.eval()
```

---

## When to Use Each File

| File                   | Use Case                                  |
|------------------------|-------------------------------------------|
| `student_checkpoint.pth` | Resume training mid-way                   |
| `student_model_v1.pth`   | Use for testing basic model (**L1 + KD**) |
| `student_model_v2.pth`   | Use for final results (**L1 + KD + VGG**) |

---

## Folder Placement

These files should be placed in:
```
ImageSharpening-KD-Restormer-UNet/
└── checkpoints/
    ├── student_checkpoint.pth
    ├── student_model_v1.pth
    └── student_model_v2.pth
```

---

## Model Summary

- **Architecture**: Mini-UNet (depth = 4)
- **Losses**:
  - `student_model_v1.pth`: L1 + KD
  - `student_model_v2.pth`: L1 + KD + VGG (λ_vgg = 0.1)
- **Dataset**: DIV2K (512×512 patches)
- **SSIM Achieved**: ~0.90 on benchmark images (`student_model_v2.pth`)

---

## Note

These checkpoints are specific to this project. Using them with another dataset or architecture may lead to unexpected results.

