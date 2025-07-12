# DATA Folder – Structure & Overview

This folder contains the full preprocessed dataset used for training, validating, and evaluating our image sharpening models. The data is derived from the **DIV2K** high-resolution image dataset and organized into separate blurry/sharp pairs with an additional teacher output for distillation.

---

## Download the Full Dataset Folder

Due to its large size, the dataset is hosted on Google Drive:

**[Access DATA Folder on Google Drive](https://drive.google.com/drive/folders/1KsTGURpL-TfzumqctRw_7E3F_U5vr9aj?usp=drive_link)**

---

## Folder Structure
```
data/
├── whole_dataset/                 # Original DIV2K_train_HR images (sharp)
├── blurry/
│   ├── train/
│   │   ├── train/                # 80% of 90% training blurry images
│   │   └── test/                 # 20% of 90% training blurry images
│   └── benchmark/                # 10% reserved benchmark blurry images
├── sharp/
│   ├── train/
│   │   ├── train/                # Matching sharp images (train set)
│   │   └── test/                 # Matching sharp images (test set)
│   └── benchmark/                # Matching sharp images (benchmark)
```

---

## Dataset Summary

| Component         | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| `whole_dataset/`  | Full-resolution original DIV2K images (used for patch generation)        |
| `blurry/`         | Input images for the model (artificially blurred via resize + upsample)  |
| `sharp/`          | Ground truth reference images                                             |
| `benchmark/`      | Final evaluation set (student SSIM is reported on this set)              |

---

## Patch Details

- **Patch Size**: 512 × 512 pixels
- **Stride**: No overlap (non-overlapping grid)
- **Generated Pairs**: Each blurry image has a corresponding sharp and teacher image
- **Total Images**: ~22,000+ patches across train/test/benchmark

---

## How the Data Was Prepared

1. **Original Source**: [DIV2K_train_HR dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. **Patch Creation**:
   - Used a sliding window of 512×512 (no overlap)
   - Extracted from the original DIV2K full-resolution `.jpg` images
3. **Blurring Process**:
   - Downsampled to low-res and then upscaled to introduce realistic blur
4. **Splitting Strategy**:
   - 90% → Training pool
     - 80% for `train/train`
     - 20% for `train/test`
   - 10% → Final benchmark

---

## Data Alignment

Each triplet used for student model training contains:

- `blurry/train/train/image_XXXX.png`
- `sharp/train/train/image_XXXX.png`
- `outputs/teacher_output/train/train/image_XXXX.png`

> These are matched by filename and used as (input, ground truth, teacher prediction) triplets.

---

## Notes

- All patches are in `.jpg` format and RGB
- No normalization is applied; models operate in the range [0, 1] via `ToTensor`
- Benchmark evaluation also uses patches, not full-size DIV2K images

---

## Benchmark Evaluation Set

- Used to compute final SSIM of student model
- Located in:
  - `data/blurry/benchmark/`
  - `data/sharp/benchmark/`
- SSIM calculated on Y-channel (luminance) using scikit-image

---

## Integration with Code

The training and evaluation scripts expect the folder to be structured exactly as described. No manual edits needed if the dataset is downloaded from the shared Drive.

---

## Important

- This dataset is required for:
  - Teacher inference
  - Student training
  - Benchmark evaluation
- Ensure the directory paths match those referenced in the notebook and scripts.

---

Need help loading the data?

Check:  
[`training/train_student_kd.py`](../training/train_student_kd.py)  
[`ISKD_RESTORMER.ipynb`](ISKD_RESTORMER.ipynb)



