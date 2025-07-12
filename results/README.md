# results/ — Benchmark Evaluation Outputs

This folder contains evaluation results and visual samples generated after testing the **student model** on benchmark images from the DIV2K dataset.

---

## Files Inside

| File Name                   | Description                                                      |
|-----------------------------|------------------------------------------------------------------|
| `student_ssim_scores.csv`   | Full SSIM score table comparing blurry input vs. student output |
| `highest.png`               | Top 20 student output images with highest SSIM scores         |
| `midrange.png`              | 20 student output images with mid-range SSIM (0.80–0.89)       |
| `README.md`                 | Folder-level documentation (you are here)                        |

---

## Description

- The CSV file includes SSIM scores after inference using the final student model.
- The two images are snapshots from the notebook showcasing best and mid-range performing samples.
- Scores were computed using `skimage.metrics.structural_similarity` on **Y-channel** (luminance).

---

## CSV Format

| Column Name        | Description                                   |
|--------------------|-----------------------------------------------|
| `Image`            | Filename of the evaluated image               |
| `SSIM (Blurry)`    | SSIM between blurry input and ground truth    |
| `SSIM (Student)`   | SSIM between student output and ground truth  |

CSV File: [`student_ssim_scores.csv`](./student_ssim_scores.csv)

---

## Visual Snapshots

These two PNGs show samples from the SSIM evaluation:

- [`highest.png`](./highest.png): Top 20 images with highest SSIM scores (≥ 0.92)
- [`midrange.png`](./midrange.png): 20 images with mid-range SSIM (~0.80–0.89)

> Extracted using `pandas` filtering + Matplotlib from the final evaluation cell in the notebook.

---

## Summary Table (From Notebook)

| Metric                | Score   |
|------------------------|---------|
| Average SSIM (Blurry)  | ~0.61   |
| Average SSIM (Student) | **~0.90** |

---

## How This Was Generated

All files in this folder were created by running the evaluation section of:

[`ISKD - RESTORMER.ipynb`](../ISKD%20-%20RESTORMER.ipynb)

> These outputs were generated using the final model `student_model_v2.pth`.

---

## Note

- These scores and samples pertain only to the **benchmark** set (`data/blurry/benchmark/`).
- You can replicate the process for external or test datasets by modifying paths in the notebook.

---

