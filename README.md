# Satellite Image Change Detection at Scale

Siamese change detection for LEVIR-CD using PyTorch, ResNet-18, CBAM, RAPIDS, and Dask.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab)](https://colab.research.google.com/github/YOUR_USERNAME/satellite-change-detection/blob/main/notebooks/08_full_pipeline.ipynb)

## Overview

Change detection compares two images of the same place taken at different times and identifies where meaningful changes happened. In satellite imagery, that usually means spotting new buildings, demolished structures, flood damage, or land-use changes.

This matters because manual inspection is slow, expensive, and hard to scale. Reliable automatic change detection helps with urban planning, disaster response, and long-term land monitoring.

This project builds a complete LEVIR-CD pipeline that starts from a Siamese U-Net baseline, then upgrades it with a BCE-Dice segmentation objective, a shared ResNet-18 backbone, CBAM attention, RAPIDS-accelerated data and metrics steps, and Dask-based full-resolution sliding-window inference.

## Architecture

```text
Time-1 Image ──┐
               ├─> Shared ResNet-18 Encoder ─> U-Net Decoder + CBAM ─> Dense Embedding A ──┐
Time-2 Image ──┘                                                                             ├─> L2 Distance Map ─> ChangeHead ─> Probability Map ─> Binary Mask
Time-1/Time-2 pair share the same encoder and decoder weights                               ┘
```

The model uses a Siamese design so both timestamps are encoded in the same feature space. A U-Net-style decoder restores dense spatial detail, while CBAM refines skip features before fusion so boundaries and small buildings stay sharper. The lightweight change head converts the distance map into a calibrated probability map, and BCE-Dice aligns the training objective with F1-oriented segmentation quality.

## Results

| Model | Loss | F1 | Precision | Recall | IoU |
|---|---|---:|---:|---:|---:|
| Baseline (custom encoder) | Contrastive | 0.08 | [FILL_IN] | [FILL_IN] | [FILL_IN] |
| Step 2: BCE-Dice + head | BCE-Dice | [FILL_IN] | [FILL_IN] | [FILL_IN] | [FILL_IN] |
| Step 3: ResNet-18 | BCE-Dice | [FILL_IN] | [FILL_IN] | [FILL_IN] | [FILL_IN] |
| Step 4: ResNet-18 + CBAM | BCE-Dice | [FILL_IN] | [FILL_IN] | [FILL_IN] | [FILL_IN] |
| Step 6: Full-resolution sliding window | BCE-Dice | [FILL_IN] | [FILL_IN] | [FILL_IN] | [FILL_IN] |

## Quickstart — Open in Colab

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=googlecolab&style=for-the-badge)](https://colab.research.google.com/github/YOUR_USERNAME/satellite-change-detection/blob/main/notebooks/08_full_pipeline.ipynb)

1. Open the notebook from the badge above.
2. Connect to a GPU runtime in Colab.
3. Click `Run all` to execute the full pipeline end to end.

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/satellite-change-detection
cd satellite-change-detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# For RAPIDS (requires NVIDIA GPU):
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11 cuml-cu11 cucim
```

## Dataset Setup

1. Create a Kaggle account and open `Account` settings.
2. Click `Create New API Token` to download `kaggle.json`.
3. In Colab, upload the file to `~/.kaggle/kaggle.json` or add the key as a secret.
4. Download the dataset with:

```bash
kaggle datasets download -d mdrifaturrahman33/levir-cd
unzip -o levir-cd.zip -d /content/levir_data
```

Expected folder structure:

```text
levir_data/
└── LEVIR CD/
    ├── train/
    │   ├── A/
    │   ├── B/
    │   └── label/
    ├── val/
    │   ├── A/
    │   ├── B/
    │   └── label/
    └── test/
        ├── A/
        ├── B/
        └── label/
```

## How To Run

```bash
python -m src.train --config configs/full.yaml
python -m src.evaluate --config configs/full.yaml --checkpoint results/best.pth
python -m src.inference --config configs/full.yaml --checkpoint results/best.pth --img_a A.png --img_b B.png
```

## Project Structure

```text
satellite-change-detection/
├── configs/
│   ├── baseline.yaml
│   └── full.yaml
├── notebooks/
│   ├── 01_setup_and_data.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_bce_dice_upgrade.ipynb
│   ├── 04_resnet_backbone.ipynb
│   ├── 05_cbam_attention.ipynb
│   ├── 06_rapids_pipeline.ipynb
│   ├── 07_dask_inference.ipynb
│   └── 08_full_pipeline.ipynb
├── results/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── factory.py
│   ├── inference.py
│   ├── losses.py
│   ├── train.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       ├── cbam.py
│       ├── siamese_resnet.py
│       └── siamese_unet.py
├── requirements.txt
├── requirements_rapids.txt
├── setup.py
├── README.md
└── LICENSE
```

## Roadmap

- [x] Siamese U-Net baseline
- [x] BCE-Dice loss
- [x] ResNet-18 backbone
- [x] CBAM attention
- [x] RAPIDS pipeline
- [x] Dask sliding window inference
- [ ] ChangeFormer transformer comparison
- [ ] Real-time inference API

## References

- [LEVIR-CD dataset paper](https://openaccess.thecvf.com/content/CVPR2020/html/Chen_Spatial-Temporal_Attention-Based_Method_and_a_New_Dataset_for_Remote_Sensing_Image_CVPR_2020_paper.html)
- [Fully Convolutional Siamese Networks for Change Detection](https://ieeexplore.ieee.org/document/8451652)
- [U-Net paper](https://arxiv.org/abs/1505.04597)
- [CBAM paper](https://arxiv.org/abs/1807.06521)
- [RAPIDS documentation](https://docs.rapids.ai/)

## Team And Course Info

- Course: Accelerated Data Science (`21CSE313P`)
- Team: Sarvesh Sathyanarayanan, Sanjay Siva, Sanjay Kumar

## MIT License

This project is released under the terms of the [MIT License](LICENSE).
