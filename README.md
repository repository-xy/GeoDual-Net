# GeoDual-Net: A Lightweight Heterogeneous Dual-Decoder for Remote Sensing Image Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper: **GeoDual-Net: A Lightweight Heterogeneous Dual-Decoder for Remote Sensing Image Segmentation**.

## 📝 Overview & Abstract

High-resolution semantic segmentation of remote sensing imagery is frequently hindered by the intricate spatial heterogeneity of urban environments and the performance bottlenecks of existing architectures. Current methods face inherent trade-offs: Convolutional Neural Networks (CNNs) struggle with long-range context, Vision Transformers (ViTs) incur prohibitive computational costs, and conventional feature propagation often amplifies shallow spectral confusion.

**GeoDual-Net** is introduced as a computationally efficient framework designed to resolve these trade-offs. It achieves a superior balance between lightweight design and competitive accuracy, making it highly suitable for resource-constrained Earth observation tasks, such as edge computing on airborne or spaceborne platforms.

## 🏗️ Architecture Design

Instead of relying on heavy backbones, GeoDual-Net utilizes a minimalist **ResNet-16** as the feature encoder. The core innovations lie in its decoding and feature-fusion stages:

1. **Heterogeneous Dual-Decoder**: At each decoding stage, the network splits into two parallel branches.
   - **CNN Branch**: Focuses on spatial detail recovery and precise boundary localization.
   - **Swin Transformer Branch**: Employs window-based self-attention to capture global semantic context and long-range dependencies.
   These two branches are dynamically merged at each scale to combine local textures with global semantics.
 
2. **Self-Attention Skip Connection (SASC)**: To prevent the background noise of shallow encoder layers from polluting the decoder, the SASC module is introduced. It adaptively calculates attention weights for skip-connection features, effectively filtering out noise while preserving critical multi-scale information.

## 📊 Performance & Efficiency

Based on extensive experiments on the ISPRS Potsdam and Vaihingen datasets, GeoDual-Net achieves a highly competitive balance:

* **Computational Complexity**: The model requires only **10.52 Million parameters** and **17.64 GFLOPs**.
* **ISPRS Potsdam**: Achieves an mIoU of **88.00%**.
* **ISPRS Vaihingen**: Achieves an mIoU of **86.22%**.

Compared to standard heavy models (like DeepLabV3+ and UperNet), GeoDual-Net dramatically reduces the computational cost while maintaining highly competitive segmentation accuracy. In qualitative visual comparisons, GeoDual-Net shows exceptional performance in recovering precise geometric boundaries and mitigating spectral confusion, distinguishing difficult categories such as bare soil and trees.

## 📁 Repository Structure

* `model2/`: Contains the core implementation of `GeoDual_Net.py` and other baseline models.
* `Datasets/`: Directory for ISPRS datasets and data augmentation scripts (`dataset_isprs.py`).
* `test/`: Contains evaluation scripts (`testGeoDualNetP.py`, `testGeoDualNetV.py`) and parameter/FLOPs calculation tools.
* `train.py`: The main entry point for model training.

## ⚙️ Environment Setup

We recommend using **Python 3.10+** and **PyTorch 2.5.1**. To install the required dependencies:

```bash
pip install -r requirements.txt
```
## 🗄️ Dataset Preparation

1. Download the original ISPRS Potsdam and Vaihingen datasets from the official ISPRS website.
2. Place the datasets in the `Datasets/Potsdam` and `Datasets/Vaihingen` directories.
3. Run the data preprocessing script to generate patch-based `.npz` files (all images will be cropped into non-overlapping 256×256 patches):

```bash
python create_npz3_RGB.py
```
## 🚀 Training & Evaluation

**Train from scratch**
To train the GeoDual-Net model, use train.py as the main entry point:

```bash
python train.py --model_name Res16_DualDecoder --batch_size 16 --max_epochs 150
```
Verify model complexity (Params & GFLOPs)
To verify the number of parameters and computational cost (expected: 10.52M / 17.64G):

```bash
python test/get_flops.py
```
Evaluate on test sets
To run evaluation on the respective ISPRS test sets:

```bash
# For ISPRS Potsdam Dataset
python test/testGeoDualNetP.py
```
```bash
# For ISPRS Vaihingen Dataset
python test/testGeoDualNetV.py
```
## 🔗 Pre-trained Weights

For reproducibility, the pre-trained best weights for GeoDual-Net can be accessed directly from this repository:

- **ISPRS Potsdam Dataset**: [RGBepoch_150.pth (Potsdam)](https://github.com/huangxiaoyu-hxy/GeoDual-Net/blob/main/GeoDual-Net/weights/Res16_DualDecoder_Pots_256_256/RGBepoch_150.pth)
- **ISPRS Vaihingen Dataset**: [RGBepoch_150.pth (Vaihingen)](https://github.com/huangxiaoyu-hxy/GeoDual-Net/blob/main/GeoDual-Net/weights/Res16_DualDecoder_Vai_256_256/RGBepoch_150.pth)

Ensure these `.pth` files are placed in the correct subdirectories under the `weights/` folder before running the test scripts.

## 🎓 Citation

If you find this repository or our work useful, please consider citing our paper:

```bibtex
@article{huang2026geodualnet,
    title={GeoDual-Net: A Lightweight Heterogeneous Dual-Decoder for Remote Sensing Image Segmentation},
    author={Huang, Xiaoyu and Wang, Yonggui and Zhang, Junming and Ai, Qiang},
    journal={International Journal of Remote Sensing},
    year={2026},
    note={Submitted}
```
}
