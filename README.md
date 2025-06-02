# D2S-VAE-WGAN

This study investigates the challenge of limited high-quality annotated medical imaging data by proposing an advanced Variational Autoencoder with Wasserstein Generative Adversarial Network (VAE-WGAN) framework, termed Dual-Decoder Skip VAE-WGAN (D2S-VAE-WGAN), designed to generate realistic multi-modal MRI data. The proposed architecture incorporates skip connections to enhance detail preservation and employs a dual-decoder approach—one dedicated to reconstructing multi-modal images (FLAIR, T1, T1ce, T2) and another to generating segmentation masks. Additional loss functions, including perceptual loss, dice loss, and boundary loss, are integrated to ensure both fidelity and structural precision. This framework enhances dataset diversity and enables resource-efficient processing of 3D volumes, optimized for environments with 16GB VRAM and constrained computational sessions, such as Kaggle’s 12-hour limits. 

Evaluation using a benchmark dataset produced 250 synthetic samples, achieving a mean Structural Similarity Index (SSIM) of 0.7909, Peak Signal-to-Noise Ratio (PSNR) of 22.6535 dB, and an average Dice score of 0.7409. Sub-region Dice scores include 0.7002 for necrotic core (NCR), 0.8008 for enhancing tumor (ET), 0.7216 for edema (ED), 0.8411 for whole tumor (WT), and 0.8798 for tumor core (TC). This research presents a novel, reproducible pipeline that significantly contributes to generative modelling in medical imaging, providing a scalable solution for data-constrained settings.

## Overview
This repository contains code for preprocessing, loading, training a Variational Autoencoder (VAE) and VAE-WGAN for brain tumor segmentation using the BRATS 2023 dataset, generating synthetic MRI samples with corresponding segmentations, and evaluating their quality. The code extracts NIfTI files, organizes them by modality (FLAIR, T1, t1ce, T2, segmentation), preprocesses the data (resizing, sharpening, normalization), trains a VAE-based model, generates synthetic samples, and evaluates them using metrics like SSIM, PSNR, Dice, and Hausdorff Distance.

## Repository Structure
- `src/` directory:
  - `preprocess.py`: Extracts and organizes BRATS 2023 dataset NIfTI files by modality.
  - `dataset.py`: Defines the `NIfTIDataset` class for loading and preprocessing data.
  - `losses.py`: Implements loss functions and visualization utilities.
  - `models.py`: Defines the VAE model architecture.
  - `train.py`: Contains the training loop for VAE and VAE-WGAN pretraining.
  - `synthetic.py`: Implements synthetic data generation with quality metrics.
  - `evaluate.py`: Evaluates synthetic data against real data with comprehensive metrics.
  - `main.py`: Sets up the dataset, initiates training, generates synthetic samples, or evaluates synthetic data.
- `config.yaml`: Configuration file for hyperparameters (not included in repo; see Setup).
- `requirements.txt`: Python dependencies.
- `README.md`: This file, containing setup and usage instructions.
- `data/` (not included): Directory for BRATS 2023 dataset.
- `precomputed/` (not included): Directory for precomputed data.
- `visualizations/` (not included): Directory for output visualizations.
- `brats_synthetic_samples/` (not included): Directory for synthetic samples.

## Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (recommended for faster processing)
- BRATS 2023 dataset (download from [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1))
- Sufficient storage for dataset (~100 GB), precomputed files, and synthetic samples

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/[your-username]/brats2025-segmentation.git
   cd brats2025-segmentation
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download the BRATS 2023 dataset**:
   - Download `BraTS2023_Training_Data.tar` from [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1).
   - Place the `.tar` file in the `data/` directory or update the `file_path` in `src/preprocess.py`.
5. **Set up directories**:
   - Create `data/`, `precomputed/`, `visualizations/`, and `brats_synthetic_samples/` directories:
     ```bash
     mkdir data precomputed visualizations brats_synthetic_samples
     ```
   - Ensure write permissions for these directories.
6. **Create `config.yaml`**:
   - Create a `config.yaml` file in the project root with the following structure (adjust values as needed):
     ```yaml
     general:
       data_dir: "./data"
       dataset_precomputed_dir: "./precomputed"
     latent_dim: 128
     num_seg_classes: 4
     lr: 0.0001
     beta1: 0.5
     lambda_recon: 2.0
     lambda_dice: 15.0
     lambda_focal: 2.0
     lambda_kl: 5.0
     lambda_perceptual: 0.1
     lambda_grad: 0.2
     lambda_boundary: 0.5
     num_epochs: 15
     batch_size: 2
     class_weights: [3.0, 1.0, 1.5]
     visualize_preprocess: true
     synthetic:
       num_samples: 250
       batch_size: 1
       target_ssim_range: [0.7, 0.8]
       target_dice_range: [0.65, 0.8]
       reference_nii_path: "./data/reference_flair.nii"
       output_dir: "./brats_synthetic_samples"
       checkpoint_path: "./vae_wgan_final.pth"
     evaluate_synthetic:
       precomputed: "./brats_synthetic_samples/precomputed"
       batch_size: 4
       output_dir: "./visualizations"
     ```
   - Update paths and hyperparameters as needed.

## Usage
1. **Preprocess the dataset**:
   - Run the preprocessing script to extract and organize NIfTI files:
     ```bash
     cd src
     python preprocess.py
     ```
   - This extracts the `.tar` file and organizes files into modality folders (`flair`, `t1`, `t1ce`, `t2`, `seg`).
2. **Train the model**:
   - Run the main script to prepare the dataset and train the VAE/VAE-WGAN:
     ```bash
     python main.py
     ```
   - This loads the dataset, creates train/validation/test splits, and starts training.
   - Visualizations are saved in `visualizations/`.
   - Checkpoints are saved as `pretrained_vae_epoch_X.pth` in the project root.
3. **Generate synthetic samples**:
   - Run the main script with the `--synthetic` flag to generate synthetic samples:
     ```bash
     python main.py --synthetic
     ```
   - This uses the trained VAE to generate synthetic MRI and segmentation NIfTI files, saved in `brats_synthetic_samples/`.
   - Requires a pretrained VAE checkpoint and a reference NIfTI file.
4. **Evaluate synthetic data**:
   - Run the main script with the `--evaluate` flag to analyze synthetic data:
     ```bash
     python main.py --evaluate
     ```
   - This compares synthetic data against real data, computing Dice scores, reconstruction metrics, and Hausdorff distances.
   - Results are printed to the console and saved in `visualizations/evaluation_metrics.txt`.
   - Visualizations are saved in `visualizations/`.
5. **Visualize preprocessing**:
   - Set `visualize_preprocess: true` in `config.yaml` to visualize preprocessing for the first sample.
6. **Adjust hyperparameters**:
   - Modify `config.yaml` for different training phases, synthetic generation, or evaluation.

## Notes
- **Paths**: Update paths in `preprocess.py`, `dataset.py`, `main.py`, and `config.yaml` for local use (e.g., `./data`, `./precomputed`).
- **Precomputation**: Enable precomputation by setting `precomputed_dir` in `config.yaml`. Ensure sufficient disk space.
- **Synthetic Data**: Requires a pretrained VAE checkpoint and a reference NIfTI file. Precompute synthetic data for evaluation.
- **Evaluation**: Ensure synthetic data is precomputed in `brats_synthetic_samples/precomputed/`.
- **GPU**: A CUDA-enabled GPU is recommended. CPU processing is possible but slower.
- **Checkpoints**: Pretrained checkpoints must be placed in the project root or updated in `config.yaml`.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements.
