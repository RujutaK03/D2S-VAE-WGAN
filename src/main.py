import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset import NIfTIDataset
from train import train_vae
from synthetic import generate_synthetic_samples_nii
from evaluate import evaluate_synthetic_data
from models import VAE
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train VAE, generate synthetic data, or evaluate synthetic data for BRATS 2021.")
    parser.add_argument('--synthetic', action='store_true', help="Generate synthetic samples instead of training.")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate synthetic data against real data.")
    args = parser.parse_args()
    
    config = load_config('../config.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset Setup
    modalities = ['flair', 't1', 't1ce', 't2', 'seg']
    
    if args.synthetic or args.evaluate:
        real_dataset = NIfTIDataset(
            precomputed_dir=config.get('general', {}).get('dataset_precomputed_dir', './precomputed'),
            modalities=modalities,
            normalize=True,
            target_shape=(128, 128, 128)
        )
    else:
        real_dataset = NIfTIDataset(
            target_dir=config.get('general', {}).get('data_dir', './data'),
            modalities=modalities,
            precomputed_dir=config.get('general', {}).get('dataset_precomputed_dir', './precomputed'),
            normalize=True,
            visualize_preprocess=config.get('visualize_preprocess', True),
            target_shape=(128, 128, 128),
            sharpen_alpha=1.0
        )

    # Compute class distribution
    class_weights = real_dataset.compute_class_distribution()
    config['dataset']['class_weights'] = class_weights.tolist()
    
    # Split dataset
    dataset_size = len(real_dataset)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = Subset(real_dataset, train_indices)
    val_dataset = Subset(real_dataset, val_indices)
    test_dataset = Subset(real_dataset, test_indices)
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    if args.synthetic:
        # Generate synthetic samples
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
        vae = VAE(
            in_channels=4,
            latent_dim=config['latent_dim'],
            num_seg_classes=4,
            dropout_p=0.5
        ).to(device)
        checkpoint_path = config.get('synthetic', {}).get('checkpoint_path', './vae_wgan_final.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            vae.load_state_dict(checkpoint['vae_state_dict'], strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        generate_synthetic_samples_nii(
            model=vae,
            test_loader=test_loader,
            num_samples=config['synthetic']['num_samples'],
            output_dir=config['synthetic']['output_dir'],
            device=device,
            target_ssim_range=config['synthetic']['target_ssim_range'],
            target_dice=config['synthetic']['target_dice_range'],
            reference_nii_path=config['synthetic']['reference_nii_path'],
            visualize=True
        )
    elif args.evaluate:
        # Evaluate synthetic data
        synth_dataset = NIfTIDataset(
            precomputed_dir=config.get('evaluate_synthetic', {}).get('precomputed'),
            modalities=modalities,
            normalize=True,
            target_shape=(128, 128, 128)
        )
        real_loader = DataLoader(test_dataset, batch_size=config['evaluate_synthetic']['batch_size'], shuffle=True, num_workers=1)
        synth_loader = DataLoader(synth_dataset, batch_size=config['evaluate_synthetic']['batch_size'], shuffle=True, num_workers=1)
        
        vae = VAE(
            in_channels=4,
            latent_dim=config['latent_dim'],
            num_seg_classes=4,
            dropout_p=0.5
        ).to(device)
        checkpoint_path = config.get('synthetic', {}).get('checkpoint_path', './vae_wgan_final.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            vae.load_state_dict(checkpoint['vae_state_dict'], strict=False))
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        evaluate_synthetic_data(
            real_loader=real_loader,
            synth_loader=synth_loader,
            model=vae,
            device=device,
            config=config,
            output_dir=config.get('evaluate_synthetic', {}).get('output_dir', './visualizations')
        )
    else:
        # Train the model
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        checkpoint_path = config.get('checkpoint_path', './pretrained_vae_epoch_10.pth')
        train_vae(train_loader, val_loader, config, device, checkpoint_path=checkpoint_path, start_epoch=11)

if __name__ == "__main__":
    main()
