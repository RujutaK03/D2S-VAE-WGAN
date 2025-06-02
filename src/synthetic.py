import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import gc
from models import VAE

def compute_subregion_dice(pred, target):
    """
    Compute Dice scores for NCR (1), ET (3), ED (2), WT (1+2+3), TC (1+3).
    Assumes pred and target are [B, C, D, H, W] tensors with C=4 (0, 1, 2, 3).
    """
    pred_labels = torch.argmax(pred, dim=1)  # [B, D, H, W]
    target_labels = torch.argmax(target, dim=1)  # [B, D, H, W]
    
    dice_scores = {}
    
    # NCR: Label 1
    pred_ncr = (pred_labels == 1).float()
    target_ncr = (target_labels == 1).float()
    dice_scores['NCR'] = (2. * (pred_ncr * target_ncr).sum() + 1e-5) / (pred_ncr.sum() + target_ncr.sum() + 1e-5)
    
    # ET: Label 3
    pred_et = (pred_labels == 3).float()
    target_et = (target_labels == 3).float()
    dice_scores['ET'] = (2. * (pred_et * target_et).sum() + 1e-5) / (pred_et.sum() + target_et.sum() + 1e-5)
    
    # ED: Label 2
    pred_ed = (pred_labels == 2).float()
    target_ed = (target_labels == 2).float()
    dice_scores['ED'] = (2. * (pred_ed * target_ed).sum() + 1e-5) / (pred_ed.sum() + target_ed.sum() + 1e-5)
    
    # WT: Labels 1, 2, 3
    pred_wt = (pred_labels > 0).float()  # Any tumor region
    target_wt = (target_labels > 0).float()
    dice_scores['WT'] = (2. * (pred_wt * target_wt).sum() + 1e-5) / (pred_wt.sum() + target_wt.sum() + 1e-5)
    
    # TC: Labels 1, 3
    pred_tc = ((pred_labels == 1) | (pred_labels == 3)).float()
    target_tc = ((target_labels == 1) | (target_labels == 3)).float()
    dice_scores['TC'] = (2. * (pred_tc * target_tc).sum() + 1e-5) / (pred_tc.sum() + target_tc.sum() + 1e-5)
    
    return {k: v.item() for k, v in dice_scores.items()}

def compute_multi_class_dice(pred, target, num_classes=4):
    dice_scores = []
    pred = torch.argmax(pred, dim=1)
    target = torch.argmax(target, dim=1)
    for c in range(1, num_classes):  # Skip background
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_scores.append(dice.item())
    return np.mean(dice_scores)

def generate_synthetic_samples_nii(
    model, test_loader, num_samples, output_dir, device,
    target_ssim_range, target_dice_range, reference_nii_path, visualize=False
):
    model.eval()
    model.to(device)
    
    # Create output directories
    modality_names = ['flair', 't1', 't1ce', 't2', 'seg']
    modality_dirs = {}
    for name in modality_names:
        dir_path = os.path.join(output_dir, name)
        os.makedirs(dir_path, exist_ok=True)
        modality_dirs[name] = dir_path
    
    # Load reference NIfTI
    if not os.path.exists(reference_nii_path):
        raise ValueError(f"Reference NIfTI file not found at {reference_nii_path}")
    reference_nii = nib.load(reference_nii_path)
    
    num_generated = 0
    with torch.no_grad():
        sample_idx = 0
        while num_generated < num_samples:
            # Get a test sample
            batch = test_loader.dataset[sample_idx % len(test_loader.dataset)]
            orig_modalities = torch.from_numpy(batch['modalities']).float().to(device).unsqueeze(0)
            orig_seg = torch.from_numpy(batch['seg']).float().to(device).unsqueeze(0)
            sample_idx += 1
            
            # Generate synthetic sample
            modalities_recon, seg_recon, _, _ = model(orig_modalities, use_checkpoint=False)
            
            # Compute metrics
            orig_np = orig_modalities.cpu().numpy()[0, 0, 64, :, :]  # FLAIR mid-slice
            synth_np = modalities_recon.cpu().numpy()[0, 0, 64, :, :]
            ssim_val = ssim(orig_np, synth_np, data_range=synth_np.max() - synth_np.min())
            psnr_val = psnr(orig_np, synth_np, data_range=synth_np.max() - synth_np.min())
            avg_dice = compute_multi_class_dice(seg_recon, orig_seg)
            subregion_dice = compute_subregion_dice(seg_recon, orig_seg)
            
            if (target_ssim_range[0] <= ssim_val <= target_ssim_range[1]) and (target_dice_range[0] <= avg_dice <= target_dice_range[1]):
                # Save as NIfTI
                modalities_np = modalities_recon.squeeze(0).cpu().numpy()
                seg_np = torch.argmax(seg_recon, dim=1).squeeze(0).cpu().numpy()
                
                for m, name in enumerate(modality_names[:-1]):
                    nii_img = nib.Nifti1Image(modalities_np[m], affine=reference_nii.affine, header=reference_nii.header)
                    nib.save(nii_img, os.path.join(modality_dirs[name], f"synthetic_{num_generated}.nii.gz"))
                
                seg_nii = nib.Nifti1Image(seg_np.astype(np.uint8), affine=reference_nii.affine, header=reference_nii.header)
                nib.save(seg_nii, os.path.join(modality_dirs['seg'], f"synthetic_{num_generated}.nii.gz"))
                
                print(f"Generated {num_generated + 1}/{num_samples} | "
                      f"SSIM: {ssim_val:.4f} | "
                      f"PSNR: {psnr_val:.4f} | "
                      f"Avg Dice: {avg_dice:.4f} | "
                      f"NCR Dice: {subregion_dice['NCR']:.4f} | "
                      f"ET Dice: {subregion_dice['ET']:.4f} | "
                      f"ED Dice: {subregion_dice['ED']:.4f} | "
                      f"WT Dice: {subregion_dice['WT']:.4f} | "
                      f"TC Dice: {subregion_dice['TC']:.4f}")
                
                num_generated += 1
                
                if visualize and num_generated < 5:
                    fig, axes = plt.subplots(4, 1, figsize=(5, 20))
                    axes[0].imshow(orig_np, cmap='gray')
                    axes[0].set_title(f"Orig {num_generated} - FLAIR")
                    axes[0].axis('off')
                    
                    orig_seg_labels = torch.argmax(orig_seg, dim=1).cpu().numpy()[0, 64, :, :]
                    axes[1].imshow(orig_seg_labels, cmap='jet', vmin=0, vmax=3)
                    axes[1].set_title(f"Orig {num_generated} - Seg")
                    axes[1].axis('off')
                    
                    axes[2].imshow(synth_np, cmap='gray')
                    axes[2].set_title(f"Synth {num_generated} - FLAIR")
                    axes[2].axis('off')
                    
                    seg_labels = torch.argmax(seg_recon, dim=1).cpu().numpy()[0, 64, :, :]
                    axes[3].imshow(seg_labels, cmap='jet', vmin=0, vmax=3)
                    axes[3].set_title(f"Synth {num_generated} - Seg")
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"synthetic_{num_generated}_visualization.png"))
                    plt.close()
                
                del modalities_recon, seg_recon
                torch.cuda.empty_cache()
                gc.collect()
