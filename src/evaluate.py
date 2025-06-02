import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.spatial.distance import directed_hausdorff
from dataset import NIfTIDataset
from models import VAE
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.cuda.amp as amp

def find_tumor_slice(seg):
    num_slices = seg.shape[0]
    for s in range(num_slices):
        slice_seg = seg[s]
        if np.any(slice_seg > 0):
            return s
    return num_slices // 2

def visualize_results(modalities, modalities_recon, seg_gt, seg_recon, sample_idx, output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    modalities = modalities[0].cpu().numpy()
    modalities_recon = modalities_recon[0].cpu().numpy()
    
    if seg_gt.dim() == 5 and seg_gt.shape[1] == 4:
        seg_gt = torch.argmax(seg_gt, dim=1)
    seg_gt = seg_gt[0].cpu().numpy()
    
    if seg_recon.dim() == 5 and seg_recon.shape[1] == 4:
        seg_recon = torch.argmax(seg_recon, dim=1)
    seg_recon = seg_recon[0].cpu().numpy()
    
    slice_idx = find_tumor_slice(seg_gt)
    
    modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
    tumor_colors = {1: 'red', 2: 'green', 3: 'blue'}
    tumor_labels = {1: 'NCR', 2: 'ED', 3: 'ET'}
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i in range(4):
        axes[0, i].imshow(modalities[i, slice_idx], cmap='gray')
        axes[0, i].set_title(f"Real {modality_names[i]}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(modalities_recon[i, slice_idx], cmap='gray')
        axes[1, i].set_title(f"Synth {modality_names[i]}")
        axes[1, i].axis('off')
    
    axes[0, 4].imshow(modalities[0, slice_idx], cmap='gray')
    handles = []
    labels = []
    for c in range(1, 4):
        mask = (seg_gt[slice_idx] == c)
        if mask.sum() > 0:
            contour = axes[0, 4].contour(mask, colors=[tumor_colors[c]], levels=[0.5], linewidths=1)
            handles.append(contour.collections[0])
            labels.append(f"{tumor_labels[c]}")
    axes[0, 4].set_title("Ground Truth Seg")
    axes[0, 4].axis('off')
    if handles:
        axes[0, 4].legend(handles, labels, loc='upper right', fontsize='small')
    
    axes[1, 4].imshow(modalities[0, slice_idx], cmap='gray')
    handles = []
    labels = []
    for c in range(1, 4):
        mask = (seg_recon[slice_idx] == c)
        if mask.sum() > 0:
            contour = axes[1, 4].contour(mask, colors=[tumor_colors[c]], levels=[0.5], linewidths=1)
            handles.append(contour.collections[0])
            labels.append(f"{tumor_labels[c]}")
    axes[1, 4].set_title("Generated Seg")
    axes[1, 4].axis('off')
    if handles:
        axes[1, 4].legend(handles, labels, loc='upper right', fontsize='small')
    
    plt.suptitle(f"Sample {sample_idx}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"montage_sample_{sample_idx}.png"))
    plt.close()

def compute_per_class_dice(pred, target, num_classes=4):
    pred = torch.argmax(pred, dim=1)
    
    if target.dim() == 5 and target.shape[1] == num_classes:
        target = torch.argmax(target, dim=1)
    elif target.dim() != 4:
        raise ValueError(f"Expected target shape (B, D, H, W), got {target.shape}")
    
    dice_scores = []
    for c in range(1, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(dice.item())
    return dice_scores

def compute_brats_dice(pred, target):
    pred = torch.argmax(pred, dim=1)
    
    if target.dim() == 5 and target.shape[1] == 4:
        target = torch.argmax(target, dim=1)
    elif target.dim() != 4:
        raise ValueError(f"Expected target shape (B, D, H, W), got {target.shape}")
    
    pred_wt = (pred > 0).float()
    target_wt = (target > 0).float()
    pred_tc = torch.zeros_like(pred).float()
    pred_tc[(pred == 1) | (pred == 3)] = 1.0
    target_tc = torch.zeros_like(target).float()
    target_tc[(target == 1) | (target == 3)] = 1.0
    pred_et = (pred == 3).float()
    target_et = (target == 3).float()
    
    def dice_score(pred_region, target_region):
        intersection = (pred_region * target_region).sum()
        union = pred_region.sum() + target_region.sum()
        return (2.0 * intersection + 1e-6) / (union + 1e-6)
    
    wt_dice = dice_score(pred_wt, target_wt)
    tc_dice = dice_score(pred_tc, target_tc)
    et_dice = dice_score(pred_et, target_et)
    return wt_dice.item(), tc_dice.item(), et_dice.item()

def compute_hausdorff_distance(pred, target, num_classes=4):
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    
    if target.dim() == 5 and target.shape[1] == num_classes:
        target = torch.argmax(target, dim=1)
    elif target.dim() != 4:
        raise ValueError(f"Expected target shape (B, D, H, W), got {target.shape}")
    target = target.cpu().numpy()
    
    batch_size = pred.shape[0]
    hd_scores = {f"class_{c}": [] for c in range(1, num_classes)}
    hd95_scores = {f"class_{c}": [] for c in range(1, num_classes)}
    hd_regions = {"WT": [], "TC": [], "ET": []}
    hd95_regions = {"WT": [], "TC": [], "ET": []}
    
    for b in range(batch_size):
        pred_b = pred[b]
        target_b = target[b]
        
        for c in range(1, num_classes):
            pred_c = (pred_b == c).astype(np.uint8)
            target_c = (target_b == c).astype(np.uint8)
            if pred_c.sum() == 0 or target_c.sum() == 0:
                continue
            coords_pred = np.where(pred_c)
            coords_target = np.where(target_c)
            points_pred = np.stack(coords_pred, axis=1)
            points_target = np.stack(coords_target, axis=1)
            if len(points_pred) == 0 or len(points_target) == 0:
                continue
            hd = max(directed_hausdorff(points_pred, points_target)[0],
                     directed_hausdorff(points_target, points_pred)[0])
            distances = []
            for p in points_pred:
                dists = np.linalg.norm(points_target - p, axis=1)
                distances.append(np.min(dists))
            for p in points_target:
                dists = np.linalg.norm(points_pred - p, axis=1)
                distances.append(np.min(dists))
            hd95 = np.percentile(distances, 95) if distances else hd
            hd_scores[f"class_{c}"].append(hd)
            hd95_scores[f"class_{c}"].append(hd95)
        
        pred_wt = (pred_b > 0).astype(np.uint8)
        target_wt = (target_b > 0).astype(np.uint8)
        pred_tc = np.zeros_like(pred_b, dtype=np.uint8)
        pred_tc[(pred_b == 1) | (pred_b == 3)] = 1
        target_tc = np.zeros_like(target_b, dtype=np.uint8)
        target_tc[(target_b == 1) | (target_b == 3)] = 1
        pred_et = (pred_b == 3).astype(np.uint8)
        target_et = (target_b == 3).astype(np.uint8)
        
        for region, pred_r, target_r in [("WT", pred_wt, target_wt), ("TC", pred_tc, target_tc), ("ET", pred_et, target_et)]:
            if pred_r.sum() == 0 or target_r.sum() == 0:
                continue
            coords_pred = np.where(pred_r)
            coords_target = np.where(target_r)
            points_pred = np.stack(coords_pred, axis=1)
            points_target = np.stack(coords_target, axis=1)
            if len(points_pred) == 0 or len(points_target) == 0:
                continue
            hd = max(directed_hausdorff(points_pred, points_target)[0],
                     directed_hausdorff(points_target, points_pred)[0])
            distances = []
            for p in points_pred:
                dists = np.linalg.norm(points_target - p, axis=1)
                distances.append(np.min(dists))
            for p in points_target:
                dists = np.linalg.norm(points_pred - p, axis=1)
                distances.append(np.min(dists))
            hd95 = np.percentile(distances, 95) if distances else hd
            hd_regions[region].append(hd)
            hd95_regions[region].append(hd95)
    
    avg_hd = {k: np.mean(v) if v else float('nan') for k, v in hd_scores.items()}
    avg_hd95 = {k: np.mean(v) if v else float('nan') for k, v in hd95_scores.items()}
    avg_hd_regions = {k: np.mean(v) if v else float('nan') for k, v in hd_regions.items()}
    avg_hd95_regions = {k: np.mean(v) if v else float('nan') for k, v in hd95_regions.items()}
    return avg_hd, avg_hd95, avg_hd_regions, avg_hd95_regions

def compute_reconstruction_metrics(real, synth):
    real = real.cpu().numpy()
    synth = synth.cpu().numpy()
    batch_size, num_modalities = real.shape[0], real.shape[1]
    ssim_scores = []
    psnr_scores = []
    mae_scores = []
    nmse_scores = []
    
    for b in range(batch_size):
        for m in range(num_modalities):
            real_m = real[b, m]
            synth_m = synth[b, m]
            ssim_m = 0.0
            psnr_m = 0.0
            mae_m = 0.0
            nmse_m = 0.0
            num_slices = real_m.shape[0]
            for s in range(num_slices):
                real_slice = real_m[s]
                synth_slice = synth_m[s]
                ssim_val = ssim(real_slice, synth_slice, data_range=1.0)
                ssim_m += ssim_val
                psnr_val = psnr(real_slice, synth_slice, data_range=1.0)
                psnr_m += psnr_val
                mae_val = np.mean(np.abs(real_slice - synth_slice))
                mae_m += mae_val
                mse = np.mean((real_slice - synth_slice) ** 2)
                norm = np.mean(real_slice ** 2)
                nmse_val = mse / norm if norm > 0 else 0.0
                nmse_m += nmse_val
            ssim_m /= num_slices
            psnr_m /= num_slices
            mae_m /= num_slices
            nmse_m /= num_slices
            ssim_scores.append(ssim_m)
            psnr_scores.append(psnr_m)
            mae_scores.append(mae_m)
            nmse_scores.append(nmse_m)
    
    return (np.mean(ssim_scores), np.mean(psnr_scores), np.mean(mae_scores), np.mean(nmse_scores))

def evaluate_synthetic_data(real_loader, synth_loader, model, device, config, output_dir="visualizations"):
    model.eval()
    perceptual_net = r3d_18(weights=R3D_18_Weights.DEFAULT).to(device).eval()
    
    all_ncr_dice, all_ed_dice, all_et_dice = [], [], []
    all_wt_dice, all_tc_dice, all_et_dice_brats = [], [], []
    all_ssim, all_psnr, all_mae, all_nmse = [], [], [], []
    hd_per_class, hd95_per_class, hd_regions, hd95_regions = [], [], [], []
    
    num_samples = min(len(real_loader.dataset), len(synth_loader.dataset))
    print(f"Evaluating {num_samples} samples.")
    
    sample_idx = 0
    for real_batch, synth_batch in tqdm(zip(real_loader, synth_loader), total=len(real_loader), desc="Evaluating"):
        real_modalities = real_batch['modalities'].to(device)
        real_seg = real_batch['seg'].to(device)
        synth_modalities = synth_batch['modalities'].to(device)
        synth_seg = synth_batch['seg'].to(device)
        
        with torch.cuda.amp.autocast():
            _, real_seg_recon, _, _ = model(real_modalities)
            _, synth_seg_recon, _, _ = model(synth_modalities)
        
        per_class_dice = compute_per_class_dice(synth_seg_recon, real_seg)
        all_ncr_dice.append(per_class_dice[0])
        all_ed_dice.append(per_class_dice[1])
        all_et_dice.append(per_class_dice[2])
        
        wt_dice, tc_dice, et_dice = compute_brats_dice(synth_seg_recon, real_seg)
        all_wt_dice.append(wt_dice)
        all_tc_dice.append(tc_dice)
        all_et_dice_brats.append(et_dice)
        
        ssim_val, psnr_val, mae_val, nmse_val = compute_reconstruction_metrics(real_modalities, synth_modalities)
        all_ssim.append(ssim_val)
        all_psnr.append(psnr_val)
        all_mae.append(mae_val)
        all_nmse.append(nmse_val)
        
        hd_pc, hd95_pc, hd_r, hd95_r = compute_hausdorff_distance(synth_seg_recon, real_seg)
        hd_per_class.append(hd_pc)
        hd95_per_class.append(hd95_pc)
        hd_regions.append(hd_r)
        hd95_regions.append(hd95_r)
        
        if sample_idx % 5 == 0:
            visualize_results(real_modalities, synth_modalities, real_seg, synth_seg_recon, sample_idx, output_dir)
        
        sample_idx += 1
        torch.cuda.empty_cache()
    
    avg_ncr_dice = np.mean(all_ncr_dice)
    avg_ed_dice = np.mean(all_ed_dice)
    avg_et_dice = np.mean(all_et_dice)
    avg_wt_dice = np.mean(all_wt_dice)
    avg_tc_dice = np.mean(all_tc_dice)
    avg_et_dice_brats = np.mean(all_et_dice_brats)
    avg_ssim = np.mean(all_ssim)
    avg_psnr = np.mean(all_psnr)
    avg_mae = np.mean(all_mae)
    avg_nmse = np.mean(all_nmse)
    
    avg_hd_per_class = {k: np.nanmean([d[k] for d in hd_per_class]) for k in hd_per_class[0].keys()}
    avg_hd95_per_class = {k: np.nanmean([d[k] for d in hd95_per_class]) for k in hd95_per_class[0].keys()}
    avg_hd_regions = {k: np.nanmean([d[k] for d in hd_regions]) for k in hd_regions[0].keys()}
    avg_hd95_regions = {k: np.nanmean([d[k] for d in hd95_regions]) for k in hd95_regions[0].keys()}
    
    # Print and save results
    results = [
        "Evaluation Results:",
        "Per-Tumor-Class Dice Scores:",
        f"NCR Dice: {avg_ncr_dice:.4f}",
        f"ED Dice: {avg_ed_dice:.4f}",
        f"ET Dice: {avg_et_dice:.4f}",
        "\nBraTS Dice Scores:",
        f"WT Dice: {avg_wt_dice:.4f}",
        f"TC Dice: {avg_tc_dice:.4f}",
        f"ET Dice: {avg_et_dice_brats:.4f}",
        "\nReconstruction Metrics:",
        f"SSIM: {avg_ssim:.4f}",
        f"PSNR: {avg_psnr:.4f} dB",
        f"MAE: {avg_mae:.4f}",
        f"NMSE: {avg_nmse:.4f}",
        "\nHausdorff Distance (HD):",
        "Per Class:",
        *[f"{k}: {v:.2f} voxels" for k, v in avg_hd_per_class.items()],
        "HD95 Per Class:",
        *[f"{k}: {v:.2f} voxels" for k, v in avg_hd95_per_class.items()],
        "Regions:",
        *[f"{k}: {v:.2f} voxels" for k, v in avg_hd_regions.items()],
        "HD95 Regions:",
        *[f"{k}: {v:.2f} voxels" for k, v in avg_hd95_regions.items()]
    ]
    
    for line in results:
        print(line)
    
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w") as f:
        for line in results:
            f.write(line + "\n")
