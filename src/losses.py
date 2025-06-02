import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def dice_loss(pred, target, smooth=1e-5, class_weights=None, epoch=0, apply_weights_after_epoch=0):
    pred = torch.softmax(pred, dim=1)
    fg_pred = pred[:, 1:]
    fg_target = target[:, 1:]
    
    intersection = (fg_pred * fg_target).sum(dim=(2, 3, 4))
    union = fg_pred.sum(dim=(2, 3, 4)) + fg_target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    
    class_presence = fg_target.sum(dim=(2, 3, 4)) > 0
    dice = dice * class_presence.float()
    
    dice_loss_per_class = 1 - dice
    
    if class_weights is not None and epoch >= apply_weights_after_epoch:
        dice_loss_per_class = dice_loss_per_class * class_weights.view(1, 3)
    
    num_present = class_presence.sum(dim=1).clamp(min=1)
    dice_loss = (dice_loss_per_class.sum(dim=1) / num_present)
    return dice_loss.mean()

def compute_dice(pred, target, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    fg_pred = pred[:, 1:]  # NCR, ED, ET
    fg_target = target[:, 1:]
    
    intersection = (fg_pred * fg_target).sum(dim=(2, 3, 4))
    union = fg_pred.sum(dim=(2, 3, 4)) + fg_target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    
    class_presence = fg_target.sum(dim=(2, 3, 4)) > 0
    dice = dice * class_presence.float()
    
    num_present = class_presence.sum(dim=1).clamp(min=1)
    mean_dice = dice.sum(dimances) / num_present
    return dice, mean_dice

def perceptual_loss(real, fake, perceptual_net):
    real_input = real[:, [1, 2, 3], :, :, :]
    fake_input = fake[:, [1, 2, 3], :, :, :]
    with torch.amp.autocast('cuda'):
        real_features = perceptual_net(real_input)
        fake_features = perceptual_net(fake_input)
        loss = nn.MSELoss()(real_features, fake_features)
    return loss

def gradient_loss(real, fake):
    def compute_gradients(img):
        grad_x = torch.abs(img[:, :, :, :-1, :] - img[:, :, :, 1:, :])
        grad_y = torch.abs(img[:, :, :-1, :, :] - img[:, :, 1:, :, :])
        grad_z = torch.abs(img[:, :, :, :, :-1] - img[:, :, :, :, 1:])
        return grad_x.mean() + grad_y.mean() + grad_z.mean()
    
    real_grad = compute_gradients(real)
    fake_grad = compute_gradients(fake)
    return torch.abs(real_grad - fake_grad)

def focal_loss(pred, target, alpha=0.25, gamma=2.0, smooth=1e-5):
    pred = torch.softmax(pred, dim=1)
    fg_pred = pred[:, 1:]  # NCR, ED, ET (exclude background)
    fg_target = target[:, 1:]  # Shape: (batch_size, 3, D, H, W)
    
    bce = -fg_target * torch.log(fg_pred + smooth) - (1 - fg_target) * torch.log(1 - fg_pred + smooth)
    
    pt = torch.exp(-bce)
    focal_weight = alpha * (1 - pt) ** gamma
    
    focal_loss = focal_weight * bce
    
    class_presence = fg_target.sum(dim=(2, 3, 4)) > 0
    class_presence = class_presence.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    class_presence = class_presence.expand(-1, -1, fg_target.size(2), fg_target.size(3), fg_target.size(4))
    
    focal_loss = focal_loss * class_presence.float()
    
    focal_loss = focal_loss.sum(dim=(2, 3, 4))
    num_present = class_presence.sum(dim=(2, 3, 4)).clamp(min=1)
    focal_loss = focal_loss / num_present
    
    return focal_loss.mean()

def boundary_loss(y_pred, y_true):
    grad_pred = torch.gradient(y_pred, dim=(2, 3, 4))
    grad_true = torch.gradient(y_true.float(), dim=(2, 3, 4))
    
    loss = 0.0
    for gp, gt in zip(grad_pred, grad_true):
        loss += torch.mean(torch.abs(gp - gt))
    
    return loss / len(grad_pred)

def visualize_results(modalities, modalities_recon, seg_gt, seg_reconstructed, epoch, output_dir="visualizations"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    batch_size = modalities.size(0)
    modalities = modalities.cpu().detach().numpy()
    modalities_recon = modalities_recon.cpu().detach().numpy()
    seg_gt = seg_gt.cpu().numpy()
    seg_recon = seg_recon.cpu().numpy()
    
    mid_slice = modalities.shape[2] // 2
    
    for b in range(batch_size):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        modality_names = ['FLAIR', 'T1', 'T1CE', 'T2']
        for i in range(4):
            axes[0, i].imshow(modalities[b, i, mid_slice, :, :], cmap='gray')
            axes[0, i].set_title(f'Input {modality_names[i]}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(modalities_recon[b, i, mid_slice, :, :], cmap='gray')
            axes[1, i].set_title(f'Recon {modality_names[i]}')
            axes[1, i].axis('off')
        
        seg_gt_labels = np.argmax(seg_gt[b], axis=0)
        seg_recon_labels = np.argmax(seg_recon[b], axis=0)
        
        axes[0, 4].imshow(seg_gt_labels[mid_slice, :, :], cmap='jet', vmin=0, vmax=3)
        axes[0, 4].set_title('Ground Truth Seg')
        axes[0, 4].axis('off')
        
        axes[1, 4].imshow(seg_recon_labels[mid_slice, :, :], cmap='jet', vmin=0, vmax=3)
        axes[1, 4].set_title('Predicted Seg')
        axes[1, 4].axis('off')
        
        plt.suptitle(f'VAE-WGAN Epoch {epoch} - Sample {b}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'vae_wgan_epoch_{epoch}_sample_{b}.png'))
        plt.close()
