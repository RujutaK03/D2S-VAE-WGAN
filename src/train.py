import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import gc
from torchvision.models.video import r3d_18, R3D_18_Weights
from models import VAE
from losses import (kl_divergence, dice_loss, compute_dice, perceptual_loss,
                    gradient_loss, focal_loss, boundary_loss, visualize_results)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_vae(train_loader, val_loader, config, device, checkpoint_path=None, start_epoch=11):
    # Load hyperparameters
    latent_dim = config['latent_dim']
    num_seg_classes = config['num_seg_classes']
    lr = config['lr']
    beta1 = config['beta1']
    lambda_recon = config['lambda_recon']
    lambda_dice = config['lambda_dice']
    lambda_focal = config['lambda_focal']
    lambda_kl = config['lambda_kl']
    lambda_perceptual = config['lambda_perceptual']
    lambda_grad = config['lambda_grad']
    lambda_boundary = config['lambda_boundary']
    num_epochs = config['num_epochs']
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)

    # Initialize model
    vae = VAE(in_channels=4, latent_dim=latent_dim, num_seg_classes=num_seg_classes, dropout_p=0.7).to(device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        vae.load_state_dict(checkpoint['vae_state_dict'], strict=False)
        optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=5e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.0, patience=5, verbose=True)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(beta1, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
 verbose=True)

    # Initialize perceptual network
    perceptual_net = r3d_18(weights=R3D_18_Weights.DEFAULT).to(device).eval()
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training phase
    for epoch in range(start_epoch, num_epochs + 1):
        vae.train()
        epoch_recon_total, epoch_dice_loss_total, epoch_kl_total = 0, 0, 0
        epoch_perceptual_total, epoch_grad_total, epoch_focal_total = 0, 0, 0
        epoch_boundary_total, epoch_mean_dice_total = 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Pretraining Epoch {epoch}/{num_epochs}", leave=True)
        
        for i, batch in enumerate(progress_bar):
            modalities = batch['modalities'].to(device)
            seg_gt = batch['seg'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                modalities_recon, seg_recon, mu, logvar = vae(modalities, use_checkpoint=True)
                recon_loss = nn.MSELoss()(modalities_recon, modalities)
                dice_loss_value = dice_loss(seg_recon, seg_gt, class_weights=class_weights, epoch=epoch)
                focal_loss_value = focal_loss(seg_recon, seg_gt)
                dice_per_class, mean_dice = compute_dice(seg_recon, seg_gt)
                kl_loss = kl_divergence(mu, logvar)
                p_loss = perceptual_loss(modalities, modalities_recon, perceptual_net)
                grad_loss = gradient_loss(modalities, modalities_recon)
                boundary_loss_value = boundary_loss(seg_recon, seg_gt)

                loss = (
                    lambda_recon * recon_loss +
                    lambda_dice * dice_loss_value +
                    lambda_focal * focal_loss_value +
                    lambda_kl * kl_loss +
                    lambda_perceptual * p_loss +
                    lambda_grad * grad_loss +
                    lambda_boundary * boundary_loss_value
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_recon_total += recon_loss.item()
            epoch_dice_loss_total += dice_loss_value.item()
            epoch_focal_total += focal_loss_value.item()
            epoch_mean_dice_total += mean_dice.mean().item()
            epoch_kl_total += kl_loss.item()
            epoch_perceptual_total += p_loss.item()
            epoch_grad_total += grad_loss.item()
            epoch_boundary_total += boundary_loss_value.item()
            
            progress_bar.set_postfix({
                "Recon": f"{recon_loss.item():.4f}",
                "Dice Loss": f"{dice_loss_value.item():.4f}",
                "Focal": f"{focal_loss_value.item():.4f}",
                "NCR": f"{dice_per_class[:,0].mean():.4f}",
                "ED": f"{dice_per_class[:,1].mean():.4f}",
                "ET": f"{dice_per_class[:,2].mean():.4f}",
                "Dice": f"{mean_dice.mean():.4f}",
                "KL": f"{kl_loss.item():.4f}",
                "Perceptual": f"{p_loss.item():.4f}",
                "Grad": f"{grad_loss.item():.4f}",
                "Boundary": f"{boundary_loss_value.item():.4f}"
            })
            
            torch.cuda.empty_cache()
            gc.collect()
        
        # Validation
        vae.eval()
        val_recon_total, val_dice_loss_total, val_focal_total = 0, 0, 0
        val_mean_dice_total, val_boundary_loss_total = 0, 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                val_modalities = batch['modalities'].to(device)
                val_seg_gt = batch['seg'].to(device)
                with torch.cuda.amp.autocast():
                    val_modalities_recon, val_seg_recon, _, _ = vae(val_modalities, use_checkpoint=False)
                    recon_loss = nn.MSELoss()(val_modalities_recon, val_modalities)
                    dice_loss_value = dice_loss(val_seg_recon, val_seg_gt, class_weights=class_weights, epoch=epoch)
                    focal_loss_value = focal_loss(val_seg_recon, val_seg_gt)
                    val_boundary_loss = boundary_loss(val_seg_recon, val_seg_gt)
                    val_dice_per_class, mean_dice = compute_dice(val_seg_recon, val_seg_gt)
                val_recon_total += recon_loss.item()
                val_dice_loss_total += dice_loss_value.item()
                val_focal_total += focal_loss_value.item()
                val_mean_dice_total += mean_dice.mean().item()
                val_boundary_loss_total += val_boundary_loss.item()

                if batch_idx == 0:
                    visualize_results(val_modalities, val_modalities_recon, val_seg_gt, val_seg_recon, epoch)
                
                torch.cuda.empty_cache()
                gc.collect()
        
        avg_recon = epoch_recon_total / len(train_loader)
        avg_dice_loss = epoch_dice_loss_total / len(train_loader)
        avg_focal = epoch_focal_total / len(train_loader)
        avg_mean_dice = epoch_mean_dice_total / len(train_loader)
        avg_kl = epoch_kl_total / len(train_loader)
        avg_perceptual = epoch_perceptual_total / len(train_loader)
        avg_grad = epoch_grad_total / len(train_loader)
        avg_boundary_loss = epoch_boundary_total / len(train_loader)
        avg_val_recon = val_recon_total / len(val_loader)
        avg_val_dice_loss = val_dice_loss_total / len(val_loader)
        avg_val_focal = val_focal_total / len(val_loader)
        avg_val_mean_dice = val_mean_dice_total / len(val_loader)
        avg_val_boundary_loss = val_boundary_loss_total / len(val_loader)
        
        print(f"Pretraining Epoch [{epoch}/{num_epochs}] "
              f"Train Recon: {avg_recon:.4f}, Dice Loss: {avg_dice_loss:.4f}, Focal Loss: {avg_focal:.4f}, Dice Score: {avg_mean_dice:.4f}, KL: {avg_kl:.4f}, "
              f"Perceptual: {avg_perceptual:.4f}, Grad: {avg_grad:.4f}, Boundary Loss: {avg_boundary_loss:.4f} "
              f"Val Recon: {avg_val_recon:.4f}, Val Dice Loss: {avg_val_dice_loss:.4f}, Val Focal Loss: {avg_val_focal:.4f}, Val Dice Score: {avg_val_mean_dice:.4f}, "
              f"Val Boundary Loss: {avg_val_boundary_loss:.4f}, NCR: {val_dice_per_class[:,0].mean():.4f}, ED: {val_dice_per_class[:,1].mean():.4f}, "
              f"ET: {val_dice_per_class[:,2].mean():.4f}")

        scheduler.step(avg_val_mean_dice)
        
        torch.save({
            'epoch': epoch,
            'vae_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, f"pretrained_vae_epoch_{epoch}.pth")

        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Pretraining complete. Final checkpoint saved as 'pretrained_vae_epoch_{num_epochs}.pth'.")

if __name__ == "__main__":
    config = load_config('../config.yaml')
    print("Run train.py via main.py with appropriate DataLoader instances.")
