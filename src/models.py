import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, in_channels=4, latent_dim=128, num_seg_classes=4, dropout_p=0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_seg_classes = num_seg_classes
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Conv3d(256, latent_dim, 1)
        self.fc_logvar = nn.Conv3d(256, latent_dim, 1)
        
        # Decoder for modalities (with skip connections)
        self.dec_mod1 = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec_mod2 = nn.Sequential(
            nn.ConvTranspose3d(256 + 128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec_mod3 = nn.Sequential(
            nn.ConvTranspose3d(128 + 64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.dec_mod4 = nn.Sequential(
            nn.ConvTranspose3d(64 + 32, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )
        self.dec_mod5 = nn.Sequential(
            nn.ConvTranspose3d(32 + in_channels, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, in_channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Decoder for segmentation (with skip connections)
        self.dec_seg1 = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.dec_seg2 = nn.Sequential(
            nn.ConvTranspose3d(256 + 128, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.dec_seg3 = nn.Sequential(
            nn.ConvTranspose3d(128 + 64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.dec_seg4 = nn.Sequential(
            nn.ConvTranspose3d(64 + 32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.dec_seg5 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, num_seg_classes, 3, stride=1, padding=1),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, use_checkpoint=True):
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        
        mu = self.fc_mu(enc4_out)
        logvar = self.fc_logvar(enc4_out)
        z_sample = self.reparameterize(mu, logvar)
        
        # Modality Decoder with Skip Connections
        if use_checkpoint:
            dec_mod1_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_mod1, segments=1, input=z_sample)
            dec_mod2_in = torch.cat([dec_mod1_out, enc3_out], dim=1)
            dec_mod2_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_mod2, segments=1, input=dec_mod2_in)
            dec_mod3_in = torch.cat([dec_mod2_out, enc2_out], dim=1)
            dec_mod3_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_mod3, segments=1, input=dec_mod3_in)
            dec_mod4_in = torch.cat([dec_mod3_out, enc1_out], dim=1)
            dec_mod4_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_mod4, segments=1, input=dec_mod4_in)
            dec_mod5_in = torch.cat([dec_mod4_out, x], dim=1)
            modalities_recon = torch.utils.checkpoint.checkpoint_sequential(self.dec_mod5, segments=2, input=dec_mod5_in)
            
            # Segmentation Decoder with Skip Connections
            dec_seg1_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_seg1, segments=1, input=z_sample)
            dec_seg2_in = torch.cat([dec_seg1_out, enc3_out], dim=1)
            dec_seg2_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_seg2, segments=1, input=dec_seg2_in)
            dec_seg3_in = torch.cat([dec_seg2_out, enc2_out], dim=1)
            dec_seg3_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_seg3, segments=1, input=dec_seg3_in)
            dec_seg4_in = torch.cat([dec_seg3_out, enc1_out], dim=1)
            dec_seg4_out = torch.utils.checkpoint.checkpoint_sequential(self.dec_seg4, segments=1, input=dec_seg4_in)
            seg_recon = torch.utils.checkpoint.checkpoint_sequential(self.dec_seg5, segments=1, input=dec_seg4_out)
        else:
            dec_mod1_out = self.dec_mod1(z_sample)
            dec_mod2_in = torch.cat([dec_mod1_out, enc3_out], dim=1)
            dec_mod2_out = self.dec_mod2(dec_mod2_in)
            dec_mod3_in = torch.cat([dec_mod2_out, enc2_out], dim=1)
            dec_mod3_out = self.dec_mod3(dec_mod3_in)
            dec_mod4_in = torch.cat([dec_mod3_out, enc1_out], dim=1)
            dec_mod4_out = self.dec_mod4(dec_mod4_in)
            dec_mod5_in = torch.cat([dec_mod4_out, x], dim=1)
            modalities_recon = self.dec_mod5(dec_mod5_in)
            
            # Segmentation Decoder with Skip Connections
            dec_seg1_out = self.dec_seg1(z_sample)
            dec_seg2_in = torch.cat([dec_seg1_out, enc3_out], dim=1)
            dec_seg2_out = self.dec_seg2(dec_seg2_in)
            dec_seg3_in = torch.cat([dec_seg2_out, enc2_out], dim=1)
            dec_seg3_out = self.dec_seg3(dec_seg3_in)
            dec_seg4_in = torch.cat([dec_seg3_out, enc1_out], dim=1)
            dec_seg4_out = self.dec_seg4(dec_seg4_in)
            seg_reconstructed = self.dec_seg5(dec_seg4_out)
        
        return modalities_recon, seg_recon, mu, logvar
