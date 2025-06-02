import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

class NIfTIDataset(Dataset):
    def __init__(self, target_dir=None, modalities=None, precomputed_dir=None, transform=None, 
                 normalize=True, visualize_preprocess=False, target_shape=(128, 128, 128), sharpen_alpha=1.0):
        self.precomputed_dir = precomputed_dir
        self.transform = transform
        self.normalize = normalize
        self.visualize_preprocess = visualize_preprocess
        self.target_shape = target_shape
        self.sharpen_alpha = sharpen_alpha

        if precomputed_dir and not target_dir:
            # Precomputed-only mode (e.g., for synthetic data generation)
            self.modalities = modalities or ['flair', 't1', 't1ce', 't2', 'seg']
            self.patient_ids = sorted([
                os.path.basename(f).split('_')[1].split('.')[0]
                for f in glob(os.path.join(precomputed_dir, "sample_*.npz"))
            ])
            if not self.patient_ids:
                raise ValueError(f"No precomputed files found in {precomputed_dir}.")
            print(f"Found {len(self.patient_ids)} precomputed patient samples in {precomputed_dir}.")
        elif target_dir and modalities:
            # Full dataset mode
            self.target_dir = target_dir
            self.modalities = modalities
            self.patient_ids = sorted(set(
                os.path.basename(f).split('_')[1]
                for modality in self.modalities
                for f in glob(os.path.join(target_dir, modality, "*_" + modality + ".nii.gz"))
            ))
            if precomputed_dir:
                try:
                    print("Visualizing a subset of preprocessed data before full precomputation...")
                    self.visualize_preprocessed_subset(num_samples=5)
                    print("Subset visualization complete. Proceeding with full precomputation...")
                    print("Precomputing dataset...")
                    self.precompute()
                except Exception as e:
                    print(f"Precomputation failed: {e}. Falling back to on-the-fly loading.")
                    self.precomputed_dir = None
            if visualize_preprocess:
                self.visualize_preprocessing_sample()
        else:
            raise ValueError("Must provide either precomputed_dir or both target_dir and modalities.")

    def __len__(self):
        return len(self.patient_ids)

    def apply_sharpening(self, data):
        data = data.to('cuda')
        sigma = 1.0
        kernel_size = 3
        x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device='cuda')
        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        num_channels = data.shape[0]
        kernel_1d = kernel_1d.view(1, 1, kernel_size, 1, 1)
        kernel_1d = kernel_1d.expand(num_channels, 1, kernel_size, 1, 1)
        kernel_1d_y = kernel_1d.transpose(2, 3)
        kernel_1d_z = kernel_1d.transpose(2, 4)
        
        data = data.unsqueeze(0)
        blurred = F.conv3d(data, kernel_1d, groups=num_channels, padding=(kernel_size // 2, 0, 0))
        blurred = F.conv3d(blurred, kernel_1d_y, groups=num_channels, padding=(0, kernel_size // 2, 0))
        blurred = F.conv3d(blurred, kernel_1d_z, groups=num_channels, padding=(0, 0, kernel_size // 2))
        blurred = blurred.squeeze(0)
        
        detail = data.squeeze(0) - blurred
        sharpened = data.squeeze(0) + self.sharpen_alpha * detail
        return sharpened.cpu().numpy()

    def load_nifti(self, file_path, is_seg=False):
        img = nib.load(file_path).get_fdata(dtype=np.float32)
        original = img.copy() if self.visualize_preprocess else None

        if is_seg:
            img_tensor = torch.from_numpy(img).float().to('cuda')
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            scale_factor = [t / o for t, o in zip(self.target_shape, img.shape)]
            resampled = F.interpolate(img_tensor, scale_factor=scale_factor, mode='nearest')
            resampled = resampled.squeeze(0).squeeze(0).cpu().numpy()
            resampled = np.round(resampled).astype(np.int32)
            resampled = np.clip(resampled, 0, 4)
            seg_labels = np.zeros(self.target_shape + (4,), dtype=np.float32)
            for label in [0, 1, 2, 4]:
                idx = 0 if label == 0 else (1 if label == 1 else (2 if label == 2 else 3))
                seg_labels[..., idx] = (resampled == label).astype(np.float32)
            
            seg_labels = torch.from_numpy(seg_labels).float().to('cuda')
            sigma = 0.5
            kernel_size = 3
            x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device='cuda')
            kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            num_channels = seg_labels.shape[-1]
            kernel_1d = kernel_1d.view(1, 1, kernel_size, 1, 1)
            kernel_1d = kernel_1d.expand(num_channels, 1, kernel_size, 1, 1)
            kernel_1d_y = kernel_1d.transpose(2, 3)
            kernel_1d_z = kernel_1d.transpose(2, 4)
            
            seg_labels = seg_labels.permute(3, 0, 1, 2).unsqueeze(0)
            smoothed = F.conv3d(seg_labels, kernel_1d, groups=num_channels, padding=(kernel_size // 2, 0, 0))
            smoothed = F.conv3d(smoothed, kernel_1d_y, groups=num_channels, padding=(0, kernel_size // 2, 0))
            smoothed = F.conv3d(smoothed, kernel_1d_z, groups=num_channels, padding=(0, 0, kernel_size // 2))
            smoothed_seg = smoothed.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            
            return smoothed_seg, original
        else:
            img_tensor = torch.from_numpy(img).float().to('cuda')
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            scale_factor = [t / o for t, o in zip(self.target_shape, img.shape)]
            resampled = F.interpolate(img_tensor, scale_factor=scale_factor, mode='trilinear', align_corners=False)
            resampled = resampled.squeeze(0).squeeze(0)
            sharpened = self.apply_sharpening(resampled[None, ...])[0]
            if self.normalize:
                min_val, max_val = sharpened.min(), sharpened.max()
                if max_val > min_val:
                    sharpened = (sharpened - min_val) / (max_val - min_val + 1e-8)
            return sharpened, original

    def precompute(self):
        os.makedirs(self.precomputed_dir, exist_ok=True)
        total_patients = len(self)
        print(f"Starting precomputation for {total_patients} patients...")
        for idx in tqdm(range(total_patients), desc="Precomputing", unit="patient"):
            patient_id = self.patient_ids[idx]
            patient_data = []
            modality_files = []
            try:
                for modality in self.modalities:
                    file_path = os.path.join(self.target_dir, modality, f"BraTS2021_{patient_id}_{modality}.nii.gz")
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Missing file: {file_path}")
                    modality_files.append(file_path)
                    is_seg = (modality == 'seg')
                    data, _ = self.load_nifti(file_path, is_seg=is_seg)
                    if is_seg:
                        seg_data = data
                    else:
                        patient_data.append(data)
                modalities_data = np.stack(patient_data, axis=0).astype(np.float16)
                seg_data = seg_data.astype(np.float16)
                
                precomputed_path = os.path.join(self.precomputed_dir, f"sample_{patient_id}.npz")
                np.savez_compressed(
                    precomputed_path,
                    modalities=modalities_data,
                    seg=seg_data
                )
            except Exception as e:
                print(f"Error precomputing patient {patient_id}: {e}")
                if 'precomputed_path' in locals() and os.path.exists(precomputed_path):
                    try:
                        os.remove(precomputed_path)
                    except OSError:
                        pass
                continue
        
        print(f"Precomputation completed for {total_patients} patients.")

    def visualize_preprocessed_subset(self, num_samples=5):
        num_samples = min(num_samples, len(self))
        print(f"Visualizing preprocessed data for {num_samples} samples...")
        
        for idx in range(num_samples):
            patient_id = self.patient_ids[idx]
            patient_data = []
            original_modalities = []
            for modality in self.modalities:
                file_path = os.path.join(self.target_dir, modality, f"BraTS2021_{patient_id}_{modality}.nii.gz")
                is_seg = (modality == 'seg')
                data, original = self.load_nifti(file_path, is_seg=is_seg)
                if is_seg:
                    seg_data = data
                    original_seg = original
                else:
                    patient_data.append(data)
                    original_modalities.append(original)
            
            modalities = np.stack(patient_data, axis=0).astype(np.float32)
            seg = seg_data
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for j, modality in enumerate(['FLAIR', 'T1', 'T1ce', 'T2']):
                axes[0, j].imshow(original_modalities[j][:, :, 77], cmap='gray')
                axes[0, j].set_title(f"Original {modality} (z=77)")
                axes[0, j].axis('off')
                axes[1, j].imshow(modalities[j, :, :, 64], cmap='gray')
                axes[1, j].set_title(f"Preprocessed {modality} (z=64)")
                axes[1, j].axis('off')
            
            remapped_seg = np.argmax(seg, axis=-1)
            remapped_seg_display = np.zeros_like(remapped_seg, dtype=np.int32)
            remapped_seg_display[remapped_seg == 0] = 0
            remapped_seg_display[remapped_seg == 1] = 1
            remapped_seg_display[remapped_seg == 2] = 2
            remapped_seg_display[remapped_seg == 3] = 4
            
            axes[0, 4].imshow(original_seg[:, :, 77], cmap='jet', vmin=0, vmax=4)
            axes[0, 4].set_title("Original Seg (z=77)")
            axes[0, 4].axis('off')
            axes[1, 4].imshow(remapped_seg_display[:, :, 64], cmap='jet', vmin=0, vmax=4)
            axes[1, 4].set_title("Preprocessed Seg (z=64)")
            axes[1, 4].axis('off')
            
            plt.suptitle(f"Patient {patient_id}: Original vs Preprocessed")
            plt.tight_layout()
            plt.show()

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        if self.precomputed_dir and os.path.exists(os.path.join(self.precomputed_dir, f"sample_{patient_id}.npz")):
            data = np.load(os.path.join(self.precomputed_dir, f"sample_{patient_id}.npz"))
            modalities = data['modalities'].astype(np.float32)
            seg = data['seg'].astype(np.float32)
            if seg.shape[-1] == 4:
                seg = seg.transpose(3, 0, 1, 2)
            sample = {'modalities': modalities, 'seg': seg}
            if self.transform:
                sample = self.transform(sample)
            return sample
        elif hasattr(self, 'target_dir'):
            patient_data = []
            original_modalities = []
            for modality in self.modalities:
                file_path = os.path.join(self.target_dir, modality, f"BraTS2021_{patient_id}_{modality}.nii.gz")
                is_seg = (modality == 'seg')
                data, original = self.load_nifti(file_path, is_seg=is_seg)
                if is_seg:
                    seg_data = data
                    original_seg = original
                else:
                    patient_data.append(data)
                    original_modalities.append(original)
            modalities = np.stack(patient_data, axis=0).astype(np.float32)
            seg = seg_data
            sample = {
                'modalities': modalities,
                'seg': seg,
                'original_modalities': original_modalities,
                'original_seg': original_seg
            }
            if self.transform:
                sample = self.transform(sample)
            return sample
        else:
            raise ValueError("No valid data source available.")

    def compute_class_distribution(self):
        print("Computing class distribution...")
        class_counts = np.zeros(4)
        total_voxels = 0
        
        for idx in tqdm(range(len(self)), desc="Computing class distribution"):
            sample = self.__getitem__(idx)
            seg = sample['seg']
            seg_labels = np.argmax(seg, axis=0)
            remapped_labels = np.zeros_like(seg_labels, dtype=np.int32)
            remapped_labels[seg_labels == 0] = 0
            remapped_labels[seg_labels == 1] = 1
            remapped_labels[seg_labels == 2] = 2
            remapped_labels[seg_labels == 3] = 4
            
            for label in [0, 1, 2, 4]:
                class_idx = 0 if label == 0 else (1 if label == 1 else (2 if label == 2 else 3))
                class_counts[class_idx] += np.sum(remapped_labels == label)
            total_voxels += np.prod(seg_labels.shape)
        
        class_frequencies = class_counts / total_voxels
        print(f"Class frequencies - Background: {class_frequencies[0]:.4f}, Necrotic: {class_frequencies[1]:.4f}, Edema: {class_frequencies[2]:.4f}, Enhancing: {class_frequencies[3]:.4f}")
        
        inverse_freq = 1.0 / (class_frequencies + 1e-8)
        inverse_freq[0] = 0.0
        sqrt_inverse_freq = np.sqrt(inverse_freq)
        min_weight = np.min(sqrt_inverse_freq[1:])
        class_weights = sqrt_inverse_freq[1:] / min_weight
        print(f"Computed class weights - Necrotic: {class_weights[0]:.2f}, Edema: {class_weights[1]:.2f}, Enhancing: {class_weights[2]:.2f}")
        return class_weights
