import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

class RetinalFundusDataset(Dataset):
    """
    Dataset for Multi-Task Retinal Image Analysis:
    - Fundus Images (RGB)
    - Multi-Channel Lesion Masks:
        * Channel 0: Optic Disc & Edema
        * Channel 1: Microaneurysms (MA)
        * Channel 2: Neovascularization (NV)
        * Channel 3: Retinal Vasculature
    - Systemic Vascular / CVD Risk Target (0: Low Risk, 1: High Risk / Malignant Hypertension / CAD)
    """
    def __init__(
        self,
        image_paths: list,
        mask_paths: list = None,
        risk_labels: list = None,
        image_size: tuple = (512, 512),
        transform = None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.risk_labels = risk_labels
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)
        img_np = np.array(image, dtype=np.float32) / 255.0
        # Transpose to [C, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # Load or generate masks
        if self.mask_paths is not None and len(self.mask_paths) > idx:
            mask_path = self.mask_paths[idx]
            if mask_path.endswith('.npy'):
                mask_np = np.load(mask_path).astype(np.float32)
                mask_t = torch.from_numpy(mask_np)
                if mask_t.shape[1:] != self.image_size:
                    mask_tensor = F.interpolate(mask_t.unsqueeze(0), size=self.image_size, mode='nearest').squeeze(0)
                else:
                    mask_tensor = mask_t
            else:
                m_img = Image.open(mask_path).resize(self.image_size, Image.NEAREST)
                m_np = np.array(m_img, dtype=np.float32)
                if m_np.ndim == 2:
                    m_np = np.expand_dims(m_np, axis=0)
                else:
                    m_np = np.transpose(m_np, (2, 0, 1))
                mask_np = (m_np > 127).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_np).float()
        else:
            # Default empty 4-channel mask if training classification-only or unlabeled mask
            mask_tensor = torch.zeros((4, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        # Risk label
        if self.risk_labels is not None:
            risk = torch.tensor(self.risk_labels[idx], dtype=torch.float32)
        else:
            risk = torch.tensor(0.0, dtype=torch.float32)

        return img_tensor, mask_tensor, risk


class SyntheticRetinalDataset(Dataset):
    """
    Synthetic Retinal Dataset generator for rapid prototyping, pipeline verification,
    and sanity-checking shapes and gradients without needing immediate raw medical files.
    Generates realistic circular fundus discs with synthetic vessels and lesions.
    """
    def __init__(self, num_samples: int = 50, image_size: tuple = (512, 512)):
        self.num_samples = num_samples
        self.image_size = image_size
        H, W = image_size

        # Precompute coordinate grids for circular fundus mask
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H // 2, W // 2
        radius = min(H, W) // 2 - 10
        self.fundus_fov_mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        H, W = self.image_size

        # Base fundus background: reddish-orange with green variation
        img = np.zeros((3, H, W), dtype=np.float32)
        img[0, :, :] = 0.75 + np.random.uniform(-0.05, 0.05, (H, W)) # Red
        img[1, :, :] = 0.40 + np.random.uniform(-0.05, 0.05, (H, W)) # Green
        img[2, :, :] = 0.15 + np.random.uniform(-0.03, 0.03, (H, W)) # Blue

        # Mask out black non-fundus area
        for c in range(3):
            img[c][~self.fundus_fov_mask] = 0.0

        # Multi-channel ground truth mask:
        # Ch 0: Optic disc / edema (bright circular spot on nasal side)
        # Ch 1: Microaneurysms (scattered punctate spots)
        # Ch 2: Neovascularization (frond-like branching clusters)
        # Ch 3: Retinal vasculature (main vascular arcade)
        masks = np.zeros((4, H, W), dtype=np.float32)

        # 1. Optic Disc (Ch 0)
        od_cy, od_cx = H // 2, W // 3
        od_r = 35
        y, x = np.ogrid[:H, :W]
        od_mask = ((x - od_cx)**2 + (y - od_cy)**2) <= od_r**2
        masks[0, od_mask] = 1.0

        # 2. Microaneurysms (Ch 1) - random tiny 2x2 spots
        num_mas = np.random.randint(5, 20)
        for _ in range(num_mas):
            ma_y = np.random.randint(H // 4, 3 * H // 4)
            ma_x = np.random.randint(W // 4, 3 * W // 4)
            if self.fundus_fov_mask[ma_y, ma_x]:
                masks[1, ma_y-1:ma_y+2, ma_x-1:ma_x+2] = 1.0

        # 3. Neovascularization (Ch 2) - small cluster near disc or periphery
        nv_y = np.random.randint(H // 3, 2 * H // 3)
        nv_x = np.random.randint(W // 3, 2 * W // 3)
        masks[2, nv_y:nv_y+8, nv_x:nv_x+8] = 1.0

        # 4. Vasculature (Ch 3) - simulated arcade lines
        masks[3, H // 2 - 2: H // 2 + 2, :] = 1.0
        masks[3][~self.fundus_fov_mask] = 0.0

        # Simulated CVD Risk Label: high if lesion density is high
        high_risk = 1.0 if (num_mas > 12) else 0.0

        return torch.tensor(img, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32), torch.tensor([high_risk], dtype=torch.float32)
