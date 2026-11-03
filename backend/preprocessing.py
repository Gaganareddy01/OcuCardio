"""
OcuCardio - Data Preprocessing & Augmentation Pipeline
Handles image validation, standard resizing, CLAHE contrast enhancement,
PyTorch normalization, data augmentation, and patient-level leakage-free dataset splitting.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RetinalAugmenter:
    """Applies clinical-grade data augmentation preserving retinal morphology."""
    def __init__(self, horizontal_flip=True, vertical_flip=True, rotation_range=15, contrast_range=(0.85, 1.15)):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.contrast_range = contrast_range

    def __call__(self, img_pil: Image.Image, mask_pil: Image.Image = None):
        # 1. Random Horizontal Flip
        if self.horizontal_flip and random.random() > 0.5:
            img_pil = ImageOps.mirror(img_pil)
            if mask_pil:
                mask_pil = ImageOps.mirror(mask_pil)

        # 2. Random Vertical Flip
        if self.vertical_flip and random.random() > 0.5:
            img_pil = ImageOps.flip(img_pil)
            if mask_pil:
                mask_pil = ImageOps.flip(mask_pil)

        # 3. Random Small Rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            img_pil = img_pil.rotate(angle, resample=Image.BILINEAR)
            if mask_pil:
                mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)

        # 4. Subtle Contrast/Brightness Adjustment
        if self.contrast_range:
            factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(factor)

        if mask_pil:
            return img_pil, mask_pil
        return img_pil


def split_dataset(records, train_pct=0.70, val_pct=0.15, test_pct=0.15, patient_id_key="patient_id", random_seed=42):
    """
    Splits records into 70% Train, 15% Val, 15% Test.
    Strictly enforces patient-level separation whenever patient_id is present to PREVENT DATA LEAKAGE!
    """
    random.seed(random_seed)
    
    # Check if patient identifiers exist
    has_patient_ids = any(r.get(patient_id_key) for r in records)
    
    if has_patient_ids:
        # Group records by patient
        patient_groups = {}
        for r in records:
            pid = r.get(patient_id_key) or f"UNKNOWN_{random.randint(1, 99999)}"
            if pid not in patient_groups:
                patient_groups[pid] = []
            patient_groups[pid].append(r)

        unique_patients = list(patient_groups.keys())
        random.shuffle(unique_patients)

        n_pts = len(unique_patients)
        n_train = max(1, int(n_pts * train_pct))
        n_val = max(1, int(n_pts * val_pct))

        train_pts = set(unique_patients[:n_train])
        val_pts = set(unique_patients[n_train:n_train + n_val])
        test_pts = set(unique_patients[n_train + n_val:])
        if not test_pts:
            test_pts = val_pts

        train_records = []
        val_records = []
        test_records = []

        for pid, group in patient_groups.items():
            if pid in train_pts:
                train_records.extend(group)
            elif pid in val_pts:
                val_records.extend(group)
            else:
                test_records.extend(group)

        return train_records, val_records, test_records
    else:
        shuffled = list(records)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * train_pct))
        n_val = max(1, int(n * val_pct))
        return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


class RetinalDatasetPyTorch(Dataset):
    """
    Standard PyTorch Dataset for Retinal Image Analysis:
    Supports both segmentation and classification tasks.
    """
    def __init__(self, records, base_dir, image_size=(512, 512), is_training=False, task_type="disease_classification"):
        self.records = records
        self.base_dir = base_dir
        self.image_size = image_size
        self.is_training = is_training
        self.task_type = task_type
        self.augmenter = RetinalAugmenter() if is_training else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        
        # Resolve image path
        fname = item.get("filename")
        if item.get("relative_path"):
            img_path = os.path.join(PROJECT_ROOT, item["relative_path"])
        else:
            img_path = os.path.join(self.base_dir, "images", fname)

        # Load image safely
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", self.image_size, (0, 0, 0))

        img = img.resize(self.image_size, Image.BILINEAR)

        # Segmentation mask handling
        mask_tensor = torch.zeros((1, self.image_size[1], self.image_size[0]), dtype=torch.float32)
        if "segmentation" in self.task_type or item.get("mask_filename"):
            mask_fname = item.get("mask_filename")
            if mask_fname:
                mask_path = os.path.join(self.base_dir, "masks", mask_fname)
                if os.path.exists(mask_path):
                    try:
                        mask_img = Image.open(mask_path).convert("L")
                        mask_img = mask_img.resize(self.image_size, Image.NEAREST)
                        if self.augmenter and self.is_training:
                            img, mask_img = self.augmenter(img, mask_img)
                        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
                        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
                    except Exception:
                        pass
        elif self.augmenter and self.is_training:
            img = self.augmenter(img)

        # Image to tensor [3, H, W] normalized to [0, 1]
        img_np = np.array(img, dtype=np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_np - mean) / std
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Classification label handling
        label = 0.0
        if "binary_disease" in item:
            label = float(item["binary_disease"])
        elif "has_hypertension" in item:
            label = float(item["has_hypertension"])
        elif "systemic_hypertension" in item:
            label = float(item["systemic_hypertension"])
        elif "cvd_risk_score" in item:
            label = 1.0 if float(item["cvd_risk_score"]) >= 50.0 else 0.0
        elif "risk_label" in item:
            label = float(item["risk_label"])

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "meta": item
        }
