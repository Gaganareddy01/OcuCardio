import os
import json
import cv2
import numpy as np
from PIL import Image

def build_crao_case(
    source_img_path: str = "data/real_fundus/images/crao_cleaned_patient.jpg",
    base_dir: str = "data/real_fundus",
    num_augmented: int = 8
):
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "masks"), exist_ok=True)

    img = Image.open(source_img_path).convert('RGB')
    target_size = (512, 512)
    img_512 = img.resize(target_size, Image.BILINEAR)
    img_np = np.array(img_512)

    H, W = target_size

    # Build 4-channel ground truth mask for CRAO
    # Ch 0: Optic Disc
    # Ch 1: Cherry-Red Spot (focal dark red foveal island)
    # Ch 2: Retinal Pallor / Ischemic Edema (milky white posterior pole)
    # Ch 3: Attenuated Vasculature
    masks_4ch = np.zeros((4, H, W), dtype=np.float32)

    # 1. Optic Disc (Ch 0):
    # Located on nasal side, roughly x ~ 120, y ~ 260 in 512x512 coordinate space
    od_cx, od_cy, od_r = 125, 260, 32
    y_grid, x_grid = np.ogrid[:H, :W]
    od_mask = ((x_grid - od_cx)**2 + (y_grid - od_cy)**2) <= od_r**2
    masks_4ch[0, od_mask] = 1.0

    # 2. Cherry-Red Spot (Ch 1):
    # Fovea center is located at roughly x ~ 275, y ~ 305
    crs_cx, crs_cy, crs_r = 275, 305, 14
    crs_mask = ((x_grid - crs_cx)**2 + (y_grid - crs_cy)**2) <= crs_r**2
    masks_4ch[1, crs_mask] = 1.0

    # 3. Retinal Pallor / Ischemic Infarction Edema (Ch 2):
    # Surrounding the cherry-red spot across the temporal macula and arcades
    pallor_cx, pallor_cy, pallor_rx, pallor_ry = 280, 290, 110, 80
    pallor_mask = (((x_grid - pallor_cx)**2 / (pallor_rx**2)) + ((y_grid - pallor_cy)**2 / (pallor_ry**2))) <= 1.0
    # The cherry-red spot itself is not pale (it retains choroidal color)
    pallor_mask[crs_mask] = False
    masks_4ch[2, pallor_mask] = 1.0

    # 4. Attenuated Vasculature (Ch 3):
    # Thin narrowed arterioles extracted via green-channel thresholding
    green = img_np[:, :, 1].astype(np.float32)
    vessels = (green < np.percentile(green, 20)).astype(np.float32)
    masks_4ch[3] = vessels

    # Load existing metadata
    meta_path = os.path.join(base_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []

    # Save Primary CRAO patient
    primary_id = "patient_crao_01"
    p_img_path = os.path.join(base_dir, "images", f"{primary_id}.jpg")
    p_mask_path = os.path.join(base_dir, "masks", f"{primary_id}_mask.npy")
    img_512.save(p_img_path)
    np.save(p_mask_path, masks_4ch)

    metadata.append({
        'base_name': primary_id,
        'image_path': p_img_path,
        'mask_path': p_mask_path,
        'risk_label': 1.0,  # Maximum Systemic Vascular & Acute Stroke Risk
        'diagnosis': 'Central Retinal Artery Occlusion (CRAO)',
        'ma_pixels': int(np.sum(masks_4ch[1])),
        'he_pixels': int(np.sum(masks_4ch[2]))
    })

    # Generate clinical variations (rotation, contrast, illumination changes) to form a robust CRAO cohort
    print(f"[*] Generating {num_augmented} augmented clinical CRAO cases...")
    for i in range(1, num_augmented + 1):
        aug_id = f"patient_crao_aug_{i:02d}"
        aug_img_path = os.path.join(base_dir, "images", f"{aug_id}.jpg")
        aug_mask_path = os.path.join(base_dir, "masks", f"{aug_id}_mask.npy")

        # Slight affine / photometric augmentation
        angle = np.random.uniform(-15, 15)
        scale = np.random.uniform(0.95, 1.05)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)

        aug_cv = cv2.warpAffine(img_np, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Photometric variation
        gamma = np.random.uniform(0.85, 1.15)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        aug_cv = cv2.LUT(aug_cv, table)

        # Warp masks accordingly
        aug_masks = np.zeros_like(masks_4ch)
        for c in range(4):
            aug_masks[c] = cv2.warpAffine(masks_4ch[c], M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Save
        cv2.imwrite(aug_img_path, cv2.cvtColor(aug_cv, cv2.COLOR_RGB2BGR))
        np.save(aug_mask_path, aug_masks)

        metadata.append({
            'base_name': aug_id,
            'image_path': aug_img_path,
            'mask_path': aug_mask_path,
            'risk_label': 1.0,
            'diagnosis': 'Central Retinal Artery Occlusion (CRAO) Variant',
            'ma_pixels': int(np.sum(aug_masks[1])),
            'he_pixels': int(np.sum(aug_masks[2]))
        })

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Successfully integrated CRAO / Acute Ocular Stroke cohort! Total dataset size: {len(metadata)} patient records.")

if __name__ == '__main__':
    build_crao_case()
