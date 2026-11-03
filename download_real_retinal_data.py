import os
import urllib.request
import ssl
import json
import numpy as np
from PIL import Image

def download_file(url, out_path):
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        content = resp.read()
    with open(out_path, 'wb') as f:
        f.write(content)

def collect_ddr_samples(num_samples=16, base_dir="data/real_fundus"):
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "masks"), exist_ok=True)

    print(f"[*] Querying DDR dataset on Hugging Face...")
    ctx = ssl._create_unverified_context()
    api_url = "https://huggingface.co/api/datasets/Achyuth12/DDR-dataset"
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
        meta = json.loads(resp.read().decode())

    # Find matching image and MA annotation pairs
    all_files = [s['rfilename'] for s in meta.get('siblings', [])]
    ma_files = [f for f in all_files if f.startswith('lesion_segmentation/annotations/tet/MA/')]
    
    selected_samples = []
    for ma_path in ma_files:
        filename = os.path.basename(ma_path)
        base_name, _ = os.path.splitext(filename)
        img_path = f"lesion_segmentation/images/test/{base_name}.jpg"
        if img_path in all_files:
            selected_samples.append((base_name, img_path, ma_path))
        if len(selected_samples) >= num_samples:
            break

    print(f"[*] Found {len(selected_samples)} valid real patient fundus samples with expert annotations.")

    meta_records = []
    hf_base = "https://huggingface.co/datasets/Achyuth12/DDR-dataset/resolve/main"

    for idx, (base_name, img_subpath, ma_subpath) in enumerate(selected_samples):
        print(f"[{idx+1}/{len(selected_samples)}] Downloading real clinical fundus: {base_name}...")
        img_local = os.path.join(base_dir, "images", f"{base_name}.jpg")
        mask_local = os.path.join(base_dir, "masks", f"{base_name}_mask.npy")

        # 1. Download Fundus Image
        if not os.path.exists(img_local):
            download_file(f"{hf_base}/{img_subpath}", img_local)

        # 2. Download MA mask
        ma_temp = os.path.join(base_dir, "masks", f"{base_name}_ma.tif")
        if not os.path.exists(ma_temp):
            download_file(f"{hf_base}/{ma_subpath}", ma_temp)

        # 3. Download Hemorrhage / Neovascularization surrogate mask
        he_subpath = f"lesion_segmentation/annotations/tet/HE/{base_name}.tif"
        he_temp = os.path.join(base_dir, "masks", f"{base_name}_he.tif")
        if he_subpath in all_files and not os.path.exists(he_temp):
            try:
                download_file(f"{hf_base}/{he_subpath}", he_temp)
            except Exception:
                pass

        # Open image and masks to construct unified 4-channel ground truth
        img = Image.open(img_local).convert('RGB')
        w, h = img.size
        target_size = (512, 512)
        img_resized = img.resize(target_size, Image.BILINEAR)
        img_resized.save(img_local)

        # Build 4-channel mask:
        # Ch 0: Optic Disc (approx from bright intensity region)
        # Ch 1: Real Microaneurysms (from DDR MA expert mask)
        # Ch 2: Real Hemorrhages / Neovascular lesions (from DDR HE expert mask)
        # Ch 3: Retinal Vasculature
        masks_4ch = np.zeros((4, 512, 512), dtype=np.float32)

        # Channel 1: Real Microaneurysms
        if os.path.exists(ma_temp):
            ma_img = Image.open(ma_temp).resize(target_size, Image.NEAREST)
            ma_np = (np.array(ma_img) > 0).astype(np.float32)
            masks_4ch[1] = ma_np

        # Channel 2: Real Neovascularization / Hemorrhages
        if os.path.exists(he_temp):
            he_img = Image.open(he_temp).resize(target_size, Image.NEAREST)
            he_np = (np.array(he_img) > 0).astype(np.float32)
            masks_4ch[2] = he_np

        # Channel 0: Optic Disc (localized via bright disc profile on red channel)
        img_np = np.array(img_resized)
        red_channel = img_np[:, :, 0].astype(np.float32)
        # Optic disc is the brightest circular region
        threshold = np.percentile(red_channel, 98)
        od_approx = (red_channel >= threshold).astype(np.float32)
        masks_4ch[0] = od_approx

        # Channel 3: Vessels (high contrast green channel threshold)
        green_channel = img_np[:, :, 1].astype(np.float32)
        vessel_approx = (green_channel < np.percentile(green_channel, 25)).astype(np.float32)
        masks_4ch[3] = vessel_approx

        # Save unified 4-channel mask array
        np.save(mask_local, masks_4ch)

        # Cleanup raw single-lesion tifs
        if os.path.exists(ma_temp):
            os.remove(ma_temp)
        if os.path.exists(he_temp):
            os.remove(he_temp)

        # Risk score: High risk (1.0) if MA or lesion burden is extensive
        ma_count = np.sum(masks_4ch[1])
        he_count = np.sum(masks_4ch[2])
        cvd_risk = 1.0 if (ma_count > 10 or he_count > 50) else 0.0

        meta_records.append({
            'base_name': base_name,
            'image_path': img_local,
            'mask_path': mask_local,
            'risk_label': cvd_risk,
            'ma_pixels': int(ma_count),
            'he_pixels': int(he_count)
        })

    # Save metadata JSON
    meta_path = os.path.join(base_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_records, f, indent=2)

    print(f"\n[+] Successfully collected {len(meta_records)} real clinical fundus images & masks in {base_dir}!")
    return meta_records

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=40, help="Number of real clinical fundus samples to download")
    parser.add_argument('--base-dir', type=str, default='data/real_fundus', help="Output directory")
    args = parser.parse_args()
    collect_ddr_samples(num_samples=args.num_samples, base_dir=args.base_dir)
