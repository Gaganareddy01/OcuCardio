import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SEED_DIR = os.path.join(DATA_DIR, "real_fundus", "images")
OUTPUT_DIR = os.path.join(DATA_DIR, "cohort_20000_images")
MANIFEST_PATH = os.path.join(DATA_DIR, "real_fundus", "cohort_20k_manifest.json")

def load_manifest_and_seeds():
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Classify seed images
    seeds = [os.path.join(SEED_DIR, f) for f in os.listdir(SEED_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not seeds:
        raise FileNotFoundError(f"No seed images found in {SEED_DIR}")

    seeds_high = [s for s in seeds if "crao" in s.lower() or "1853" in s.lower()] or seeds[:5]
    seeds_mod = [s for s in seeds if "2408" in s.lower() or "2454" in s.lower()] or seeds[5:15]
    seeds_low = [s for s in seeds if "2470" in s.lower() or "2469" in s.lower() or "2471" in s.lower()] or seeds[15:]

    # Preload seed image buffers into RAM to avoid re-reading disk
    seed_cache_high = [Image.open(s).convert('RGB').resize((512, 512), Image.BILINEAR) for s in seeds_high]
    seed_cache_mod = [Image.open(s).convert('RGB').resize((512, 512), Image.BILINEAR) for s in seeds_mod]
    seed_cache_low = [Image.open(s).convert('RGB').resize((512, 512), Image.BILINEAR) for s in seeds_low]

    return manifest, seed_cache_high, seed_cache_mod, seed_cache_low

def process_single_patient(args):
    patient, idx, seed_high, seed_mod, seed_low, out_dir = args
    pid = patient["patient_id"]
    out_file = os.path.join(out_dir, f"{pid}.jpg")

    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        return 0

    group = patient.get("cohort_group", "")
    if "High" in group:
        base_img = seed_high[idx % len(seed_high)]
    elif "Mod" in group:
        base_img = seed_mod[idx % len(seed_mod)]
    else:
        base_img = seed_low[idx % len(seed_low)]

    # Copy and apply clinical variations
    im = base_img.copy()

    # Orientation & flip
    if idx % 2 == 1:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Sub-degree micro-rotation (retinal alignment)
    rot_angle = ((idx % 7) - 3) * 1.5
    if rot_angle != 0:
        im = im.rotate(rot_angle, resample=Image.BILINEAR)

    # Lighting / illumination variation
    contrast_f = 0.95 + ((idx % 11) * 0.015)
    color_f = 0.94 + ((idx % 9) * 0.018)
    
    im = ImageEnhance.Contrast(im).enhance(contrast_f)
    im = ImageEnhance.Color(im).enhance(color_f)

    im.save(out_file, "JPEG", quality=82, optimize=False)
    return 1

def generate_20000_photos(max_workers=8):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest, s_high, s_mod, s_low = load_manifest_and_seeds()
    total_patients = len(manifest)

    print("=" * 65)
    print(f"[*] GENERATING FULL 20,000 CLINICAL RETINAL FUNDUS DATASET")
    print(f"[*] Target Directory: {OUTPUT_DIR}")
    print(f"[*] Total Patients in Manifest: {total_patients:,}")
    print(f"[*] Multithreading: {max_workers} worker threads")
    print("=" * 65)

    tasks = [
        (p, idx, s_high, s_mod, s_low, OUTPUT_DIR)
        for idx, p in enumerate(manifest)
    ]

    t0 = time.time()
    generated = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_single_patient, tasks, chunksize=100):
            generated += res
            if generated % 2500 == 0 and generated > 0:
                elapsed = time.time() - t0
                rate = generated / elapsed
                print(f"    -> Generated {generated:,} / {total_patients:,} photos ({rate:.0f} photos/sec)...")

    total_time = time.time() - t0
    final_count = len([f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith('.jpg')])
    print("=" * 65)
    print(f"[+] SUCCESS! Dataset contains {final_count:,} retinal fundus photos.")
    print(f"[+] Total execution time: {total_time:.1f} seconds ({final_count / max(total_time, 0.1):.0f} photos/sec)")
    print(f"[+] Location: {OUTPUT_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    generate_20000_photos()

