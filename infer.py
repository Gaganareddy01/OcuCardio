import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from models.multitask_resunet import RetinalMultiTaskResUNet

def run_retinal_inference(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_retinal_resunet.pth",
    backbone: str = "resnet34",
    image_size: tuple = (256, 256),
    output_dir: str = "outputs"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference on device: {device}")

    # 1. Load trained model
    model = RetinalMultiTaskResUNet(
        num_seg_classes=4,
        num_risk_classes=1,
        backbone=backbone,
        pretrained=False
    ).to(device)

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[*] Loaded trained checkpoint: {checkpoint_path}")
    else:
        print(f"[!] Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model.eval()

    # 2. Preprocess input fundus image
    raw_img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    img_resized = raw_img.resize(image_size, Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 3. Model Forward Pass
    with torch.no_grad():
        seg_logits, risk_logits = model(img_tensor)

        # Lesion probabilities
        seg_probs = torch.sigmoid(seg_logits).squeeze(0).cpu().numpy()  # [4, H, W]
        # CVD Risk probability
        risk_prob = torch.sigmoid(risk_logits).item() * 100.0

    # 4. Extract Lesions
    # Ch 0: Optic Disc & Edema
    # Ch 1: Microaneurysms
    # Ch 2: Neovascularization
    # Ch 3: Vasculature
    disc_pred = (seg_probs[0] > 0.35).astype(np.float32)
    ma_pred = (seg_probs[1] > 0.15).astype(np.float32)
    nv_pred = (seg_probs[2] > 0.20).astype(np.float32)
    vessel_pred = (seg_probs[3] > 0.30).astype(np.float32)

    # Quantitative Biomarkers
    ma_count = int(np.sum(ma_pred))
    nv_area = int(np.sum(nv_pred))
    disc_area = int(np.sum(disc_pred))

    risk_tier = "HIGH RISK" if risk_prob >= 60 else ("MODERATE RISK" if risk_prob >= 35 else "LOW RISK")
    color = "red" if risk_prob >= 60 else ("orange" if risk_prob >= 35 else "green")

    print("\n" + "="*50)
    print("      CLINICAL RETINAL CVD RISK ASSESSMENT      ")
    print("="*50)
    print(f"Image Input:                {os.path.basename(image_path)}")
    print(f"Systemic CVD Risk Score:    {risk_prob:.1f}% ({risk_tier})")
    print(f"Microaneurysm (MA) Burden:  {ma_count} pixels")
    print(f"Neovascularization Extent:  {nv_area} pixels")
    print(f"Optic Disc Boundary Extent: {disc_area} pixels")
    print("="*50)

    # 5. Composite Visual Diagnostic Report
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Input Fundus Image\n(RGB Clinical Scan)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(img_np)
    axes[1].imshow(disc_pred, cmap='autumn', alpha=0.55)
    axes[1].set_title(f"Optic Disc & Edema\nArea: {disc_area}px", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(img_np)
    axes[2].imshow(ma_pred, cmap='Reds', alpha=0.7)
    axes[2].set_title(f"Microaneurysms (MA)\nCount: {ma_count}px", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(img_np)
    axes[3].imshow(nv_pred, cmap='cool', alpha=0.6)
    axes[3].set_title(f"Neovascularization\nArea: {nv_area}px", fontsize=11, fontweight='bold')
    axes[3].axis('off')

    axes[4].imshow(vessel_pred, cmap='gray')
    axes[4].set_title("Retinal Vessel Bed\n(Caliber / Arcade)", fontsize=11, fontweight='bold')
    axes[4].axis('off')

    plt.suptitle(
        f"AI Retinal Analysis for Systemic Vascular Risk | CVD Score: {risk_prob:.1f}% [{risk_tier}]",
        fontsize=14, fontweight='bold', color=color, y=1.02
    )

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"report_{base_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()

    print(f"[+] Diagnostic visual report saved to: {out_path}\n")
    return out_path, risk_prob, risk_tier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='data/real_fundus/images/007-1853-100.jpg')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_retinal_resunet.pth')
    parser.add_argument('--backbone', type=str, default='resnet34')
    args = parser.parse_args()

    run_retinal_inference(args.image, args.checkpoint, args.backbone)
