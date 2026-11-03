import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.multitask_resunet import RetinalMultiTaskResUNet
from models.losses import MultiTaskRetinalLoss
from dataset import SyntheticRetinalDataset, RetinalFundusDataset

def compute_dice_score(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """
    Computes per-class Dice Score for evaluation.
    preds: [B, C, H, W] (raw logits)
    targets: [B, C, H, W] (ground truth binary masks)
    """
    probs = torch.sigmoid(preds)
    bin_preds = (probs > threshold).float()

    bin_preds = bin_preds.view(bin_preds.size(0), bin_preds.size(1), -1)
    targets = targets.view(targets.size(0), targets.size(1), -1)

    intersection = (bin_preds * targets).sum(dim=2)
    cardinality = bin_preds.sum(dim=2) + targets.sum(dim=2)

    dice = (2.0 * intersection + eps) / (cardinality + eps)
    # Average across batch, return per-class score
    return dice.mean(dim=0).cpu().numpy()


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    seg_loss_sum = 0.0
    cls_loss_sum = 0.0

    for batch_idx, (images, masks, risks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        risks = risks.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            seg_logits, risk_logits = model(images)
            loss, loss_details = criterion(seg_logits, masks, risk_logits, risks)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss_details['total_loss']
        seg_loss_sum += loss_details['seg_loss']
        cls_loss_sum += loss_details['cls_loss']

    num_batches = len(dataloader)
    return {
        'total_loss': total_loss / num_batches,
        'seg_loss': seg_loss_sum / num_batches,
        'cls_loss': cls_loss_sum / num_batches
    }


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_dice = []
    correct_cls = 0
    total_cls = 0

    class_names = ['Optic Disc / Edema', 'Microaneurysms (MA)', 'Neovascularization (NV)', 'Vasculature']

    with torch.no_grad():
        for images, masks, risks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            risks = risks.to(device)

            seg_logits, risk_logits = model(images)
            _, loss_details = criterion(seg_logits, masks, risk_logits, risks)
            total_loss += loss_details['total_loss']

            # Compute Dice
            batch_dice = compute_dice_score(seg_logits, masks)
            all_dice.append(batch_dice)

            # Compute Classification Accuracy
            probs = torch.sigmoid(risk_logits).view(-1)
            preds = (probs > 0.5).float()
            target_risks = risks.view(-1).float()
            correct_cls += (preds == target_risks).sum().item()
            total_cls += target_risks.numel()

    avg_dice_per_class = torch.from_numpy(np.array(all_dice)).mean(dim=0)
    avg_dice = avg_dice_per_class.mean().item()
    cls_acc = (correct_cls / total_cls) if total_cls > 0 else 0.0

    return {
        'val_loss': total_loss / len(dataloader),
        'mean_dice': avg_dice,
        'per_class_dice': dict(zip(class_names, avg_dice_per_class.tolist())),
        'classification_accuracy': cls_acc
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Retinal CVD Risk and Lesion Segmentation Model")
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet34'], help="ResNet backbone")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--image-size', type=int, default=512, help="Input fundus image resolution (e.g. 512 for 512x512)")
    parser.add_argument('--demo', action='store_true', default=False, help="Run in demo mode using synthetic retinal dataset")
    parser.add_argument('--data-dir', type=str, default='data/real_fundus', help="Directory containing real retinal fundus dataset")
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running on device: {device}")

    # Initialize model
    # 4 segmentation channels: Optic Disc/Edema, Microaneurysms, Neovascularization, Vasculature
    # 1 CVD risk classification logit
    model = RetinalMultiTaskResUNet(
        num_seg_classes=4,
        num_risk_classes=1,
        backbone=args.backbone,
        pretrained=True
    ).to(device)

    print(f"[*] Instantiated Multi-Task ResUNet with {args.backbone.upper()} backbone.")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable parameters: {total_params:,}")

    # Loss and Optimizer
    criterion = MultiTaskRetinalLoss(
        seg_dice_weight=0.5,
        seg_focal_weight=0.5,
        task_seg_weight=1.0,
        task_cls_weight=1.0,
        cls_loss_type='bce'
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Load Data
    meta_file = os.path.join(args.data_dir, "metadata.json") if args.data_dir else ""
    if not args.demo and os.path.exists(meta_file):
        print(f"[*] Loading REAL Clinical Retinal Fundus Dataset from {args.data_dir}...")
        with open(meta_file, 'r') as f:
            records = json.load(f)
        img_paths = [r['image_path'] for r in records]
        mask_paths = [r['mask_path'] for r in records]
        risk_labels = [r['risk_label'] for r in records]

        # 80/20 train/val split
        split_idx = max(1, int(len(img_paths) * 0.75))
        train_dataset = RetinalFundusDataset(
            image_paths=img_paths[:split_idx],
            mask_paths=mask_paths[:split_idx],
            risk_labels=risk_labels[:split_idx],
            image_size=(args.image_size, args.image_size)
        )
        val_dataset = RetinalFundusDataset(
            image_paths=img_paths[split_idx:],
            mask_paths=mask_paths[split_idx:],
            risk_labels=risk_labels[split_idx:],
            image_size=(args.image_size, args.image_size)
        )
        print(f"[*] Loaded {len(train_dataset)} training and {len(val_dataset)} validation REAL clinical patient fundus images.")
    else:
        print("[*] Generating synthetic retinal dataset (simulating optic disc edema, MAs, and neovascularization)...")
        train_dataset = SyntheticRetinalDataset(num_samples=20, image_size=(args.image_size, args.image_size))
        val_dataset = SyntheticRetinalDataset(num_samples=10, image_size=(args.image_size, args.image_size))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[*] Training for {args.epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"\n--- Epoch [{epoch}/{args.epochs}] (Time: {elapsed:.2f}s) ---")
        print(f"  Train Total Loss: {train_metrics['total_loss']:.4f} | Seg Loss: {train_metrics['seg_loss']:.4f} | Cls Loss: {train_metrics['cls_loss']:.4f}")
        print(f"  Val Total Loss:   {val_metrics['val_loss']:.4f} | Mean Dice: {val_metrics['mean_dice']:.4f} | Cls Acc: {val_metrics['classification_accuracy']*100:.1f}%")
        for k, v in val_metrics['per_class_dice'].items():
            print(f"    - Dice [{k}]: {v:.4f}")

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_path = os.path.join(args.save_dir, 'best_retinal_resunet.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  [*] Best checkpoint saved to {save_path}")

    print("\n[+] Training routine completed successfully.")

if __name__ == '__main__':
    main()
