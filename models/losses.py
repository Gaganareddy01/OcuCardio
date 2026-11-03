import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-label / multi-class retinal lesion segmentation.
    Operates per-channel to ensure rare lesions (microaneurysms, neovascularization)
    are not swamped by larger structures or background.
    """
    def __init__(self, smooth: float = 1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        targets: [B, C, H, W] with binary values {0, 1}
        """
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (probs * targets).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # 1 - mean Dice across batch and classes
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe foreground/background class imbalance
    (e.g., microaneurysms occupying <0.1% of retinal image pixels).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_factor * torch.pow((1.0 - p_t), self.gamma)
        
        return (focal_weight * bce).mean()


class MultiTaskRetinalLoss(nn.Module):
    """
    Combined Multi-Task Loss for:
    1. Retinal Lesion Segmentation (Dice + Focal)
    2. Systemic Vascular / CVD Risk Classification (BCE or Cross-Entropy)
    """
    def __init__(
        self,
        seg_dice_weight: float = 0.5,
        seg_focal_weight: float = 0.5,
        task_seg_weight: float = 1.0,
        task_cls_weight: float = 1.0,
        cls_loss_type: str = 'bce'  # 'bce' for binary/continuous risk, 'ce' for multi-class
    ):
        super(MultiTaskRetinalLoss, self).__init__()
        self.dice_loss = SoftDiceLoss()
        self.focal_loss = FocalLoss(alpha=0.75, gamma=2.0)
        self.seg_dice_weight = seg_dice_weight
        self.seg_focal_weight = seg_focal_weight
        self.task_seg_weight = task_seg_weight
        self.task_cls_weight = task_cls_weight
        self.cls_loss_type = cls_loss_type

        if cls_loss_type == 'bce':
            self.cls_criterion = nn.BCEWithLogitsLoss()
        else:
            self.cls_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        risk_logits: torch.Tensor,
        risk_targets: torch.Tensor
    ):
        """
        Returns:
            total_loss: scalar tensor for backward()
            loss_dict: dictionary of individual loss components for logging
        """
        # 1. Segmentation Loss
        l_dice = self.dice_loss(seg_logits, seg_targets)
        l_focal = self.focal_loss(seg_logits, seg_targets)
        l_seg = (self.seg_dice_weight * l_dice) + (self.seg_focal_weight * l_focal)

        # 2. Risk Classification Loss
        if self.cls_loss_type == 'bce':
            # Ensure risk_targets has same shape as risk_logits
            if risk_targets.ndim == 1:
                risk_targets = risk_targets.unsqueeze(1).float()
            else:
                risk_targets = risk_targets.float()
            l_cls = self.cls_criterion(risk_logits, risk_targets)
        else:
            l_cls = self.cls_criterion(risk_logits, risk_targets.long())

        # Total combined weighted loss
        total_loss = (self.task_seg_weight * l_seg) + (self.task_cls_weight * l_cls)

        return total_loss, {
            'total_loss': total_loss.item(),
            'seg_loss': l_seg.item(),
            'seg_dice': l_dice.item(),
            'seg_focal': l_focal.item(),
            'cls_loss': l_cls.item()
        }
