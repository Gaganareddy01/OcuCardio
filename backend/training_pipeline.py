"""
OcuCardio - AI Model Training Pipeline
Provides asynchronous background training for Multi-Task ResUNet, ResNet-50 Disease Classifier,
and Retinal Vessel Segmentation models with live progress monitoring, pausing, and stopping.
"""

import os
import sys
import time
import json
import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.multitask_resunet import RetinalMultiTaskResUNet
from models.losses import MultiTaskRetinalLoss
from backend.preprocessing import split_dataset, RetinalDatasetPyTorch
from backend.dataset_hub import DatasetHub

CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

class TrainingPipeline:
    def __init__(self):
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        self.dataset_hub = DatasetHub()
        
        # Thread & Control Flags
        self._thread = None
        self._pause_event = threading.Event()
        self._pause_event.set() # Unpaused by default
        self._stop_requested = False
        
        # Live Training State
        self.state = {
            "status": "idle", # "idle", "training", "paused", "completed", "stopped", "error"
            "model_name": "",
            "architecture": "multitask_resunet",
            "dataset_id": "drive",
            "task_type": "vessel_segmentation",
            "current_epoch": 0,
            "total_epochs": 5,
            "current_step": 0,
            "total_steps": 0,
            "progress_pct": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "train_loss_history": [],
            "val_loss_history": [],
            "metric_name": "Dice Score",
            "val_metric_history": [],
            "device": "cpu",
            "start_time": None,
            "elapsed_seconds": 0,
            "error_message": "",
            "saved_model_path": "",
            "logs": []
        }

    def log(self, message):
        """Appends timestamped log message."""
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.state["logs"].append(entry)
        if len(self.state["logs"]) > 200:
            self.state["logs"].pop(0)
        print(entry)

    def get_status(self):
        """Returns the current training state."""
        if self.state["status"] == "training" and self.state["start_time"]:
            self.state["elapsed_seconds"] = int(time.time() - self.state["start_time"])
        return dict(self.state)

    def pause_training(self):
        """Pauses training."""
        if self.state["status"] == "training":
            self._pause_event.clear()
            self.state["status"] = "paused"
            self.log("Training paused by user.")
            return {"status": "paused"}
        return {"status": self.state["status"]}

    def resume_training(self):
        """Resumes paused training."""
        if self.state["status"] == "paused":
            self._pause_event.set()
            self.state["status"] = "training"
            self.log("Training resumed.")
            return {"status": "training"}
        return {"status": self.state["status"]}

    def stop_training(self):
        """Stops training gracefully."""
        if self.state["status"] in ["training", "paused"]:
            self._stop_requested = True
            self._pause_event.set() # Release if paused
            self.state["status"] = "stopped"
            self.log("Training abort requested by user.")
            return {"status": "stopped"}
        return {"status": self.state["status"]}

    def start_training(self, config):
        """Starts asynchronous training loop with user configuration."""
        if self.state["status"] == "training":
            return {"status": "error", "error": "A training process is already running."}

        # Reset state
        self._stop_requested = False
        self._pause_event.set()
        
        self.state["status"] = "training"
        self.state["model_name"] = config.get("model_name", f"Model_{int(time.time())}")
        self.state["architecture"] = config.get("architecture", "multitask_resunet")
        self.state["dataset_id"] = config.get("dataset_id", "drive")
        self.state["total_epochs"] = int(config.get("epochs", 5))
        self.state["current_epoch"] = 0
        self.state["current_step"] = 0
        self.state["progress_pct"] = 0.0
        self.state["train_loss_history"] = []
        self.state["val_loss_history"] = []
        self.state["val_metric_history"] = []
        self.state["logs"] = []
        self.state["error_message"] = ""
        self.state["start_time"] = time.time()
        self.state["elapsed_seconds"] = 0

        # Launch thread
        self._thread = threading.Thread(target=self._run_training_loop, args=(config,), daemon=True)
        self._thread.start()
        
        return {"status": "started", "model_name": self.state["model_name"]}

    def _run_training_loop(self, config):
        """Internal training thread."""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.state["device"] = str(device)
            self.log(f"Starting OcuCardio training on compute device: {device}")

            # 1. Load Dataset Records
            dataset_id = config.get("dataset_id", "drive")
            dataset_info = self.dataset_hub.get_dataset_by_id(dataset_id)
            if not dataset_info:
                raise ValueError(f"Dataset {dataset_id} not found.")

            task_type = dataset_info.get("task_type", "disease_classification")
            self.state["task_type"] = task_type
            self.log(f"Selected Dataset: {dataset_info['name']} | Task: {task_type}")

            # Load raw items
            preview = self.dataset_hub.preview_dataset_images(dataset_id, page=1, page_size=20000)
            raw_records = preview.get("items", [])
            if not raw_records:
                raise ValueError(f"No valid images found in dataset {dataset_id}")

            self.log(f"Loaded {len(raw_records)} records from {dataset_id}.")

            # 2. Patient-level Dataset Splitting (70% Train, 15% Val, 15% Test)
            train_recs, val_recs, test_recs = split_dataset(
                raw_records, train_pct=0.70, val_pct=0.15, test_pct=0.15, patient_id_key="patient_id"
            )
            self.log(f"Leakage-Free Splitting: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)} (Patient-level separated).")

            # 3. Create PyTorch DataLoaders
            img_size = (int(config.get("image_size", 512)), int(config.get("image_size", 512)))
            batch_size = max(1, min(int(config.get("batch_size", 4)), len(train_recs)))
            
            base_dir = os.path.join(PROJECT_ROOT, "data", "datasets", dataset_id)
            train_dataset = RetinalDatasetPyTorch(train_recs, base_dir, image_size=img_size, is_training=True, task_type=task_type)
            val_dataset = RetinalDatasetPyTorch(val_recs, base_dir, image_size=img_size, is_training=False, task_type=task_type)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            self.state["total_steps"] = len(train_loader) * self.state["total_epochs"]

            # 4. Instantiate Selected Model Architecture
            arch = config.get("architecture", "multitask_resunet")
            lr = float(config.get("learning_rate", 0.0001))
            optim_choice = config.get("optimizer", "adamw").lower()

            if arch == "multitask_resunet":
                model = RetinalMultiTaskResUNet(num_seg_classes=4, num_risk_classes=1, backbone='resnet34', pretrained=False)
                # If existing checkpoint exists, load backbone weights
                ckpt_p = os.path.join(CHECKPOINTS_DIR, "best_retinal_resunet.pth")
                if os.path.exists(ckpt_p):
                    try:
                        state_dict = torch.load(ckpt_p, map_location='cpu')
                        model.load_state_dict(state_dict, strict=False)
                        self.log("[+] Initialized with pretrained retinal feature weights.")
                    except Exception as e:
                        self.log(f"[!] Warning loading baseline: {e}")
                self.state["metric_name"] = "Mean Dice & Accuracy"

            elif arch == "resnet50_classifier":
                model = models.resnet50(weights=None)
                model.fc = nn.Linear(model.fc.in_features, 1)
                self.state["metric_name"] = "Validation Accuracy"

            else:
                # Default to ResUNet
                model = RetinalMultiTaskResUNet(num_seg_classes=1, num_risk_classes=1, backbone='resnet34', pretrained=False)
                self.state["metric_name"] = "Vessel Dice Score"

            model = model.to(device)

            # 5. Optimizer & Loss Functions
            if optim_choice == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            elif optim_choice == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

            bce_criterion = nn.BCEWithLogitsLoss()
            
            def dice_loss(pred_logits, target_mask, eps=1e-6):
                probs = torch.sigmoid(pred_logits)
                num = 2.0 * (probs * target_mask).sum() + eps
                den = probs.sum() + target_mask.sum() + eps
                return 1.0 - (num / den)

            self.log(f"Optimizer: {optim_choice.upper()} (lr={lr}) | Total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

            # 6. Training Loop
            best_metric = -1.0
            epochs = self.state["total_epochs"]

            for epoch in range(1, epochs + 1):
                if self._stop_requested:
                    break

                self.state["current_epoch"] = epoch
                model.train()
                epoch_loss = 0.0
                batches_done = 0

                for batch in train_loader:
                    # Check pause
                    self._pause_event.wait()
                    if self._stop_requested:
                        break

                    imgs = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    labels = batch["label"].to(device)

                    optimizer.zero_grad()

                    if arch == "multitask_resunet":
                        seg_out, risk_out = model(imgs)
                        # Multi-task loss
                        # If masks has 1 channel and seg_out has 4, compute on channel 3 or channel 0
                        if masks.shape[1] == 1 and seg_out.shape[1] >= 1:
                            l_seg = dice_loss(seg_out[:, :1], masks)
                        else:
                            l_seg = dice_loss(seg_out, masks)
                        l_risk = bce_criterion(risk_out.view(-1), labels)
                        loss = l_seg + l_risk
                    elif arch == "resnet50_classifier":
                        out = model(imgs).view(-1)
                        loss = bce_criterion(out, labels)
                    else:
                        seg_out, _ = model(imgs)
                        loss = dice_loss(seg_out[:, :1], masks)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batches_done += 1
                    self.state["current_step"] += 1
                    self.state["progress_pct"] = round((self.state["current_step"] / max(1, self.state["total_steps"])) * 100, 1)

                if self._stop_requested:
                    break

                train_loss_avg = round(epoch_loss / max(1, batches_done), 4)
                self.state["train_loss"] = train_loss_avg
                self.state["train_loss_history"].append(train_loss_avg)

                # Validation Phase
                model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                val_dice_sum = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        imgs = batch["image"].to(device)
                        masks = batch["mask"].to(device)
                        labels = batch["label"].to(device)

                        if arch == "multitask_resunet":
                            seg_out, risk_out = model(imgs)
                            l_seg = dice_loss(seg_out[:, :1], masks)
                            l_risk = bce_criterion(risk_out.view(-1), labels)
                            v_loss = l_seg + l_risk

                            # Dice
                            p_mask = (torch.sigmoid(seg_out[:, :1]) > 0.5).float()
                            num = 2.0 * (p_mask * masks).sum()
                            den = p_mask.sum() + masks.sum() + 1e-6
                            val_dice_sum += (num / den).item()

                            # Accuracy
                            preds = (torch.sigmoid(risk_out.view(-1)) > 0.5).float()
                            val_correct += (preds == labels).sum().item()
                            val_total += labels.numel()

                        elif arch == "resnet50_classifier":
                            out = model(imgs).view(-1)
                            v_loss = bce_criterion(out, labels)
                            preds = (torch.sigmoid(out) > 0.5).float()
                            val_correct += (preds == labels).sum().item()
                            val_total += labels.numel()

                        else:
                            seg_out, _ = model(imgs)
                            v_loss = dice_loss(seg_out[:, :1], masks)
                            p_mask = (torch.sigmoid(seg_out[:, :1]) > 0.5).float()
                            num = 2.0 * (p_mask * masks).sum()
                            den = p_mask.sum() + masks.sum() + 1e-6
                            val_dice_sum += (num / den).item()

                        val_loss_sum += v_loss.item()
                        val_batches += 1

                val_loss_avg = round(val_loss_sum / max(1, val_batches), 4)
                self.state["val_loss"] = val_loss_avg
                self.state["val_loss_history"].append(val_loss_avg)

                # Compute target metric
                if arch == "resnet50_classifier":
                    metric_val = round((val_correct / max(1, val_total)) * 100, 2)
                    metric_str = f"Accuracy: {metric_val}%"
                elif "segmentation" in task_type:
                    metric_val = round((val_dice_sum / max(1, val_batches)) * 100, 2)
                    metric_str = f"Mean Dice: {metric_val}%"
                else:
                    metric_val = round(((val_dice_sum / max(1, val_batches)) * 0.5 + (val_correct / max(1, val_total)) * 0.5) * 100, 2)
                    metric_str = f"Multi-Task Score: {metric_val}%"

                self.state["val_metric_history"].append(metric_val)
                self.log(f"Epoch [{epoch}/{epochs}] Complete -> Train Loss: {train_loss_avg} | Val Loss: {val_loss_avg} | {metric_str}")

                # Save best checkpoint
                if metric_val > best_metric or epoch == epochs:
                    best_metric = metric_val
                    save_name = f"{self.state['model_name'].replace(' ', '_')}.pth"
                    save_path = os.path.join(CHECKPOINTS_DIR, save_name)
                    torch.save(model.state_dict(), save_path)
                    self.state["saved_model_path"] = save_path
                    self.log(f"[*] Checkpoint saved: {save_name} (Best score: {best_metric}%)")

            # Finalize
            if self._stop_requested:
                self.state["status"] = "stopped"
                self.log("Training stopped by user.")
            else:
                self.state["status"] = "completed"
                self.state["progress_pct"] = 100.0
                self.log(f"[+] Model training completed successfully! Final Score: {best_metric}%")
                
                # Auto-register newly trained model in Model Registry
                from backend.model_registry import ModelRegistry
                registry = ModelRegistry()
                registry.register_model(
                    model_id=self.state["model_name"].lower().replace(' ', '_'),
                    name=self.state["model_name"],
                    architecture=self.state["architecture"],
                    dataset_id=dataset_id,
                    task_type=task_type,
                    epochs=self.state["total_epochs"],
                    checkpoint_path=self.state["saved_model_path"],
                    val_score=best_metric,
                    test_split_records=test_recs,
                    metrics_history={
                        "train_loss": self.state["train_loss_history"],
                        "val_loss": self.state["val_loss_history"],
                        "val_metric": self.state["val_metric_history"]
                    }
                )
                self.log(f"[+] Model registered in OcuCardio Model Registry!")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.state["status"] = "error"
            self.state["error_message"] = str(e)
            self.log(f"[!] Training error: {str(e)}")

