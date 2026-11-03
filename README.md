# OcuCardio: AI-Powered Retinal Image Analysis for Cardiovascular Risk Assessment

An end-to-end medical deep learning platform and clinical research workstation combining high-resolution retinal fundus image analysis, real-world multi-institutional dataset integration, live deep learning model training, and rigorous performance evaluation.

---

## 1. Overview & Key Capabilities

**OcuCardio** transforms fundus photography into an objective, non-invasive biomarker for systemic cardiovascular risk profiling. It bridges deep neural networks (Multi-Task ResUNet, ResNet-50) with clinical microvascular geometry (AVR, vascular tortuosity, Frisén papilledema staging, hard exudate foveal threat).

### Primary Platform Sections:
1. **Scan Inspector (Vertical Clinical Dashboard)**:
   - **Section 1: Top Navigation Bar**: Model status indicator, active model pill, benchmark case picker, and navigation tabs.
   - **Section 2: Retinal Image Upload**: Drag-and-drop supporting JPG, PNG, TIFF, and DICOM exports with automated Laplacian sharpness validation and quality grade checks.
   - **Section 3: Interactive Retinal Canvas**: 512×512 centered display with RGB, Red-Free (Green), Contrast CLAHE, 2.5× Loupe Magnifier, and 7 toggleable segmentation layers with opacity control.
   - **Section 4: Dynamic CVD Risk Assessment**: Framingham & UK Biobank calibrated composite CVD risk index, 5-year ischemic stroke hazard, CAD atherosclerosis risk, and Keith-Wagener-Barker classification. (Displays clear notification if active model is specialized for pure vessel segmentation).
   - **Section 5: Detailed Retinal Analysis**: 6 discrete biomarker cards for Disc Edema, Neovascularization, Hard Exudates, Microaneurysms, AVR, and Vessel Density/Complexity.
   - **Section 6: Clinical AI Findings & Specialist Action Plan**: Synthesized diagnostic narrative, prioritized referrals (Neuro-Ophthalmology, Vitreoretinal, Cardiology), and research disclaimers.

2. **Dynamic Dataset Hub & Library**:
   - Manage real clinical retinal fundus datasets:
     - **DRIVE** (Digital Retinal Images for Vessel Extraction) – Retinal vessel segmentation
     - **STARE** (Structured Analysis of the Retina) – Vessel segmentation and arteriolar narrowing
     - **CHASE_DB1** (Child Heart and Health Study) – Microvascular caliber & tortuosity
     - **RFMiD** (Retinal Fundus Multi-Disease) – Multi-class disease classification (45 conditions)
     - **ODIR-5K** (Ocular Disease Intelligent Recognition) – 8 diagnostic classes including Systemic Hypertension
     - **BRSET** (Brazilian Multilabel Dataset) – Multilabel retinal findings and systemic clinical variables
     - **Clinical Retinal CVD Cohort (20,000 Scans)** – UK Biobank and Framingham calibrated outcomes
   - Ingest custom datasets via `.zip` archive upload with metadata CSV parsing and integrity validation.
   - Interactive Image Explorer with search, class filtering, and instant inspection.

3. **Model Training & Performance Evaluation Studio**:
   - 4-step training wizard: Dataset Selection, Preprocessing & Augmentation, Leakage-Free Splitting (70% Train, 15% Val, 15% Test with patient-level separation), Architecture & Hyperparameters.
   - Live background training loop with Start, Pause, Resume, and Stop controls.
   - Real-time epoch loss and validation metric curves, step progress bar, and streaming terminal logs.
   - Independent test set evaluation computing Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, Dice Coefficient, IoU, 2×2 Confusion Matrix, and ROC/PR curves.

4. **Model Management & Benchmarking**:
   - Model Registry tracking saved model versions, architectures, training datasets, and test metrics.
   - One-click active model switcher for live scan inspection.
   - Side-by-side comparison table flagging the **Best Performing Model**.
   - Chronological benchmark experiment history.

---

## 2. Directory Structure

```
retinal_cvd_risk/
├── backend/
│   ├── ai_engine.py           # Deep Multi-Task ResUNet + OpenCV Biomarker Analyzers
│   ├── dataset_hub.py         # Multi-dataset registry, ZIP extraction, metadata parser
│   ├── dataset_manager.py     # Scan indexing & catalog caching
│   ├── training_pipeline.py   # Asynchronous PyTorch training loop & live state
│   ├── evaluation_pipeline.py # Test set evaluation (Dice, IoU, ROC-AUC, Confusion Matrix)
│   ├── model_registry.py      # Checkpoint management & model comparison registry
│   ├── preprocessing.py       # Patient-level splitting, CLAHE, data augmentation
│   ├── seed_datasets.py       # Dataset initialization script
│   └── server.py              # Multi-threaded HTTP REST API server
├── checkpoints/
│   ├── best_retinal_resunet.pth  # Pretrained baseline model (122 MB)
│   └── model_registry.json       # Registry metadata and benchmark history
├── data/
│   ├── datasets/                 # Structured clinical datasets (DRIVE, STARE, CHASE, RFMiD, ODIR, BRSET)
│   ├── cohort_20000_images/      # 20,000 patient fundus images (512x512)
│   └── real_fundus/              # Benchmark seed images, masks, and manifest
├── frontend/
│   └── index.html             # OcuCardio Healthcare AI Dashboard
├── models/
│   ├── multitask_resunet.py   # Dual-Head ResNet-34/50 U-Net architecture
│   └── losses.py              # Soft Dice, Focal, and Multi-Task losses
├── index.html                 # Root web entry (synchronized with frontend/index.html)
├── run_app.bat                # 1-Click Windows Desktop Launcher
├── run_app.py                 # Cross-platform launcher
└── README.md                  # System documentation
```

---

## 3. How to Launch and Use

### Option 1: 1-Click Windows Desktop Launcher
Double-click `run_app.bat` in the project root.  
It starts the multi-threaded backend server and automatically opens `http://localhost:8080` in your default browser.

### Option 2: Python Command Line
```powershell
# Navigate to project folder
cd C:\Users\gagan\.gemini\antigravity\scratch\retinal_cvd_risk

# Start server with active virtual environment
.\.venv\Scripts\python.exe backend\server.py 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your web browser.
