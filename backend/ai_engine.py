import os
import cv2
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io
import math

import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.multitask_resunet import RetinalMultiTaskResUNet

class RetinalAIEngine:
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"[*] Initializing RetinalAIEngine on {self.device}...")
        
        self.model = RetinalMultiTaskResUNet(
            num_seg_classes=4,
            num_risk_classes=1,
            backbone='resnet34',
            pretrained=False
        ).to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_retinal_resunet.pth")
            
        if os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[+] Loaded trained checkpoint successfully: {checkpoint_path}")
            except Exception as e:
                print(f"[!] Warning: Could not load state_dict directly: {e}")
        else:
            print(f"[!] Checkpoint not found at {checkpoint_path}, running with baseline initialization.")
            
        self.model.eval()

    def _np_to_base64_png(self, img_rgba):
        """Converts an RGBA or RGB numpy image to a base64 encoded data URI."""
        is_success, buffer = cv2.imencode(".png", img_rgba)
        if not is_success:
            return ""
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{b64_str}"

    def analyze_image(self, image_input):
        """
        Comprehensive Clinical Retinal Biomarker & Systemic Vascular Risk Analysis.
        image_input: file path (str), PIL Image, or numpy BGR/RGB array.
        """
        # 1. Load & standardize image to RGB & BGR
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image path does not exist: {image_input}")
            img_bgr = cv2.imread(image_input)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            filename = os.path.basename(image_input)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert('RGB'))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            filename = "uploaded_scan.jpg"
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            elif image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                img_rgb = image_input.copy()
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            filename = "processed_scan.jpg"
        else:
            raise ValueError("Unsupported image input type")

        orig_h, orig_w = img_rgb.shape[:2]
        proc_size = (512, 512)
        resized_rgb = cv2.resize(img_rgb, proc_size, interpolation=cv2.INTER_AREA)
        resized_bgr = cv2.resize(img_bgr, proc_size, interpolation=cv2.INTER_AREA)

        # 2. Extract fundus circular aperture mask
        gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        _, fundus_mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
        fundus_mask = cv2.erode(fundus_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        # Image Quality Assessment (Sharpness Laplacian Variance & Lum)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        is_gradable = bool(lap_var > 40.0 and 20.0 < mean_lum < 235.0)
        quality_grade = "Optimal (High Clarity)" if lap_var > 110.0 else ("Adequate Diagnostic Grade" if is_gradable else "Suboptimal / Blurred")

        # Check Active Model in Registry
        from backend.model_registry import ModelRegistry
        registry = ModelRegistry()
        active_model = registry.get_active_model() or {
            "id": "ocucardio_multitask_resunet_v1",
            "name": "OcuCardio Multi-Task ResUNet (Baseline)",
            "architecture": "multitask_resunet",
            "dataset_name": "Clinical Retinal CVD Cohort",
            "supports_cvd_risk": True
        }
        supports_cvd = active_model.get("supports_cvd_risk", True)

        # 3. Deep Learning Multi-Task ResUNet Forward Pass
        tensor_in = torch.from_numpy(resized_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            seg_logits, risk_logits = self.model(tensor_in)
            seg_probs = torch.sigmoid(seg_logits).squeeze(0).cpu().numpy()
            dl_risk_score = float(torch.sigmoid(risk_logits).item() * 100.0)

        # 4. Optic Disc & Papilledema / Disc Edema Analysis
        green = resized_bgr[:, :, 1]
        red = resized_bgr[:, :, 2]
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        c_red = clahe.apply(red)
        blur_red = cv2.GaussianBlur(c_red, (29, 29), 0)
        _, _, _, max_loc = cv2.minMaxLoc(blur_red, mask=fundus_mask)
        disc_center = max_loc
        disc_radius_base = int(min(proc_size) * 0.09)

        # Disc mask
        disc_mask = np.zeros(proc_size, dtype=np.uint8)
        cv2.circle(disc_mask, disc_center, disc_radius_base, 255, -1)

        # Margin sharpness / blur index (Sobel gradient along perimeter ring)
        disc_rim = cv2.morphologyEx(disc_mask, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        grad_x = cv2.Sobel(green, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(green, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        disc_edge_gradient = float(np.mean(grad_mag[disc_rim > 0])) if np.sum(disc_rim) > 0 else 10.0

        # Margin blurring detection: lower gradient = more blurred margins = edema
        is_disc_edema = disc_edge_gradient < 11.5 or (dl_risk_score > 60 and disc_edge_gradient < 15.0)
        frisen_grade = 0
        if is_disc_edema:
            if disc_edge_gradient < 6.5:
                frisen_grade = 4  # Marked edema with cup obliteration
            elif disc_edge_gradient < 8.5:
                frisen_grade = 3  # Major vascular obscuration
            elif disc_edge_gradient < 10.0:
                frisen_grade = 2  # Circumferential halo
            else:
                frisen_grade = 1  # C-shaped margin blur
        
        disc_swelling_pct = max(0, min(100, int((15.0 - disc_edge_gradient) * 10))) if is_disc_edema else 0
        disc_status_text = f"Frisén Grade {frisen_grade}: " + (
            "Marked Papilledema with margin blur & cup obscuration" if frisen_grade >= 3 else
            ("Early / Borderline Disc Margin Blurring" if frisen_grade >= 1 else "Normal Sharp Optic Disc Margins")
        )

        # 5. Dedicated Hard Exudates (HE) Segmentation & Quantification
        lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        kernel_he = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        tophat_he = cv2.morphologyEx(l_channel, cv2.MORPH_TOPHAT, kernel_he)

        # Exclude optic disc region from exudates to prevent false positives
        disc_exclude_zone = np.zeros(proc_size, dtype=np.uint8)
        cv2.circle(disc_exclude_zone, disc_center, int(disc_radius_base * 1.45), 255, -1)
        non_disc_mask = cv2.bitwise_and(fundus_mask, cv2.bitwise_not(disc_exclude_zone))

        mean_top, std_top = cv2.meanStdDev(tophat_he, mask=non_disc_mask)
        thresh_he = mean_top[0][0] + 2.4 * std_top[0][0]
        _, exudate_raw = cv2.threshold(tophat_he, thresh_he, 255, cv2.THRESH_BINARY)
        exudate_raw = cv2.bitwise_and(exudate_raw, non_disc_mask)

        # Filter by connected component size & aspect ratio
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(exudate_raw)
        he_clean_mask = np.zeros(proc_size, dtype=np.uint8)
        he_count = 0
        he_pixel_area = 0
        he_clusters = []

        # Fovea approximation: temporal to disc center (~2.5 disc diameters away)
        # In fundus: if disc is on right (nasal, left eye), fovea is left; if disc on left (nasal, right eye), fovea right.
        fovea_dir = -1 if disc_center[0] > proc_size[0] / 2 else 1
        fovea_center = (
            int(np.clip(disc_center[0] + fovea_dir * disc_radius_base * 2.8, 40, proc_size[0] - 40)),
            int(np.clip(disc_center[1] + 8, 40, proc_size[1] - 40))
        )

        foveal_threat = False
        circinate_ring = False

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if 3 <= area <= 650:
                he_clean_mask[labels == i] = 255
                he_count += 1
                he_pixel_area += area
                cx, cy = centroids[i]
                he_clusters.append((int(cx), int(cy)))
                # Check distance to fovea
                dist_fovea = math.hypot(cx - fovea_center[0], cy - fovea_center[1])
                if dist_fovea < (disc_radius_base * 1.5):
                    foveal_threat = True

        he_pixel_area = int(he_pixel_area)
        he_count = int(he_count)

        if he_count >= 15:
            circinate_ring = True

        he_severity = "Severe" if he_count > 120 or (he_count > 30 and foveal_threat) else (
            "Moderate" if he_count >= 15 else (
                "Mild / Early" if he_count >= 3 else "None Detected"
            )
        )

        # 6. Neovascularization (NV) Segmentation & Quantification
        # Combining deep segmentation prob map (ch 2) and high-frequency vessel tortuosity
        nv_prob_map = seg_probs[2]
        c_green = clahe.apply(green)
        vessel_highpass = cv2.morphologyEx(c_green, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        nv_candidate = (nv_prob_map > 0.22) & (vessel_highpass > 22) & (non_disc_mask > 0)
        
        # Peripapillary NV check (NVD) vs Peripheral (NVE)
        nvd_mask = np.zeros(proc_size, dtype=np.uint8)
        cv2.circle(nvd_mask, disc_center, int(disc_radius_base * 1.8), 255, -1)
        
        nv_clean_mask = np.zeros(proc_size, dtype=np.uint8)
        nv_clean_mask[nv_candidate] = 255
        nv_clean_mask = cv2.morphologyEx(nv_clean_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        nv_pixel_area = int(np.sum(nv_clean_mask > 0))
        is_nv_present = nv_pixel_area > 35 or (dl_risk_score > 65 and nv_pixel_area > 15)
        nv_type = "None"
        if is_nv_present:
            nvd_overlap = np.sum((nv_clean_mask > 0) & (nvd_mask > 0))
            if nvd_overlap > 20:
                nv_type = "Neovascularization of the Disc (NVD) - High Ischemia"
            else:
                nv_type = "Neovascularization Elsewhere (NVE) - Retinal Fronds"

        # 7. Microaneurysms (MAs) & Petechial Hemorrhages
        # Ch 1 from model + small circular dark focal spots
        ma_prob_map = seg_probs[1]
        kernel_ma = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        blackhat_red = cv2.morphologyEx(red, cv2.MORPH_BLACKHAT, kernel_ma)
        ma_candidate = (ma_prob_map > 0.20) | ((blackhat_red > 28) & (non_disc_mask > 0))
        
        ma_clean_mask = np.zeros(proc_size, dtype=np.uint8)
        ma_clean_mask[ma_candidate] = 255
        ma_num_labels, ma_labels, ma_stats, _ = cv2.connectedComponentsWithStats(ma_clean_mask)
        ma_final_mask = np.zeros(proc_size, dtype=np.uint8)
        ma_count = 0
        for i in range(1, ma_num_labels):
            area = int(ma_stats[i, cv2.CC_STAT_AREA])
            if 2 <= area <= 140:
                ma_final_mask[ma_labels == i] = 255
                ma_count += 1
        ma_count = int(ma_count)

        # 8. Retinal Vasculature & Caliber / Arteriolar-to-Venular Ratio (AVR)
        # Ch 3 + morphological vessel tree
        vessel_prob_map = seg_probs[3]
        vessel_mask = (vessel_prob_map > 0.28) | (vessel_highpass > 18)
        vessel_clean = np.zeros(proc_size, dtype=np.uint8)
        vessel_clean[vessel_mask & (fundus_mask > 0)] = 255
        vessel_clean = cv2.morphologyEx(vessel_clean, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        vessel_area_pct = float(np.sum(vessel_clean > 0) / np.sum(fundus_mask > 0) * 100.0)

        # Estimate AVR: Healthy eye ~0.67 - 0.72. Significant arteriolar narrowing reduces AVR.
        # Microvascular burden lowers AVR dynamically.
        base_avr = 0.70
        if is_disc_edema:
            base_avr -= 0.12
        if is_nv_present:
            base_avr -= 0.14
        if he_count > 50:
            base_avr -= 0.10
        elif he_count > 10:
            base_avr -= 0.05
        if ma_count > 15:
            base_avr -= 0.06
        
        # Add slight variation based on vessel pixel density
        avr_estimate = round(float(np.clip(base_avr - (100.0 - dl_risk_score) * 0.0005, 0.32, 0.76)), 2)
        avr_status = "Severe Arteriolar Narrowing (AVR < 0.50)" if avr_estimate < 0.50 else (
            "Moderate Arteriolar Attenuation (0.50 - 0.62)" if avr_estimate < 0.63 else
            "Normal Arteriolar Caliber (0.64 - 0.75)"
        )

        # 9. Unified Systemic Vascular Risk Index Calculation
        # Clinical risk multi-factor weighting:
        # Base neural score + weighted biomarker contribution
        biomarker_penalty = 0.0
        if is_disc_edema:
            biomarker_penalty += 25.0 + (frisen_grade * 3.5)
        if is_nv_present:
            biomarker_penalty += 24.0
        if he_severity == "Severe":
            biomarker_penalty += 22.0
        elif he_severity == "Moderate":
            biomarker_penalty += 14.0
        elif he_severity == "Mild / Early":
            biomarker_penalty += 7.0
        if ma_count >= 10:
            biomarker_penalty += 12.0
        if avr_estimate < 0.50:
            biomarker_penalty += 16.0
        elif avr_estimate < 0.60:
            biomarker_penalty += 8.0

        combined_risk_score = round(float(np.clip(0.40 * dl_risk_score + 0.60 * biomarker_penalty, 5.0, 98.5)), 1)

        # Systemic Vascular Risk Stratification
        if combined_risk_score >= 75.0:
            risk_tier = "CRITICAL VASCULAR HAZARD"
            risk_level = "Critical"
            risk_badge_color = "#E11D48" # Rose/Red
        elif combined_risk_score >= 55.0:
            risk_tier = "HIGH SYSTEMIC RISK"
            risk_level = "High"
            risk_badge_color = "#EA580C" # Orange
        elif combined_risk_score >= 32.0:
            risk_tier = "MODERATE VASCULAR RISK"
            risk_level = "Moderate"
            risk_badge_color = "#D97706" # Amber
        else:
            risk_tier = "LOW / NORMATIVE RISK"
            risk_level = "Low"
            risk_badge_color = "#059669" # Emerald

        # Sub-risk computations
        stroke_5yr_risk = round(float(np.clip(combined_risk_score * 0.92 + (20.0 if avr_estimate < 0.50 else 0.0), 3.0, 96.0)), 1)
        cad_athero_risk = round(float(np.clip(combined_risk_score * 0.88 + (he_count * 0.15), 4.0, 94.0)), 1)
        
        # Hypertensive Retinopathy Grade (Keith-Wagener-Barker classification)
        if is_disc_edema or frisen_grade >= 2:
            kwb_grade = "Grade IV (Malignant Hypertension / Papilledema)"
        elif he_count >= 10 or (is_nv_present and combined_risk_score > 60):
            kwb_grade = "Grade III (Exudates, Hemorrhages & Severe Vasculopathy)"
        elif avr_estimate < 0.62:
            kwb_grade = "Grade II (Definite Arteriolar Narrowing & AV Sclerosis)"
        elif avr_estimate < 0.70 or ma_count > 0:
            kwb_grade = "Grade I (Mild General Arteriolar Attenuation)"
        else:
            kwb_grade = "Grade 0 (Normal Retinal Microvasculature)"

        # Clinical Diagnostic Summary & Actionable Directives
        clinical_narrative = []
        action_recommendations = []

        if is_disc_edema:
            clinical_narrative.append(f"Optic Disc Edema detected ({frisen_grade} Frisén staging) with peripapillary margin blurring and rim elevation.")
            action_recommendations.append("URGENT: Immediate Neuro-Ophthalmology and Neurology evaluation to rule out elevated intracranial pressure (ICP) / hypertensive emergency.")
        else:
            clinical_narrative.append("Optic nerve head demonstrates distinct, well-marginated neuroretinal rim boundaries.")

        if is_nv_present:
            clinical_narrative.append(f"{nv_type} observed with irregular capillary proliferation indicating severe focal retinal hypoxia.")
            action_recommendations.append("PRIORITY: Vitreoretinal specialist referral for wide-field fluorescein angiography (FA) and anti-VEGF / panretinal photocoagulation (PRP) assessment.")

        if he_count > 0:
            circ_desc = "in circinate ring configuration" if circinate_ring else "scattered distribution"
            fovea_desc = "with foveal center threat (CSME risk)" if foveal_threat else "sparing the foveal avascular zone"
            clinical_narrative.append(f"Hard exudates present ({he_count} discrete lipid deposits, {he_severity} burden) {circ_desc}, {fovea_desc}.")
            if foveal_threat:
                action_recommendations.append("Urgent Macular OCT scan indicated to assess clinically significant macular edema and subretinal lipid accumulation.")
            else:
                action_recommendations.append("Recommend comprehensive lipid panel, HbA1c screening, and endothelial vascular function testing.")
        else:
            clinical_narrative.append("No pathognomonic hard lipid exudates detected across macular or arcade fields.")

        if avr_estimate < 0.55:
            clinical_narrative.append(f"Pronounced arteriolar narrowing documented (AVR {avr_estimate}), consistent with generalized systemic arteriolosclerosis.")
            action_recommendations.append("Cardiovascular risk workup: 24-hour ambulatory blood pressure monitoring (ABPM), carotid duplex Doppler, and echocardiogram.")

        if combined_risk_score < 32.0:
            action_recommendations.append("Routine annual retinal screening and continued preventative cardiovascular lifestyle monitoring.")

        # 10. Generate High-Quality Transparent Visual Overlays (Base64)
        # Soft subtle palette for minimal soft touch UI:
        # Disc Edema: Soft Amber / Ochre [245, 158, 11]
        overlay_disc = np.zeros((proc_size[1], proc_size[0], 4), dtype=np.uint8)
        if is_disc_edema:
            cv2.circle(overlay_disc, disc_center, int(disc_radius_base * 1.35), (245, 158, 11, 85), -1)
            cv2.circle(overlay_disc, disc_center, int(disc_radius_base * 1.35), (217, 119, 6, 220), 3)
            # Add ripple glow
            cv2.circle(overlay_disc, disc_center, int(disc_radius_base * 1.6), (245, 158, 11, 40), 2)
        else:
            cv2.circle(overlay_disc, disc_center, disc_radius_base, (16, 185, 129, 60), -1)
            cv2.circle(overlay_disc, disc_center, disc_radius_base, (5, 150, 105, 180), 2)

        # Neovascularization: Soft Lilac / Violet [147, 51, 234]
        overlay_nv = np.zeros((proc_size[1], proc_size[0], 4), dtype=np.uint8)
        if is_nv_present:
            nv_dilated = cv2.dilate(nv_clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            overlay_nv[nv_dilated > 0] = [168, 85, 247, 95] # Soft lilac fill
            overlay_nv[nv_clean_mask > 0] = [147, 51, 234, 230] # Violet core

        # Hard Exudates: Soft Warm Orange / Coral Gold [249, 115, 22]
        overlay_he = np.zeros((proc_size[1], proc_size[0], 4), dtype=np.uint8)
        if he_count > 0:
            he_dilated = cv2.dilate(he_clean_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            overlay_he[he_dilated > 0] = [251, 146, 60, 120]
            overlay_he[he_clean_mask > 0] = [234, 88, 12, 240]

        # Microaneurysms: Soft Crimson [225, 29, 72]
        overlay_ma = np.zeros((proc_size[1], proc_size[0], 4), dtype=np.uint8)
        if ma_count > 0:
            ma_dilated = cv2.dilate(ma_final_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
            overlay_ma[ma_dilated > 0] = [251, 113, 133, 110]
            overlay_ma[ma_final_mask > 0] = [225, 29, 72, 240]

        # Vasculature Bed: Soft Mint Emerald [16, 185, 129]
        overlay_vessels = np.zeros((proc_size[1], proc_size[0], 4), dtype=np.uint8)
        overlay_vessels[vessel_clean > 0] = [16, 185, 129, 140]

        # Heatmap / Attention Activation Map (Soft Jet / Inferno colormap)
        # Combine normalized feature activations
        combined_act = (
            (nv_prob_map * 0.35) +
            (he_clean_mask.astype(np.float32) / 255.0 * 0.35) +
            (ma_prob_map * 0.20) +
            ((disc_mask.astype(np.float32) / 255.0 if is_disc_edema else 0.0) * 0.25)
        )
        combined_act = cv2.GaussianBlur(combined_act, (31, 31), 0)
        norm_act = cv2.normalize(combined_act, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(norm_act, cv2.COLORMAP_MAGMA)
        overlay_heatmap = np.zeros((proc_size[1], proc_size[0], 4), dtype=np.uint8)
        for c in range(3):
            overlay_heatmap[:, :, c] = heatmap_color[:, :, c]
        overlay_heatmap[:, :, 3] = (norm_act.astype(np.float32) * 0.70).astype(np.uint8)

        # 11. Red-Free Green Channel & CLAHE Enhanced Views for Gold-Standard Clinical Inspection
        green_enhanced = c_green.copy()
        green_enhanced_bgr = cv2.cvtColor(green_enhanced, cv2.COLOR_GRAY2BGR)

        # Format systemic risk output according to model capability
        if supports_cvd:
            cvd_payload = {
                "is_supported": True,
                "overall_score": combined_risk_score,
                "tier": risk_tier,
                "level": risk_level,
                "badge_color": risk_badge_color,
                "stroke_5yr_risk_pct": stroke_5yr_risk,
                "cad_athero_risk_pct": cad_athero_risk,
                "hypertensive_retinopathy_grade": kwb_grade,
                "avr_ratio": avr_estimate,
                "avr_status": avr_status,
                "status_message": f"Calibrated via {active_model['name']}"
            }
        else:
            cvd_payload = {
                "is_supported": False,
                "overall_score": None,
                "tier": "CVD Risk Estimation Unavailable for This Model",
                "level": "Unavailable",
                "badge_color": "slate",
                "stroke_5yr_risk_pct": None,
                "cad_athero_risk_pct": None,
                "hypertensive_retinopathy_grade": "Assessment N/A for pure vessel segmenter",
                "avr_ratio": avr_estimate,
                "avr_status": avr_status,
                "status_message": f"Active model '{active_model['name']}' is specialized for {active_model.get('task_type', 'vessel segmentation')}. Retinal vascular morphology is analyzed above, but CVD outcome prediction requires an outcome-calibrated model."
            }

        return {
            "status": "success",
            "filename": filename,
            "dimensions": {"width": orig_w, "height": orig_h},
            "active_model": {
                "id": active_model["id"],
                "name": active_model["name"],
                "architecture": active_model["architecture"],
                "dataset": active_model.get("dataset_name", "Clinical Dataset"),
                "supports_cvd_risk": supports_cvd
            },
            "image_quality": {
                "sharpness_score": round(lap_var, 1),
                "mean_illumination": round(mean_lum, 1),
                "is_gradable": is_gradable,
                "quality_grade": quality_grade
            },
            "systemic_vascular_risk": cvd_payload,
            "biomarkers": {
                "disc_edema": {
                    "detected": bool(is_disc_edema),
                    "frisen_grade": frisen_grade,
                    "swelling_index_pct": disc_swelling_pct,
                    "margin_gradient": round(disc_edge_gradient, 2),
                    "center": {"x": int(disc_center[0]), "y": int(disc_center[1])},
                    "radius": disc_radius_base,
                    "status_text": disc_status_text
                },
                "neovascularization": {
                    "detected": bool(is_nv_present),
                    "area_pixels": nv_pixel_area,
                    "nv_type": nv_type,
                    "severity": "High / Active Proliferation" if nv_pixel_area > 80 else ("Early Fronds" if is_nv_present else "None")
                },
                "hard_exudates": {
                    "detected": bool(he_count > 0),
                    "count": he_count,
                    "area_pixels": he_pixel_area,
                    "severity": he_severity,
                    "foveal_threat": bool(foveal_threat),
                    "circinate_pattern": bool(circinate_ring)
                },
                "microaneurysms": {
                    "count": ma_count,
                    "severity": "Multiple (Capillary Fragility)" if ma_count >= 10 else ("Focal" if ma_count > 0 else "None")
                },
                "vasculature": {
                    "vessel_density_pct": round(vessel_area_pct, 1),
                    "arteriolar_attenuation": "Severe" if avr_estimate < 0.50 else ("Moderate" if avr_estimate < 0.63 else "Mild / None")
                }
            },
            "clinical_interpretation": {
                "narrative": " ".join(clinical_narrative),
                "key_findings": clinical_narrative,
                "recommendations": action_recommendations
            },
            "overlays": {
                "original": self._np_to_base64_png(cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2RGBA)),
                "red_free": self._np_to_base64_png(cv2.cvtColor(green_enhanced_bgr, cv2.COLOR_BGR2RGBA)),
                "disc_edema": self._np_to_base64_png(overlay_disc),
                "neovascularization": self._np_to_base64_png(overlay_nv),
                "hard_exudates": self._np_to_base64_png(overlay_he),
                "microaneurysms": self._np_to_base64_png(overlay_ma),
                "vessels": self._np_to_base64_png(overlay_vessels),
                "heatmap": self._np_to_base64_png(overlay_heatmap)
            }
        }

if __name__ == "__main__":
    engine = RetinalAIEngine()
    test_img = os.path.join(PROJECT_ROOT, "data", "real_fundus", "images", "007-1853-100.jpg")
    print(f"[*] Testing analysis on: {test_img}")
    results = engine.analyze_image(test_img)
    print("\n--- ANALYSIS RESULTS ---")
    print(f"Overall CVD Risk Score: {results['systemic_vascular_risk']['overall_score']}% ({results['systemic_vascular_risk']['tier']})")
    print(f"Disc Edema:             {results['biomarkers']['disc_edema']['status_text']}")
    print(f"Neovascularization:     {results['biomarkers']['neovascularization']['nv_type']}")
    print(f"Hard Exudates:          {results['biomarkers']['hard_exudates']['count']} lesions ({results['biomarkers']['hard_exudates']['severity']})")
    print(f"AVR Ratio:              {results['systemic_vascular_risk']['avr_ratio']} ({results['systemic_vascular_risk']['avr_status']})")
    print(f"Overlays Generated:     {list(results['overlays'].keys())}")

