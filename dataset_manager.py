import os
import json
import zipfile
import shutil
import time
import glob
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REAL_FUNDUS_DIR = os.path.join(DATA_DIR, "real_fundus", "images")
UPLOADED_DIR = os.path.join(DATA_DIR, "uploaded_datasets")
CATALOG_PATH = os.path.join(DATA_DIR, "dataset_catalog.json")

class DatasetManager:
    def __init__(self, ai_engine=None):
        self.ai_engine = ai_engine
        os.makedirs(UPLOADED_DIR, exist_ok=True)
        self.catalog = self._load_or_build_catalog()

    def _load_or_build_catalog(self):
        """Loads existing catalog or builds initial index from pre-existing fundus scans."""
        if os.path.exists(CATALOG_PATH):
            try:
                with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                    return catalog
            except Exception as e:
                print(f"[!] Warning reading catalog: {e}. Rebuilding index.")

        catalog = {"scans": [], "last_updated": time.time()}
        
        # Scan real fundus images
        if os.path.exists(REAL_FUNDUS_DIR):
            for fpath in glob.glob(os.path.join(REAL_FUNDUS_DIR, "*.*")):
                ext = os.path.splitext(fpath)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']:
                    fname = os.path.basename(fpath)
                    try:
                        with Image.open(fpath) as im:
                            w, h = im.size
                    except Exception:
                        w, h = (512, 512)
                    
                    # Clinical presets/heuristic initial tags for preloaded scans
                    is_crao = "crao" in fname.lower()
                    catalog["scans"].append({
                        "id": f"SCAN_{fname.replace('.', '_')}",
                        "filename": fname,
                        "relative_path": os.path.relpath(fpath, PROJECT_ROOT).replace("\\", "/"),
                        "absolute_path": fpath,
                        "source": "Cohort / Preloaded",
                        "dimensions": {"width": w, "height": h},
                        "uploaded_at": os.path.getmtime(fpath),
                        "status": "pending",
                        "analysis": None
                    })

        self._save_catalog(catalog)
        return catalog

    def _save_catalog(self, catalog=None):
        if catalog is None:
            catalog = self.catalog
        catalog["last_updated"] = time.time()
        temp_path = CATALOG_PATH + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2)
            if os.path.exists(temp_path):
                os.replace(temp_path, CATALOG_PATH)
        except Exception as e:
            print(f"[!] Error saving catalog: {e}")

    def get_all_scans(self, search_query="", filter_tier="ALL", filter_lesion="ALL", page=1, page_size=50):
        """Filterable scan library."""
        scans = self.catalog.get("scans", [])
        filtered = []

        for s in scans:
            # Search query
            if search_query:
                q = search_query.lower()
                match_id = q in s["id"].lower()
                match_name = q in s["filename"].lower()
                match_tier = s.get("analysis") and q in s["analysis"]["systemic_vascular_risk"]["tier"].lower()
                if not (match_id or match_name or match_tier):
                    continue

            # Risk Tier filter
            if filter_tier != "ALL":
                if not s.get("analysis"):
                    if filter_tier != "PENDING":
                        continue
                else:
                    tier = s["analysis"]["systemic_vascular_risk"]["level"].upper()
                    if tier != filter_tier.upper():
                        continue

            # Lesion filter
            if filter_lesion != "ALL" and s.get("analysis"):
                bm = s["analysis"].get("biomarkers", {})
                if filter_lesion == "EDEMA" and not bm.get("disc_edema", {}).get("detected"):
                    continue
                elif filter_lesion == "NV" and not bm.get("neovascularization", {}).get("detected"):
                    continue
                elif filter_lesion == "HE" and not bm.get("hard_exudates", {}).get("detected"):
                    continue

            filtered.append(s)

        total_count = len(filtered)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_items = filtered[start_idx:end_idx]

        return {
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "items": page_items,
            "stats": self.get_statistics()
        }

    def get_scan_by_id(self, scan_id):
        for s in self.catalog.get("scans", []):
            if s["id"] == scan_id:
                return s
        return None

    def add_uploaded_file(self, filename, file_bytes, source_label="User Upload"):
        """Saves an uploaded image and adds it to the dataset catalog."""
        timestamp_prefix = int(time.time() * 1000)
        safe_filename = f"{timestamp_prefix}_{filename}"
        dest_path = os.path.join(UPLOADED_DIR, safe_filename)

        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        try:
            with Image.open(dest_path) as im:
                w, h = im.size
        except Exception:
            w, h = (512, 512)

        scan_record = {
            "id": f"UPLOAD_{timestamp_prefix}",
            "filename": filename,
            "relative_path": os.path.relpath(dest_path, PROJECT_ROOT).replace("\\", "/"),
            "absolute_path": dest_path,
            "source": source_label,
            "dimensions": {"width": w, "height": h},
            "uploaded_at": time.time(),
            "status": "pending",
            "analysis": None
        }

        # Put newest at beginning
        self.catalog["scans"].insert(0, scan_record)
        self._save_catalog()
        return scan_record

    def unpack_zip_dataset(self, zip_bytes, dataset_name="uploaded_dataset"):
        """Unpacks a zip archive containing fundus images and catalogs all of them."""
        import io
        timestamp_id = int(time.time())
        dest_folder = os.path.join(UPLOADED_DIR, f"{dataset_name}_{timestamp_id}")
        os.makedirs(dest_folder, exist_ok=True)

        added_scans = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                ext = os.path.splitext(member.filename)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']:
                    extracted_path = z.extract(member, dest_folder)
                    fname = os.path.basename(extracted_path)
                    try:
                        with Image.open(extracted_path) as im:
                            w, h = im.size
                    except Exception:
                        w, h = (512, 512)

                    scan_id = f"ZIP_{timestamp_id}_{len(added_scans)+1:04d}"
                    record = {
                        "id": scan_id,
                        "filename": fname,
                        "relative_path": os.path.relpath(extracted_path, PROJECT_ROOT).replace("\\", "/"),
                        "absolute_path": extracted_path,
                        "source": f"Dataset: {dataset_name}",
                        "dimensions": {"width": w, "height": h},
                        "uploaded_at": time.time(),
                        "status": "pending",
                        "analysis": None
                    }
                    added_scans.append(record)

        self.catalog["scans"] = added_scans + self.catalog["scans"]
        self._save_catalog()
        return {
            "dataset_folder": dest_folder,
            "scans_added": len(added_scans),
            "added": added_scans
        }

    def analyze_scan_by_id(self, scan_id):
        """Runs the AI engine for a specific scan in the catalog and caches results."""
        scan = self.get_scan_by_id(scan_id)
        if not scan:
            raise ValueError(f"Scan not found: {scan_id}")

        if self.ai_engine is None:
            from backend.ai_engine import RetinalAIEngine
            self.ai_engine = RetinalAIEngine()

        results = self.ai_engine.analyze_image(scan["absolute_path"])
        
        # Save compact analysis metrics in catalog (exclude heavy base64 strings from json catalog to keep it fast)
        compact_analysis = {
            "systemic_vascular_risk": results["systemic_vascular_risk"],
            "biomarkers": results["biomarkers"],
            "clinical_interpretation": results["clinical_interpretation"],
            "analyzed_at": time.time()
        }
        
        scan["status"] = "analyzed"
        scan["analysis"] = compact_analysis
        self._save_catalog()

        # Return full results including overlays for live viewer
        return results

    def batch_analyze(self, limit=20):
        """Batch analyzes up to `limit` unanalyzed scans in the dataset."""
        if self.ai_engine is None:
            from backend.ai_engine import RetinalAIEngine
            self.ai_engine = RetinalAIEngine()

        analyzed_count = 0
        for scan in self.catalog.get("scans", []):
            if scan.get("status") == "pending":
                try:
                    res = self.ai_engine.analyze_image(scan["absolute_path"])
                    scan["status"] = "analyzed"
                    scan["analysis"] = {
                        "systemic_vascular_risk": res["systemic_vascular_risk"],
                        "biomarkers": res["biomarkers"],
                        "clinical_interpretation": res["clinical_interpretation"],
                        "analyzed_at": time.time()
                    }
                    analyzed_count += 1
                    if analyzed_count >= limit:
                        break
                except Exception as e:
                    print(f"[!] Error analyzing {scan['filename']}: {e}")
                    scan["status"] = "error"

        self._save_catalog()
        return {
            "analyzed_in_batch": analyzed_count,
            "stats": self.get_statistics()
        }

    def get_statistics(self):
        """Computes aggregate cohort statistics across the dataset."""
        scans = self.catalog.get("scans", [])
        total = len(scans)
        analyzed = [s for s in scans if s.get("status") == "analyzed" and s.get("analysis")]
        analyzed_count = len(analyzed)

        edema_count = 0
        nv_count = 0
        he_count = 0
        critical_count = 0
        high_count = 0
        mod_count = 0
        low_count = 0
        score_sum = 0.0

        for s in analyzed:
            a = s["analysis"]
            svr = a["systemic_vascular_risk"]
            bm = a["biomarkers"]
            score_sum += svr["overall_score"]

            level = svr["level"]
            if level == "Critical":
                critical_count += 1
            elif level == "High":
                high_count += 1
            elif level == "Moderate":
                mod_count += 1
            else:
                low_count += 1

            if bm.get("disc_edema", {}).get("detected"):
                edema_count += 1
            if bm.get("neovascularization", {}).get("detected"):
                nv_count += 1
            if bm.get("hard_exudates", {}).get("detected"):
                he_count += 1

        avg_risk = round(score_sum / analyzed_count, 1) if analyzed_count > 0 else 0.0

        return {
            "total_scans": total,
            "analyzed_count": analyzed_count,
            "pending_count": total - analyzed_count,
            "average_risk_score": avg_risk,
            "prevalence": {
                "disc_edema_count": edema_count,
                "disc_edema_pct": round(edema_count / analyzed_count * 100, 1) if analyzed_count else 0,
                "neovascularization_count": nv_count,
                "neovascularization_pct": round(nv_count / analyzed_count * 100, 1) if analyzed_count else 0,
                "hard_exudates_count": he_count,
                "hard_exudates_pct": round(he_count / analyzed_count * 100, 1) if analyzed_count else 0
            },
            "risk_distribution": {
                "critical": critical_count,
                "high": high_count,
                "moderate": mod_count,
                "low": low_count
            }
        }

    def export_csv(self):
        """Generates a CSV string of all analyzed scans for download."""
        lines = [
            "Scan ID,Filename,Source,Risk Score (%),Risk Tier,5-Yr Stroke (%),5-Yr CAD (%),AVR,Disc Edema,Frisen Grade,NV Detected,NV Type,Hard Exudates Count,HE Severity,Microaneurysms Count,Analyzed Date"
        ]
        for s in self.catalog.get("scans", []):
            if s.get("status") == "analyzed" and s.get("analysis"):
                a = s["analysis"]
                r = a["systemic_vascular_risk"]
                b = a["biomarkers"]
                de = b.get("disc_edema", {})
                nv = b.get("neovascularization", {})
                he = b.get("hard_exudates", {})
                ma = b.get("microaneurysms", {})
                
                dt_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.get("analyzed_at", time.time())))
                row = [
                    s["id"],
                    f'"{s["filename"]}"',
                    f'"{s.get("source", "Upload")}"',
                    str(r["overall_score"]),
                    f'"{r["tier"]}"',
                    str(r.get("stroke_5yr_risk_pct", 0)),
                    str(r.get("cad_athero_risk_pct", 0)),
                    str(r.get("avr_ratio", 0)),
                    "YES" if de.get("detected") else "NO",
                    str(de.get("frisen_grade", 0)),
                    "YES" if nv.get("detected") else "NO",
                    f'"{nv.get("nv_type", "None")}"',
                    str(he.get("count", 0)),
                    f'"{he.get("severity", "None")}"',
                    str(ma.get("count", 0)),
                    f'"{dt_str}"'
                ]
                lines.append(",".join(row))

        return "\n".join(lines)

