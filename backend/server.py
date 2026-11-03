"""
OcuCardio - HTTP Server & REST API
AI-Powered Retinal Image Analysis for Cardiovascular Risk Assessment
Hosts RESTful API endpoints for:
- Scan Inspector & AI Retinal Analysis
- Dynamic Dataset Hub (DRIVE, STARE, CHASE_DB1, RFMiD, ODIR-5K, BRSET, 20K Cohort)
- Live Model Training & Evaluation Pipeline (Progress, Metrics, Curves)
- Model Registry & Benchmarking Management
"""

import os
import sys
import json
import time
import base64
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import io
import cgi
import traceback
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ai_engine import RetinalAIEngine
from backend.dataset_hub import DatasetHub
from backend.training_pipeline import TrainingPipeline
from backend.evaluation_pipeline import EvaluationPipeline
from backend.model_registry import ModelRegistry
from backend.dataset_manager import DatasetManager

# Global Shared Singletons
AI_ENGINE = RetinalAIEngine()
DATASET_HUB = DatasetHub()
TRAINING_PIPELINE = TrainingPipeline()
EVALUATION_PIPELINE = EvaluationPipeline()
MODEL_REGISTRY = ModelRegistry()
DATASET_MGR = DatasetManager(ai_engine=AI_ENGINE)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

class OcuCardioHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, data, status_code=200):
        try:
            body = json.dumps(data, cls=NumpyEncoder).encode("utf-8")
        except Exception as e:
            print(f"[!] Error encoding JSON: {e}")
            body = json.dumps({"status": "error", "error": str(e)}).encode("utf-8")
            status_code = 500
        self.send_response(status_code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status_code=400):
        self._send_json({"status": "error", "error": str(message)}, status_code=status_code)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path.rstrip("/")
        if not path:
            path = "/"
        query = urllib.parse.parse_qs(parsed_url.query)

        # -------------------------------------------------------------
        # 1. Root / UI Template
        # -------------------------------------------------------------
        if path == "/" or path == "/index.html":
            html_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
            if not os.path.exists(html_path):
                html_path = os.path.join(PROJECT_ROOT, "index.html")
            try:
                with open(html_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                self.send_error(500, f"Error reading frontend template: {e}")
                return

        # -------------------------------------------------------------
        # 2. Dataset Hub APIs
        # -------------------------------------------------------------
        if path == "/api/datasets":
            datasets = DATASET_HUB.get_all_datasets()
            self._send_json({"datasets": datasets})
            return

        if path == "/api/datasets/stats":
            stats = DATASET_HUB.get_dataset_stats()
            self._send_json(stats)
            return

        if path.startswith("/api/datasets/") and path.endswith("/preview"):
            parts = path.split("/")
            dataset_id = parts[3]
            page = int(query.get("page", ["1"])[0])
            page_size = int(query.get("page_size", ["24"])[0])
            search_q = query.get("search", [""])[0]
            class_filter = query.get("class", [""])[0]
            result = DATASET_HUB.preview_dataset_images(dataset_id, page=page, page_size=page_size, search=search_q, class_filter=class_filter)
            self._send_json(result)
            return

        if path.startswith("/api/datasets/") and path.endswith("/validate"):
            parts = path.split("/")
            dataset_id = parts[3]
            result = DATASET_HUB.validate_dataset(dataset_id)
            self._send_json(result)
            return

        if path.startswith("/api/datasets/") and len(path.split("/")) == 4:
            dataset_id = path.split("/")[3]
            info = DATASET_HUB.get_dataset_by_id(dataset_id)
            if info:
                self._send_json(info)
            else:
                self._send_error_json(f"Dataset {dataset_id} not found", status_code=404)
            return

        # Backward-compatible catalog endpoint
        if path == "/api/dataset":
            search_q = query.get("search", [""])[0]
            filter_tier = query.get("tier", ["ALL"])[0]
            filter_lesion = query.get("lesion", ["ALL"])[0]
            page = int(query.get("page", ["1"])[0])
            page_size = int(query.get("page_size", ["50"])[0])
            data = DATASET_MGR.get_all_scans(
                search_query=search_q, filter_tier=filter_tier, filter_lesion=filter_lesion, page=page, page_size=page_size
            )
            self._send_json(data)
            return

        if path == "/api/dataset/stats":
            stats = DATASET_MGR.get_statistics()
            self._send_json(stats)
            return

        if path == "/api/dataset/cohort-20k":
            manifest_p = os.path.join(PROJECT_ROOT, "data", "real_fundus", "cohort_20k_manifest.json")
            if os.path.exists(manifest_p):
                search_q = query.get("q", [""])[0].strip().upper()
                try:
                    with open(manifest_p, "r", encoding="utf-8") as f:
                        cohort = json.load(f)
                    if search_q:
                        filtered = [p for p in cohort if search_q in p["patient_id"] or search_q in p["diagnosis"].upper()]
                        self._send_json({"total": len(filtered), "matches": filtered[:50]})
                    else:
                        self._send_json({
                            "total_patients": len(cohort),
                            "distribution": {"low_risk": 12000, "moderate_risk": 5000, "high_risk": 3000},
                            "sample": cohort[:20]
                        })
                    return
                except Exception as e:
                    self._send_error_json(str(e))
                    return

        # -------------------------------------------------------------
        # 3. Model Training & Evaluation APIs
        # -------------------------------------------------------------
        if path == "/api/training/status":
            status = TRAINING_PIPELINE.get_status()
            self._send_json(status)
            return

        if path == "/api/training/logs":
            logs = TRAINING_PIPELINE.state.get("logs", [])
            self._send_json({"logs": logs})
            return

        # -------------------------------------------------------------
        # 4. Model Registry & Benchmark APIs
        # -------------------------------------------------------------
        if path == "/api/models":
            models_list = MODEL_REGISTRY.get_all_models()
            self._send_json({"models": models_list})
            return

        if path == "/api/models/active":
            active_m = MODEL_REGISTRY.get_active_model()
            self._send_json({"active_model": active_m})
            return

        if path == "/api/models/compare":
            comparison = MODEL_REGISTRY.get_model_comparison()
            self._send_json(comparison)
            return

        # -------------------------------------------------------------
        # 5. Clinical Benchmark Samples
        # -------------------------------------------------------------
        if path == "/api/sample-scans":
            sample_ids = [
                {"id": "007-1853-100.jpg", "label": "Severe Retinopathy (Hard Exudates & Edema)", "tier": "High Risk"},
                {"id": "patient_crao_01.jpg", "label": "CRAO Eye Stroke (Arterial Occlusion)", "tier": "Critical Risk"},
                {"id": "007-2408-100.jpg", "label": "Moderate Hypertensive Retinopathy", "tier": "Moderate Risk"},
                {"id": "007-2454-100.jpg", "label": "Arteriolar Attenuation & Vascular Tortuosity", "tier": "Moderate Risk"},
                {"id": "007-2470-100.jpg", "label": "Normative Retinal Scan", "tier": "Low Risk"}
            ]
            samples = []
            for s in sample_ids:
                fpath = os.path.join(PROJECT_ROOT, "data", "real_fundus", "images", s["id"])
                if os.path.exists(fpath):
                    samples.append({
                        "filename": s["id"],
                        "label": s["label"],
                        "tier": s["tier"],
                        "relative_path": f"data/real_fundus/images/{s['id']}"
                    })
            self._send_json({"samples": samples})
            return

        # -------------------------------------------------------------
        # 6. Static File Serving
        # -------------------------------------------------------------
        if path.startswith("/data/") or path.startswith("/outputs/"):
            rel_path = path.lstrip("/")
            file_path = os.path.join(PROJECT_ROOT, rel_path)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                ext = os.path.splitext(file_path)[1].lower()
                mime = "image/jpeg" if ext in ['.jpg', '.jpeg'] else (
                    "image/png" if ext == '.png' else "application/octet-stream"
                )
                try:
                    with open(file_path, "rb") as f:
                        data = f.read()
                    self.send_response(200)
                    self._send_cors_headers()
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                except Exception as e:
                    self.send_error(500, f"Error reading file: {e}")
                    return

        self.send_error(404, f"Endpoint not found: {path}")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path.rstrip("/")
        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))

        # -------------------------------------------------------------
        # 1. Scan Inspector Analysis
        # -------------------------------------------------------------
        if path == "/api/analyze":
            try:
                if "application/json" in content_type:
                    body = self.rfile.read(content_length).decode("utf-8")
                    req_data = json.loads(body)

                    if "image" in req_data:
                        img_b64 = req_data["image"]
                        if "," in img_b64:
                            img_b64 = img_b64.split(",")[1]
                        img_bytes = base64.b64decode(img_b64)
                        from PIL import Image
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        results = AI_ENGINE.analyze_image(pil_img)
                        results["filename"] = req_data.get("filename", "uploaded_retina.jpg")
                        self._send_json(results)
                        return

                    elif "relative_path" in req_data:
                        rel = req_data["relative_path"].replace("/", os.sep)
                        fpath = os.path.join(PROJECT_ROOT, rel)
                        results = AI_ENGINE.analyze_image(fpath)
                        results["filename"] = req_data.get("filename", os.path.basename(fpath))
                        results["relative_path"] = req_data["relative_path"]
                        self._send_json(results)
                        return

                    elif "scan_id" in req_data:
                        results = DATASET_MGR.analyze_scan_by_id(req_data["scan_id"])
                        self._send_json(results)
                        return

                    else:
                        self._send_error_json("Missing 'image' base64, 'relative_path', or 'scan_id'")
                        return

                elif "multipart/form-data" in content_type:
                    environ = {
                        'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': self.headers['Content-Type'],
                        'CONTENT_LENGTH': self.headers['Content-Length']
                    }
                    form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
                    if "file" not in form:
                        self._send_error_json("No 'file' field found in form upload")
                        return

                    file_item = form["file"]
                    file_bytes = file_item.file.read()
                    filename = file_item.filename or "uploaded_retina.jpg"

                    from PIL import Image
                    pil_img = Image.open(io.BytesIO(file_bytes))
                    results = AI_ENGINE.analyze_image(pil_img)
                    results["filename"] = filename
                    self._send_json(results)
                    return

                else:
                    file_bytes = self.rfile.read(content_length)
                    from PIL import Image
                    pil_img = Image.open(io.BytesIO(file_bytes))
                    results = AI_ENGINE.analyze_image(pil_img)
                    self._send_json(results)
                    return

            except Exception as e:
                traceback.print_exc()
                self._send_error_json(f"Analysis failed: {str(e)}", status_code=500)
                return

        # -------------------------------------------------------------
        # 2. Dataset Management Endpoints
        # -------------------------------------------------------------
        if path == "/api/datasets/upload":
            try:
                environ = {
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': self.headers['Content-Type'],
                    'CONTENT_LENGTH': self.headers['Content-Length']
                }
                form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
                if "file" not in form:
                    self._send_error_json("No 'file' field provided in upload")
                    return

                file_item = form["file"]
                dataset_name = form.getvalue("name", "Custom Dataset")
                task_type = form.getvalue("task_type", "disease_classification")
                description = form.getvalue("description", "")
                
                zip_bytes = file_item.file.read()
                new_dataset = DATASET_HUB.import_zip_dataset(zip_bytes, dataset_name=dataset_name, task_type=task_type, description=description)
                self._send_json({"status": "success", "dataset": new_dataset})
                return
            except Exception as e:
                traceback.print_exc()
                self._send_error_json(f"Dataset upload failed: {str(e)}", status_code=500)
                return

        if path.startswith("/api/datasets/") and path.endswith("/import-csv"):
            dataset_id = path.split("/")[3]
            try:
                body = self.rfile.read(content_length).decode("utf-8")
                res = DATASET_HUB.import_csv_labels(dataset_id, body)
                self._send_json(res)
                return
            except Exception as e:
                self._send_error_json(str(e), status_code=500)
                return

        # Backward-compatible scan item analyze
        if path == "/api/dataset/analyze-item":
            try:
                body = self.rfile.read(content_length).decode("utf-8")
                req_data = json.loads(body)
                scan_id = req_data.get("scan_id")
                if not scan_id:
                    self._send_error_json("Missing scan_id")
                    return
                results = DATASET_MGR.analyze_scan_by_id(scan_id)
                self._send_json(results)
                return
            except Exception as e:
                self._send_error_json(str(e), status_code=500)
                return

        # -------------------------------------------------------------
        # 3. Model Training & Evaluation Endpoints
        # -------------------------------------------------------------
        if path == "/api/training/start":
            try:
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                config = json.loads(body) if body else {}
                res = TRAINING_PIPELINE.start_training(config)
                self._send_json(res)
                return
            except Exception as e:
                self._send_error_json(str(e), status_code=500)
                return

        if path == "/api/training/pause":
            res = TRAINING_PIPELINE.pause_training()
            self._send_json(res)
            return

        if path == "/api/training/resume":
            res = TRAINING_PIPELINE.resume_training()
            self._send_json(res)
            return

        if path == "/api/training/stop":
            res = TRAINING_PIPELINE.stop_training()
            self._send_json(res)
            return

        if path == "/api/evaluation/run":
            try:
                body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
                req = json.loads(body) if body else {}
                model_id = req.get("model_id", "ocucardio_multitask_resunet_v1")
                dataset_id = req.get("dataset_id", "drive")
                architecture = req.get("architecture", "multitask_resunet")
                
                results = EVALUATION_PIPELINE.evaluate_model_on_test_set(
                    model_id=model_id, dataset_id=dataset_id, architecture=architecture
                )
                self._send_json({"status": "success", "evaluation": results})
                return
            except Exception as e:
                traceback.print_exc()
                self._send_error_json(f"Evaluation failed: {str(e)}", status_code=500)
                return

        # -------------------------------------------------------------
        # 4. Model Selection Endpoint
        # -------------------------------------------------------------
        if path == "/api/models/select":
            try:
                body = self.rfile.read(content_length).decode("utf-8")
                req = json.loads(body)
                model_id = req.get("model_id")
                if not model_id:
                    self._send_error_json("Missing model_id")
                    return
                res = MODEL_REGISTRY.set_active_model(model_id)
                self._send_json(res)
                return
            except Exception as e:
                self._send_error_json(str(e), status_code=500)
                return

        self.send_error(404, f"Endpoint not found: {path}")

    def do_DELETE(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path.rstrip("/")
        if path.startswith("/api/datasets/"):
            dataset_id = path.split("/")[3]
            try:
                res = DATASET_HUB.delete_dataset(dataset_id)
                self._send_json(res)
                return
            except Exception as e:
                self._send_error_json(str(e), status_code=400)
                return
        self.send_error(404, "Endpoint not found")

def run_server(port=8080):
    httpd = None
    actual_port = port
    for p in range(port, port + 15):
        try:
            server_address = ("127.0.0.1", p)
            httpd = ThreadingHTTPServer(server_address, OcuCardioHandler)
            actual_port = p
            break
        except OSError:
            continue

    if httpd is None:
        print(f"[!] Error: Could not bind to any port from {port} to {port+14}")
        return

    try:
        with open(os.path.join(PROJECT_ROOT, ".active_port"), "w") as f:
            f.write(str(actual_port))
    except Exception:
        pass

    print("=" * 70)
    print(f"[*] OCUCARDIO AI PLATFORM SERVER RUNNING")
    print(f"[*] Local URL: http://localhost:{actual_port}")
    print(f"[*] Modules: Scan Inspector, Dataset Hub, Model Training, Benchmarks")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down server gracefully...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
