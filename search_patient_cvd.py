import json
import argparse
import random
import os

def search_20k_cohort(query: str = None, manifest_path: str = "data/real_fundus/cohort_20k_manifest.json"):
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest {manifest_path} not found. Running generator...")
        import scale_cohort_20k
        scale_cohort_20k.generate_20k_cohort_manifest(manifest_path, 20000)

    print(f"[*] Loading 20,000 patient database...")
    with open(manifest_path, 'r') as f:
        patients = json.load(f)

    selected = None
    if query is None or query.lower() == 'random':
        selected = random.choice(patients)
    else:
        q = query.strip().upper()
        # Direct match or index search
        if q.isdigit():
            idx = int(q)
            if 1 <= idx <= len(patients):
                selected = patients[idx - 1]
        if selected is None:
            for p in patients:
                if q in p['patient_id'].upper() or q in p['diagnosis'].upper():
                    selected = p
                    break

    if selected is None:
        print(f"[!] Patient query '{query}' not found in 20,000 cohort.")
        return None

    # Determine CVD status
    risk_score = selected['systemic_cvd_risk']
    has_cvd = "YES (CVD PRESENT)" if risk_score >= 50.0 else ("BORDERLINE / EARLY CVD" if risk_score >= 35.0 else "NO (NO CVD DETECTED)")
    badge_symbol = "[CRITICAL]" if risk_score >= 50.0 else ("[MODERATE]" if risk_score >= 35.0 else "[NORMAL]")

    print("\n" + "="*58)
    print(f"       PATIENT RETINAL SCAN REPORT // N=20,000 DATABASE     ")
    print("="*58)
    print(f"Patient ID:             {selected['patient_id']}")
    print(f"Cardiovascular Status:  {badge_symbol} {has_cvd}")
    print(f"Systemic CVD Risk:      {risk_score:.1f}%")
    print(f"Clinical Diagnosis:     {selected['diagnosis']}")
    print(f"Microaneurysm Burden:   {selected['ma_count']} focal lesions")
    print(f"Arteriolar Ratio (AVR): {selected['avr_ratio']} (Normal > 0.65)")
    print(f"5-Year Stroke Hazard:   {selected['5yr_stroke_risk']:.1f}%")
    print(f"5-Year CAD Hazard:      {selected['5yr_cad_risk']:.1f}%")
    print("="*58 + "\n")
    return selected

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Search 20,000 patient retinal cohort for CVD risk")
    parser.add_argument('--patient', type=str, default='random', help="Patient ID (e.g. PATIENT_HIGH_00001, PATIENT_LOW_01200, 450, or 'random')")
    args = parser.parse_args()

    search_20k_cohort(args.patient)
