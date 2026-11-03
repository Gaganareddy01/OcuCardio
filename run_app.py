import os
import sys
import time
import webbrowser
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")
PYTHON_BIN = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

def main():
    print("=" * 70)
    print("  OCUCARDIO: AI-POWERED RETINAL ANALYSIS & CVD RISK ASSESSMENT PLATFORM")
    print("=" * 70)
    print(f"[*] Python Executable: {PYTHON_BIN}")
    print("[*] Launching multi-threaded AI server on port 8080...")
    
    server_script = os.path.join(PROJECT_DIR, "backend", "server.py")
    proc = subprocess.Popen([PYTHON_BIN, "-u", server_script, "8080"])
    
    time.sleep(3)
    url = "http://localhost:8080"
    print(f"[*] Opening browser to: {url}")
    webbrowser.open(url)
    
    print("[*] Press Ctrl+C in this terminal to stop the server.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()

