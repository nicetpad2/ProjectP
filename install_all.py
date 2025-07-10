"""
NICEGOLD ENTERPRISE - Universal Auto-Installer (Python Fallback)
Usage: python install_all.py
"""
import os
import sys
import subprocess
import platform

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, '.venv')
REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, 'requirements.txt')

PYTHON_PACKAGES = [
    'tensorflow==2.19.0',
    'torch==2.7.1',
    'stable-baselines3==2.6.0',
    'gymnasium==1.1.1',
    'scikit-learn==1.7.0',
    'numpy==2.1.3',
    'pandas==2.3.0',
    'opencv-python-headless==4.11.0.0',
    'Pillow==11.2.1',
    'PyWavelets==1.8.0',
    'imbalanced-learn==0.13.0',
    'ta==0.11.0',
    'PyYAML==6.0.2',
    'shap==0.45.0',
    'optuna==3.5.0',
    'joblib',
]

def run(cmd, shell=False):
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=shell, check=True)
    return result

def main():
    # 1. Check Python version
    if sys.version_info < (3, 8):
        print(f"[ERROR] Python 3.8+ required, found {sys.version}")
        sys.exit(1)
    print(f"[INFO] Python version: {platform.python_version()}")

    # 2. Create virtual environment if not exists
    if not os.path.isdir(VENV_DIR):
        print(f"[INFO] Creating virtual environment in {VENV_DIR} ...")
        run([sys.executable, '-m', 'venv', VENV_DIR])
    else:
        print("[INFO] Virtual environment already exists.")

    # 3. Activate virtual environment
    if platform.system() == 'Windows':
        activate_script = os.path.join(VENV_DIR, 'Scripts', 'activate_this.py')
    else:
        activate_script = os.path.join(VENV_DIR, 'bin', 'activate_this.py')
    if os.path.exists(activate_script):
        with open(activate_script) as f:
            exec(f.read(), {'__file__': activate_script})
        print("[INFO] Virtual environment activated.")
    else:
        print("[WARNING] Could not auto-activate venv. Please activate manually.")

    # 4. Upgrade pip
    run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

    # 5. Install requirements.txt
    if os.path.isfile(REQUIREMENTS_FILE):
        print(f"[INFO] Installing requirements from {REQUIREMENTS_FILE} ...")
        run([sys.executable, '-m', 'pip', 'install', '-r', REQUIREMENTS_FILE])
    else:
        print("[WARNING] requirements.txt not found. Skipping.")

    # 6. Install core ML/AI packages
    for pkg in PYTHON_PACKAGES:
        try:
            run([sys.executable, '-m', 'pip', 'install', pkg])
        except Exception as e:
            print(f"[WARNING] Could not install {pkg}: {e}")

    print("[SUCCESS] All dependencies installed.")
    print("[INFO] To activate the environment:")
    if platform.system() == 'Windows':
        print(f"  .\\.venv\\Scripts\\Activate.ps1")
    else:
        print(f"  source .venv/bin/activate")
    print("[INFO] To run the main program:")
    print("  python ProjectP.py")

if __name__ == '__main__':
    main()
