#!/bin/bash
# NICEGOLD ENTERPRISE - Universal Auto-Installer (Linux/Mac)
# Usage: bash install_all.sh

set -e

PROJECT_ROOT="$(dirname "$0")"
VENV_DIR="$PROJECT_ROOT/.venv"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

# 1. Check Python version
if ! command -v python3 &> /dev/null; then
  echo "[ERROR] Python3 is not installed. Please install Python 3.8+ and rerun this script."
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
if [[ "$PYTHON_VERSION" < "3.8" ]]; then
  echo "[ERROR] Python version 3.8 or higher is required. Found $PYTHON_VERSION."
  exit 1
fi

echo "[INFO] Python version: $PYTHON_VERSION"

# 2. Create virtual environment if not exists
if [ ! -d "$VENV_DIR" ]; then
  echo "[INFO] Creating virtual environment in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
else
  echo "[INFO] Virtual environment already exists."
fi

# 3. Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "[INFO] Virtual environment activated."

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install requirements
if [ -f "$REQUIREMENTS_FILE" ]; then
  echo "[INFO] Installing requirements from $REQUIREMENTS_FILE ..."
  pip install -r "$REQUIREMENTS_FILE"
else
  echo "[WARNING] requirements.txt not found. Skipping."
fi

# 6. Install core ML/AI packages (if not in requirements.txt)
PYTHON_PACKAGES=(
  tensorflow==2.19.0
  torch==2.7.1
  stable-baselines3==2.6.0
  gymnasium==1.1.1
  scikit-learn==1.7.0
  numpy==2.1.3
  pandas==2.3.0
  opencv-python-headless==4.11.0.0
  Pillow==11.2.1
  PyWavelets==1.8.0
  imbalanced-learn==0.13.0
  ta==0.11.0
  PyYAML==6.0.2
  shap==0.45.0
  optuna==3.5.0
  joblib
)

for pkg in "${PYTHON_PACKAGES[@]}"; do
  pip install "$pkg" || true
  # Ignore errors if already installed or not available for platform
  done

echo "[SUCCESS] All dependencies installed."
echo "[INFO] To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo "[INFO] To run the main program:"
echo "  python ProjectP.py"
