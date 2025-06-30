# NICEGOLD ENTERPRISE - Universal Auto-Installer (Windows PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File install_all.ps1

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvDir = Join-Path $projectRoot ".venv"
$requirementsFile = Join-Path $projectRoot "requirements.txt"

# 1. Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed. Please install Python 3.8+ and rerun this script."
    exit 1
}

$pythonVersion = & python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ([version]$pythonVersion -lt [version]'3.8') {
    Write-Error "Python version 3.8 or higher is required. Found $pythonVersion."
    exit 1
}
Write-Host "[INFO] Python version: $pythonVersion"

# 2. Create virtual environment if not exists
if (-not (Test-Path $venvDir)) {
    Write-Host "[INFO] Creating virtual environment in $venvDir ..."
    python -m venv $venvDir
} else {
    Write-Host "[INFO] Virtual environment already exists."
}

# 3. Activate virtual environment
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
. $activateScript
Write-Host "[INFO] Virtual environment activated."

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install requirements
if (Test-Path $requirementsFile) {
    Write-Host "[INFO] Installing requirements from $requirementsFile ..."
    pip install -r $requirementsFile
} else {
    Write-Host "[WARNING] requirements.txt not found. Skipping."
}

# 6. Install core ML/AI packages (if not in requirements.txt)
$pythonPackages = @(
    "tensorflow==2.19.0",
    "torch==2.7.1",
    "stable-baselines3==2.6.0",
    "gymnasium==1.1.1",
    "scikit-learn==1.7.0",
    "numpy==2.1.3",
    "pandas==2.3.0",
    "opencv-python-headless==4.11.0.0",
    "Pillow==11.2.1",
    "PyWavelets==1.8.0",
    "imbalanced-learn==0.13.0",
    "ta==0.11.0",
    "PyYAML==6.0.2",
    "shap==0.45.0",
    "optuna==3.5.0",
    "joblib"
)
foreach ($pkg in $pythonPackages) {
    try {
        pip install $pkg
    } catch {
        Write-Host "[WARNING] Could not install $pkg (may already be installed or not available for this platform)"
    }
}

Write-Host "[SUCCESS] All dependencies installed."
Write-Host "[INFO] To activate the environment:"
Write-Host "  .\$venvDir\Scripts\Activate.ps1"
Write-Host "[INFO] To run the main program:"
Write-Host "  python ProjectP.py"
