#!/usr/bin/env python3
"""
🔧 ULTIMATE NUMPY DLL FIX FOR WINDOWS
Enterprise-Grade Solution using Anaconda/Miniconda

This script implements the most reliable solution for NumPy DLL load failures on Windows.
It downloads and installs Miniconda, creates an isolated environment, and ensures
100% compatibility with SHAP and all enterprise dependencies.
"""

import os
import sys
import subprocess
import urllib.request
import shutil
import platform
from pathlib import Path
import json
import time

class WindowsNumPyFixer:
    """Ultimate NumPy DLL fix using Anaconda/Miniconda"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.miniconda_installer = "Miniconda3-latest-Windows-x86_64.exe"
        self.miniconda_url = f"https://repo.anaconda.com/miniconda/{self.miniconda_installer}"
        self.conda_env_name = "nicegold_enterprise"
        self.conda_path = None
        self.python_version = "3.11"
        
    def log(self, message):
        """Log with timestamp"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def run_command(self, cmd, description="", shell=True, check=False):
        """Run command with comprehensive error handling"""
        self.log(f"Running: {description}")
        self.log(f"Command: {cmd}")
        
        try:
            if shell:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
            if result.stdout:
                self.log(f"STDOUT: {result.stdout}")
            if result.stderr:
                self.log(f"STDERR: {result.stderr}")
                
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
                
            return result.returncode == 0, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            self.log("ERROR: Command timed out")
            return False, "", "Command timed out"
        except Exception as e:
            self.log(f"ERROR: {e}")
            return False, "", str(e)
    
    def check_system_requirements(self):
        """Check Windows system requirements"""
        self.log("🔍 Checking System Requirements")
        self.log("=" * 50)
        
        # Check OS
        if platform.system() != "Windows":
            self.log("❌ This fix is designed for Windows only")
            return False
            
        # Check architecture
        arch = platform.architecture()[0]
        if arch != "64bit":
            self.log("❌ 64-bit Windows required")
            return False
            
        # Check available disk space (need at least 2GB)
        free_space = shutil.disk_usage(self.project_root).free
        if free_space < 2 * 1024 * 1024 * 1024:  # 2GB
            self.log("❌ Insufficient disk space (need at least 2GB)")
            return False
            
        self.log("✅ System requirements met")
        return True
    
    def download_miniconda(self):
        """Download Miniconda installer"""
        self.log("📥 Downloading Miniconda")
        self.log("=" * 50)
        
        installer_path = self.project_root / self.miniconda_installer
        
        if installer_path.exists():
            self.log("✅ Miniconda installer already exists")
            return True
            
        try:
            self.log(f"Downloading from: {self.miniconda_url}")
            urllib.request.urlretrieve(self.miniconda_url, installer_path)
            self.log(f"✅ Downloaded: {installer_path}")
            return True
            
        except Exception as e:
            self.log(f"❌ Download failed: {e}")
            return False
    
    def install_miniconda(self):
        """Install Miniconda silently"""
        self.log("🔧 Installing Miniconda")
        self.log("=" * 50)
        
        installer_path = self.project_root / self.miniconda_installer
        install_dir = self.project_root / "miniconda3"
        
        if install_dir.exists():
            self.log("✅ Miniconda already installed")
            self.conda_path = install_dir / "Scripts" / "conda.exe"
            return True
            
        # Silent installation command
        install_cmd = f'"{installer_path}" /InstallationType=JustMe /RegisterPython=0 /S /D={install_dir}'
        
        success, stdout, stderr = self.run_command(install_cmd, "Installing Miniconda")
        
        if success or install_dir.exists():
            self.conda_path = install_dir / "Scripts" / "conda.exe"
            self.log(f"✅ Miniconda installed: {install_dir}")
            
            # Clean up installer
            try:
                installer_path.unlink()
                self.log("🗑️ Cleaned up installer")
            except:
                pass
                
            return True
        else:
            self.log("❌ Miniconda installation failed")
            return False
    
    def create_conda_environment(self):
        """Create conda environment with Python"""
        self.log("🐍 Creating Conda Environment")
        self.log("=" * 50)
        
        if not self.conda_path or not self.conda_path.exists():
            self.log("❌ Conda not found")
            return False
            
        # Check if environment already exists
        env_check_cmd = f'"{self.conda_path}" env list'
        success, stdout, stderr = self.run_command(env_check_cmd, "Checking environments")
        
        if success and self.conda_env_name in stdout:
            self.log("✅ Environment already exists")
            return True
            
        # Create environment
        create_cmd = f'"{self.conda_path}" create -n {self.conda_env_name} python={self.python_version} -y'
        success, stdout, stderr = self.run_command(create_cmd, "Creating environment")
        
        if success:
            self.log(f"✅ Created environment: {self.conda_env_name}")
            return True
        else:
            self.log("❌ Environment creation failed")
            return False
    
    def install_packages_conda(self):
        """Install packages using conda"""
        self.log("📦 Installing Packages with Conda")
        self.log("=" * 50)
        
        # Core packages to install via conda (better compatibility)
        conda_packages = [
            "numpy=1.26.4",
            "scipy",
            "pandas", 
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "jupyter",
            "ipython"
        ]
        
        for package in conda_packages:
            install_cmd = f'"{self.conda_path}" install -n {self.conda_env_name} -c conda-forge {package} -y'
            success, stdout, stderr = self.run_command(install_cmd, f"Installing {package}")
            
            if success:
                self.log(f"✅ Installed: {package}")
            else:
                self.log(f"⚠️ Failed to install {package}, will try with pip")
    
    def install_packages_pip(self):
        """Install remaining packages with pip"""
        self.log("📦 Installing Packages with Pip")
        self.log("=" * 50)
        
        # Get conda environment python path
        miniconda_path = self.project_root / "miniconda3"
        env_python = miniconda_path / "envs" / self.conda_env_name / "python.exe"
        
        if not env_python.exists():
            self.log("❌ Environment Python not found")
            return False
            
        # Packages to install with pip
        pip_packages = [
            "shap==0.45.0",
            "optuna>=3.0.0",
            "plotly>=5.0.0",
            "joblib",
            "tqdm",
            "colorama",
            "rich"
        ]
        
        for package in pip_packages:
            install_cmd = f'"{env_python}" -m pip install {package}'
            success, stdout, stderr = self.run_command(install_cmd, f"Installing {package}")
            
            if success:
                self.log(f"✅ Installed: {package}")
            else:
                self.log(f"❌ Failed to install: {package}")
                
        return True
    
    def test_installation(self):
        """Test the installation thoroughly"""
        self.log("🧪 Testing Installation")
        self.log("=" * 50)
        
        miniconda_path = self.project_root / "miniconda3"
        env_python = miniconda_path / "envs" / self.conda_env_name / "python.exe"
        
        if not env_python.exists():
            self.log("❌ Environment Python not found")
            return False
            
        # Create test script
        test_script = '''
import sys
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    print("\\n=== Testing NumPy ===")
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    
    # Test the problematic import
    from numpy.linalg import _umath_linalg
    print("✅ _umath_linalg import successful!")
    
    # Test operations
    arr = np.array([1, 2, 3, 4, 5])
    result = np.dot(arr, arr)
    print(f"✅ NumPy operations: {result}")
    
    # Test linear algebra
    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    det = np.linalg.det(matrix)
    print(f"✅ Linear algebra: det={det:.2f}")
    
    print("\\n=== Testing SHAP ===")
    import shap
    print(f"SHAP version: {shap.__version__}")
    
    # Basic SHAP test
    from sklearn.ensemble import RandomForestRegressor
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[:5])
    print(f"✅ SHAP explainer: {shap_values.shape}")
    
    print("\\n🎉 ALL TESTS PASSED!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        test_file = self.project_root / "test_conda_env.py"
        with open(test_file, "w") as f:
            f.write(test_script)
            
        # Run test
        test_cmd = f'"{env_python}" "{test_file}"'
        success, stdout, stderr = self.run_command(test_cmd, "Testing installation")
        
        # Clean up test file
        try:
            test_file.unlink()
        except:
            pass
            
        return success
    
    def create_activation_scripts(self):
        """Create convenient activation scripts"""
        self.log("📝 Creating Activation Scripts")
        self.log("=" * 50)
        
        miniconda_path = self.project_root / "miniconda3"
        
        # Windows batch script
        batch_script = f"""@echo off
echo 🏢 NICEGOLD Enterprise Environment
echo Activating Conda Environment: {self.conda_env_name}
call "{miniconda_path}\\Scripts\\activate.bat" {self.conda_env_name}
echo ✅ Environment activated!
echo.
echo 🚀 You can now run:
echo   python ProjectP.py
echo.
cmd /k
"""
        
        batch_file = self.project_root / "activate_nicegold_env.bat"
        with open(batch_file, "w") as f:
            f.write(batch_script)
            
        # PowerShell script
        ps_script = f"""# NICEGOLD Enterprise Environment Activation
Write-Host "🏢 NICEGOLD Enterprise Environment" -ForegroundColor Green
Write-Host "Activating Conda Environment: {self.conda_env_name}" -ForegroundColor Yellow

& "{miniconda_path}\\Scripts\\conda.exe" activate {self.conda_env_name}

Write-Host "✅ Environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 You can now run:" -ForegroundColor Cyan
Write-Host "  python ProjectP.py" -ForegroundColor White
Write-Host ""
"""
        
        ps_file = self.project_root / "activate_nicegold_env.ps1"
        with open(ps_file, "w") as f:
            f.write(ps_script)
            
        # Python launcher script
        launcher_script = f'''#!/usr/bin/env python3
"""
🚀 NICEGOLD Enterprise Launcher
Automatically activates conda environment and runs ProjectP.py
"""
import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    miniconda_path = project_root / "miniconda3"
    env_python = miniconda_path / "envs" / "{self.conda_env_name}" / "python.exe"
    
    if not env_python.exists():
        print("❌ Conda environment not found!")
        print("Please run: python windows_ultimate_numpy_fix.py")
        sys.exit(1)
        
    print("🏢 NICEGOLD Enterprise - Starting with Conda Environment")
    print(f"Python: {{env_python}}")
    
    # Run ProjectP.py with conda environment
    try:
        subprocess.run([str(env_python), "ProjectP.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running ProjectP.py: {{e}}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
'''
        
        launcher_file = self.project_root / "run_nicegold.py"
        with open(launcher_file, "w") as f:
            f.write(launcher_script)
            
        self.log("✅ Created activation scripts:")
        self.log(f"  - {batch_file} (Windows batch)")
        self.log(f"  - {ps_file} (PowerShell)")
        self.log(f"  - {launcher_file} (Python launcher)")
    
    def update_project_config(self):
        """Update project configuration for conda environment"""
        self.log("⚙️ Updating Project Configuration")
        self.log("=" * 50)
        
        miniconda_path = self.project_root / "miniconda3"
        env_python = miniconda_path / "envs" / self.conda_env_name / "python.exe"
        
        # Create environment info file
        env_info = {
            "environment_type": "conda",
            "environment_name": self.conda_env_name,
            "python_executable": str(env_python),
            "conda_path": str(self.conda_path),
            "miniconda_path": str(miniconda_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active"
        }
        
        env_info_file = self.project_root / "conda_environment.json"
        with open(env_info_file, "w") as f:
            json.dump(env_info, f, indent=2)
            
        self.log(f"✅ Environment info saved: {env_info_file}")
        
        # Update VS Code settings if .vscode exists
        vscode_dir = self.project_root / ".vscode"
        if vscode_dir.exists():
            settings_file = vscode_dir / "settings.json"
            settings = {
                "python.defaultInterpreterPath": str(env_python).replace("\\", "/"),
                "python.terminal.activateEnvironment": True,
                "python.envFile": "${workspaceFolder}/.env"
            }
            
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=2)
                
            self.log("✅ Updated VS Code settings")
    
    def run_full_fix(self):
        """Run the complete fix process"""
        self.log("🔧 ULTIMATE NUMPY DLL FIX STARTING")
        self.log("=" * 80)
        self.log("This will install Miniconda and create an isolated environment")
        self.log("This process may take 10-15 minutes...")
        self.log("=" * 80)
        
        steps = [
            ("System Requirements", self.check_system_requirements),
            ("Download Miniconda", self.download_miniconda),
            ("Install Miniconda", self.install_miniconda),
            ("Create Environment", self.create_conda_environment),
            ("Install Core Packages", self.install_packages_conda),
            ("Install Additional Packages", self.install_packages_pip),
            ("Test Installation", self.test_installation),
            ("Create Scripts", self.create_activation_scripts),
            ("Update Configuration", self.update_project_config)
        ]
        
        for step_name, step_func in steps:
            self.log(f"\\n🔄 Step: {step_name}")
            if not step_func():
                self.log(f"❌ Failed at step: {step_name}")
                return False
            self.log(f"✅ Completed: {step_name}")
        
        self.log("\\n" + "=" * 80)
        self.log("🎉 ULTIMATE FIX COMPLETED SUCCESSFULLY!")
        self.log("=" * 80)
        self.log("\\n📋 NEXT STEPS:")
        self.log("1. Use one of these methods to activate the environment:")
        self.log("   • Double-click: activate_nicegold_env.bat")
        self.log("   • PowerShell: .\\activate_nicegold_env.ps1")
        self.log("   • Direct run: python run_nicegold.py")
        self.log("\\n2. Test the system:")
        self.log("   python ProjectP.py")
        self.log("   Select Menu 1 - Full Pipeline")
        self.log("\\n✨ NumPy DLL issue is now permanently resolved!")
        
        return True

def main():
    """Main execution"""
    print("🔧 ULTIMATE WINDOWS NUMPY DLL FIX")
    print("Enterprise-Grade Solution using Miniconda")
    print("=" * 60)
    
    fixer = WindowsNumPyFixer()
    
    try:
        success = fixer.run_full_fix()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
