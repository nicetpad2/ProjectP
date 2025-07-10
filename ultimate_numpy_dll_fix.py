#!/usr/bin/env python3
"""
Ultimate NumPy DLL Fix Script for Windows
Addresses the '_umath_linalg' DLL load failure issue
"""
import sys
import os
import subprocess
import shutil
import importlib.util
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"RETURN CODE: {result.returncode}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def check_visual_cpp_redist():
    """Check if Visual C++ Redistributables are installed"""
    print("\n" + "="*60)
    print("CHECKING: Visual C++ Redistributables")
    print("="*60)
    
    # Check common registry locations for VC++ redist
    try:
        import winreg
        
        # Check for Visual Studio 2019/2022 redistributables
        keys_to_check = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"
        ]
        
        found_redist = False
        for key_path in keys_to_check:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                version = winreg.QueryValueEx(key, "Version")[0]
                print(f"✓ Found Visual C++ Redistributable: {version}")
                found_redist = True
                winreg.CloseKey(key)
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error checking {key_path}: {e}")
                continue
        
        if not found_redist:
            print("⚠️  Visual C++ Redistributables not found!")
            print("This may be the cause of the DLL load failure.")
            print("Please install Visual C++ Redistributables from:")
            print("https://aka.ms/vs/17/release/vc_redist.x64.exe")
            return False
        
        return True
        
    except ImportError:
        print("Cannot check registry (winreg not available)")
        return True
    except Exception as e:
        print(f"Error checking Visual C++ Redistributables: {e}")
        return True

def get_site_packages_path():
    """Get the site-packages directory path"""
    try:
        import site
        site_packages = site.getsitepackages()
        if site_packages:
            return site_packages[0]
        else:
            # Fallback method
            import numpy
            numpy_path = Path(numpy.__file__).parent.parent
            return str(numpy_path)
    except:
        # Another fallback
        python_path = Path(sys.executable).parent
        return str(python_path / "Lib" / "site-packages")

def complete_cleanup():
    """Complete cleanup of NumPy and related packages"""
    print("\n" + "="*60)
    print("PHASE 1: COMPLETE CLEANUP")
    print("="*60)
    
    # Step 1: Uninstall packages
    packages_to_remove = [
        'numpy', 'scipy', 'pandas', 'scikit-learn', 'shap', 
        'matplotlib', 'seaborn', 'plotly', 'optuna'
    ]
    
    for package in packages_to_remove:
        print(f"\nUninstalling {package}...")
        run_command(f"pip uninstall -y {package}", f"Uninstall {package}")
    
    # Step 2: Clear pip cache
    print("\nClearing pip cache...")
    run_command("pip cache purge", "Clear pip cache")
    
    # Step 3: Manual cleanup of site-packages
    site_packages = get_site_packages_path()
    print(f"\nCleaning site-packages: {site_packages}")
    
    cleanup_patterns = [
        'numpy*', 'scipy*', 'pandas*', 'sklearn*', 'scikit_learn*',
        'shap*', 'matplotlib*', 'seaborn*', 'plotly*', 'optuna*'
    ]
    
    for pattern in cleanup_patterns:
        try:
            import glob
            matches = glob.glob(os.path.join(site_packages, pattern))
            for match in matches:
                try:
                    if os.path.isdir(match):
                        shutil.rmtree(match)
                        print(f"Removed directory: {match}")
                    else:
                        os.remove(match)
                        print(f"Removed file: {match}")
                except Exception as e:
                    print(f"Could not remove {match}: {e}")
        except Exception as e:
            print(f"Error cleaning pattern {pattern}: {e}")

def install_dependencies():
    """Install dependencies with specific versions and strategies"""
    print("\n" + "="*60)
    print("PHASE 2: STRATEGIC INSTALLATION")
    print("="*60)
    
    # Strategy 1: Install NumPy first with specific version
    print("\nInstalling NumPy 1.26.4 (SHAP compatible)...")
    success = run_command(
        "pip install numpy==1.26.4 --no-cache-dir --force-reinstall",
        "Install NumPy 1.26.4"
    )
    
    if not success:
        print("Trying alternative NumPy installation...")
        success = run_command(
            "pip install numpy==1.24.3 --no-cache-dir --force-reinstall",
            "Install NumPy 1.24.3 (alternative)"
        )
    
    if not success:
        print("Trying pre-compiled wheel...")
        success = run_command(
            "pip install --only-binary=numpy numpy --no-cache-dir --force-reinstall",
            "Install NumPy pre-compiled wheel"
        )
    
    # Test NumPy after installation
    test_numpy()
    
    # Install other dependencies
    dependencies = [
        "scipy>=1.9.0",
        "pandas>=1.5.0", 
        "scikit-learn>=1.1.0",
        "shap>=0.41.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "optuna>=3.0.0"
    ]
    
    for dep in dependencies:
        print(f"\nInstalling {dep}...")
        run_command(
            f"pip install {dep} --no-cache-dir",
            f"Install {dep}"
        )

def test_numpy():
    """Test NumPy import and functionality"""
    print("\n" + "="*60)
    print("TESTING: NumPy Import and Functionality")
    print("="*60)
    
    try:
        import numpy as np
        print(f"✓ NumPy import successful! Version: {np.__version__}")
        print(f"✓ NumPy location: {np.__file__}")
        
        # Test the problematic import
        try:
            from numpy.linalg import _umath_linalg
            print("✓ _umath_linalg import successful!")
        except Exception as e:
            print(f"✗ _umath_linalg import failed: {e}")
            return False
        
        # Test basic operations
        arr = np.array([1, 2, 3, 4, 5])
        result = np.dot(arr, arr)
        print(f"✓ NumPy operations successful! Test result: {result}")
        
        # Test linear algebra
        matrix = np.array([[1, 2], [3, 4]])
        det = np.linalg.det(matrix)
        print(f"✓ NumPy linear algebra successful! Determinant: {det}")
        
        return True
        
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")
        return False

def test_shap():
    """Test SHAP import"""
    print("\n" + "="*60)
    print("TESTING: SHAP Import")
    print("="*60)
    
    try:
        import shap
        print(f"✓ SHAP import successful! Version: {shap.__version__}")
        return True
    except Exception as e:
        print(f"✗ SHAP import failed: {e}")
        return False

def run_system_validation():
    """Run the system validation script"""
    print("\n" + "="*60)
    print("RUNNING: System Validation")
    print("="*60)
    
    if os.path.exists("verify_system_ready.py"):
        run_command("python verify_system_ready.py", "System validation")
    else:
        print("System validation script not found")

def main():
    """Main execution function"""
    print("🔧 ULTIMATE NUMPY DLL FIX SCRIPT")
    print("=" * 60)
    print("This script will completely fix the NumPy DLL load failure issue.")
    print("This process may take several minutes...")
    print("=" * 60)
    
    # Check system prerequisites
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    
    # Check Visual C++ Redistributables
    check_visual_cpp_redist()
    
    # Phase 1: Complete cleanup
    complete_cleanup()
    
    # Phase 2: Strategic installation
    install_dependencies()
    
    # Phase 3: Testing
    print("\n" + "="*60)
    print("PHASE 3: COMPREHENSIVE TESTING")
    print("="*60)
    
    numpy_success = test_numpy()
    shap_success = test_shap()
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    if numpy_success and shap_success:
        print("🎉 SUCCESS! All dependencies are working correctly.")
        print("✓ NumPy DLL issue resolved")
        print("✓ SHAP import working")
        print("✓ System ready for Menu 1 execution")
        
        # Run system validation
        run_system_validation()
        
        print("\nYou can now run: python ProjectP.py")
        print("And select Menu 1 - Full Pipeline")
        
    else:
        print("❌ ISSUES DETECTED:")
        if not numpy_success:
            print("  - NumPy still has issues")
        if not shap_success:
            print("  - SHAP still has issues")
        
        print("\nMANUAL STEPS REQUIRED:")
        print("1. Install Visual C++ Redistributables:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("2. Consider using Anaconda/Miniconda instead of pip")
        print("3. Check Windows updates and system integrity")

if __name__ == "__main__":
    main()
