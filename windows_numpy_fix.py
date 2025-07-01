#!/usr/bin/env python3
"""
Windows-Specific NumPy DLL Fix
Addresses DLL load failures with multiple strategies
"""
import os
import sys
import subprocess
import platform

def check_system_info():
    """Display system information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.version()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")

def run_cmd(cmd):
    """Run command and show output"""
    print(f"\n> {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def method_1_pip_binary():
    """Method 1: Force binary installation"""
    print("\n" + "="*60)
    print("METHOD 1: Pip Binary Installation")
    print("="*60)
    
    # Uninstall first
    run_cmd("pip uninstall -y numpy scipy pandas scikit-learn shap matplotlib seaborn plotly optuna")
    run_cmd("pip cache purge")
    
    # Install with binary-only flags
    success = run_cmd("pip install --only-binary=all numpy==1.26.4 --force-reinstall --no-cache-dir")
    if success:
        success = run_cmd("pip install --only-binary=all scipy pandas scikit-learn --no-cache-dir")
        if success:
            success = run_cmd("pip install shap optuna matplotlib seaborn plotly --no-cache-dir")
    
    return success

def method_2_conda():
    """Method 2: Use conda if available"""
    print("\n" + "="*60)
    print("METHOD 2: Conda Installation")
    print("="*60)
    
    # Check if conda is available
    try:
        result = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print("Conda not available, skipping this method")
            return False
    except:
        print("Conda not available, skipping this method")
        return False
    
    print("Conda detected, using conda for installation...")
    
    # Use conda to install NumPy and core packages
    success = run_cmd("conda install -c conda-forge numpy=1.26.4 -y")
    if success:
        success = run_cmd("conda install -c conda-forge scipy pandas scikit-learn -y")
        if success:
            # Use pip for packages not available in conda
            success = run_cmd("pip install shap optuna --no-cache-dir")
    
    return success

def method_3_precompiled_wheels():
    """Method 3: Download precompiled wheels from specific sources"""
    print("\n" + "="*60)
    print("METHOD 3: Precompiled Wheels")
    print("="*60)
    
    # Use wheels from specific sources known to work on Windows
    success = run_cmd("pip install -i https://pypi.org/simple/ numpy==1.26.4 --force-reinstall --no-cache-dir")
    if success:
        success = run_cmd("pip install scipy pandas scikit-learn --no-cache-dir")
        if success:
            success = run_cmd("pip install shap optuna matplotlib seaborn plotly --no-cache-dir")
    
    return success

def test_installation():
    """Test if the installation works"""
    print("\n" + "="*60)
    print("TESTING INSTALLATION")
    print("="*60)
    
    test_script = '''
import sys
try:
    print("Testing NumPy...")
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
    
    # Test the problematic import
    from numpy.linalg import _umath_linalg
    print("✓ _umath_linalg imported successfully")
    
    # Test basic operations
    arr = np.array([1, 2, 3, 4, 5])
    result = np.dot(arr, arr)
    print(f"✓ NumPy operations work: {result}")
    
    # Test linear algebra
    matrix = np.array([[1, 2], [3, 4]])
    det = np.linalg.det(matrix)
    print(f"✓ Linear algebra works: det={det}")
    
    print("\\nTesting SHAP...")
    import shap
    print(f"✓ SHAP {shap.__version__} imported successfully")
    
    print("\\n🎉 ALL TESTS PASSED!")
    sys.exit(0)
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    
    # Write test script to file
    with open("test_fix.py", "w") as f:
        f.write(test_script)
    
    # Run test
    try:
        result = subprocess.run(f"{sys.executable} test_fix.py", 
                              shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Test execution failed: {e}")
        return False

def main():
    """Main execution"""
    print("🔧 WINDOWS NUMPY DLL FIX")
    print("This script will try multiple methods to fix NumPy DLL issues")
    
    check_system_info()
    
    methods = [
        ("Binary Pip Installation", method_1_pip_binary),
        ("Conda Installation", method_2_conda),
        ("Precompiled Wheels", method_3_precompiled_wheels)
    ]
    
    for method_name, method_func in methods:
        print(f"\n🔧 Trying: {method_name}")
        try:
            success = method_func()
            if success:
                print(f"✓ {method_name} completed")
                
                # Test the installation
                if test_installation():
                    print(f"\n🎉 SUCCESS! {method_name} fixed the issue!")
                    print("\nYou can now run:")
                    print("  python ProjectP.py")
                    print("  Select Menu 1 - Full Pipeline")
                    return True
                else:
                    print(f"✗ {method_name} didn't fix the issue, trying next method...")
            else:
                print(f"✗ {method_name} failed, trying next method...")
        except Exception as e:
            print(f"✗ {method_name} crashed: {e}")
    
    print("\n❌ All methods failed!")
    print("\nMANUAL STEPS:")
    print("1. Install Anaconda/Miniconda")
    print("2. Create new environment: conda create -n projectp python=3.11")
    print("3. Activate: conda activate projectp")
    print("4. Install: conda install numpy scipy pandas scikit-learn")
    print("5. Install: pip install shap optuna")

if __name__ == "__main__":
    main()
