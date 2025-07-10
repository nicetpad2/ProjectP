#!/usr/bin/env python3
"""
🔧 NICEGOLD NumPy DLL Fix - Complete Solution
===========================================

This script fixes NumPy DLL loading issues on Windows by:
1. Completely removing corrupted NumPy installation
2. Installing clean NumPy 1.26.4 from scratch
3. Installing all dependencies in correct order
4. Verifying the fix works
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"   ✅ Success")
            return True
        else:
            print(f"   ⚠️  Warning: {result.stderr[:100]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"   ⚠️  Timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def clean_python_cache():
    """Clean Python cache and compiled files"""
    print("🧹 Cleaning Python cache...")
    
    # Remove __pycache__ directories
    for pycache in Path('.').rglob('__pycache__'):
        try:
            shutil.rmtree(pycache)
            print(f"   Removed: {pycache}")
        except:
            pass
    
    # Remove .pyc files
    for pyc in Path('.').rglob('*.pyc'):
        try:
            pyc.unlink()
        except:
            pass
    
    print("   ✅ Cache cleaned")

def fix_numpy_dll_issue():
    """Complete NumPy DLL fix"""
    print("🚀 NICEGOLD NumPy DLL Fix - Starting...")
    print("=" * 50)
    
    success_count = 0
    total_steps = 8
    
    # Step 1: Clean Python cache
    clean_python_cache()
    success_count += 1
    
    # Step 2: Uninstall NumPy completely
    if run_command("pip uninstall numpy -y", "Uninstalling NumPy"):
        success_count += 1
    
    # Step 3: Clear pip cache
    if run_command("pip cache purge", "Clearing pip cache"):
        success_count += 1
    
    # Step 4: Install NumPy 1.26.4 (SHAP compatible)
    if run_command("pip install numpy==1.26.4 --no-cache-dir --force-reinstall", 
                   "Installing NumPy 1.26.4"):
        success_count += 1
    
    # Step 5: Install core dependencies
    core_packages = [
        "pandas==2.2.3",
        "scikit-learn==1.5.2"
    ]
    
    for package in core_packages:
        if run_command(f"pip install {package} --no-cache-dir", f"Installing {package}"):
            success_count += 0.5
    
    # Step 6: Install SHAP (the problematic package)
    if run_command("pip install shap==0.45.0 --no-cache-dir", "Installing SHAP"):
        success_count += 1
    
    # Step 7: Test NumPy functionality
    print("🧪 Testing NumPy functionality...")
    try:
        test_script = '''
import numpy as np
import sys
print(f"NumPy version: {np.__version__}")
print(f"NumPy path: {np.__file__}")

# Test basic operations
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
print(f"Sum: {np.sum(arr)}")

# Test linalg (the problematic module)
matrix = np.array([[1, 2], [3, 4]])
determinant = np.linalg.det(matrix)
print(f"Matrix determinant: {determinant}")

print("✅ NumPy test successful!")
'''
        
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ NumPy working correctly!")
            print(f"   Output: {result.stdout}")
            success_count += 1
        else:
            print("   ❌ NumPy test failed!")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"   ❌ NumPy test error: {e}")
    
    # Step 8: Test SHAP functionality
    print("🔬 Testing SHAP functionality...")
    try:
        shap_test = '''
import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

print(f"SHAP version: {shap.__version__}")

# Quick SHAP test
X, y = make_regression(n_samples=20, n_features=3, random_state=42)
model = RandomForestRegressor(n_estimators=3, random_state=42)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[:2])

print(f"SHAP values shape: {shap_values.shape}")
print("✅ SHAP test successful!")
'''
        
        result = subprocess.run([sys.executable, '-c', shap_test],
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ SHAP working correctly!")
            success_count += 1
        else:
            print("   ❌ SHAP test failed!")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"   ❌ SHAP test error: {e}")
    
    # Final status
    print("\n" + "=" * 50)
    print(f"📊 Fix Progress: {success_count}/{total_steps} steps completed")
    
    if success_count >= 7:  # Almost all steps successful
        print("🎉 NumPy DLL fix completed successfully!")
        print("✅ System ready for production")
        print("\n🚀 Next steps:")
        print("   1. Run: python ProjectP.py")
        print("   2. Test Menu 1 (Full Pipeline)")
        return True
    elif success_count >= 5:  # Partial success
        print("⚠️  NumPy fix partially successful")
        print("💡 Manual intervention may be required")
        print("\n🔧 Try these steps:")
        print("   1. Restart terminal/IDE")
        print("   2. Run this fix script again")
        print("   3. Check Windows Visual C++ Redistributables")
        return False
    else:
        print("❌ NumPy fix failed")
        print("🆘 Critical issues detected")
        print("\n🔧 Manual fixes required:")
        print("   1. Check Python installation")
        print("   2. Reinstall Visual C++ Redistributables")
        print("   3. Consider using Anaconda/Miniconda")
        return False

def main():
    """Main fix routine"""
    try:
        return fix_numpy_dll_issue()
    except KeyboardInterrupt:
        print("\n❌ Fix interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
