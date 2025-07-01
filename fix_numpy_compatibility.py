#!/usr/bin/env python3
"""
NICEGOLD ProjectP - NumPy Compatibility Fix
==========================================
🔧 Fixes NumPy 2.x incompatibility with SHAP and other ML libraries
📋 Downgrades NumPy to 1.26.4 and reinstalls all compatible packages

This script:
1. Uninstalls problematic NumPy 2.x version
2. Installs NumPy 1.26.4 (SHAP compatible)
3. Reinstalls all dependencies from requirements.txt
4. Verifies all imports work correctly
5. Tests critical functionality

Author: NICEGOLD Enterprise
Date: 2025-01-07
Status: Production Ready
"""

import subprocess
import sys
import os
import json
from datetime import datetime


def run_command(cmd, description):
    """Run a command and capture output"""
    print(f"\n🔄 {description}")
    print(f"   Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()[:200]}...")
        else:
            print(f"   ⚠️ Warning/Error: {result.stderr.strip()[:200]}...")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")
        return False, "", str(e)


def check_numpy_version():
    """Check current NumPy version"""
    try:
        import numpy as np
        version = np.__version__
        print(f"📊 Current NumPy version: {version}")
        return version
    except ImportError:
        print("❌ NumPy not installed")
        return None


def test_problematic_imports():
    """Test imports that were failing"""
    failed_imports = []
    
    test_imports = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__} OK')"),
        ("pandas", "import pandas as pd; print(f'Pandas {pd.__version__} OK')"),
        ("sklearn", "import sklearn; print(f'Scikit-learn {sklearn.__version__} OK')"),
        ("shap", "import shap; print(f'SHAP {shap.__version__} OK')"),
        ("tensorflow", "import tensorflow as tf; print(f'TensorFlow {tf.__version__} OK')"),
        ("torch", "import torch; print(f'PyTorch {torch.__version__} OK')"),
        ("optuna", "import optuna; print(f'Optuna {optuna.__version__} OK')"),
    ]
    
    print("\n🧪 Testing critical imports...")
    for name, import_cmd in test_imports:
        try:
            exec(import_cmd)
            print(f"   ✅ {name} import successful")
        except Exception as e:
            print(f"   ❌ {name} import failed: {str(e)[:100]}")
            failed_imports.append((name, str(e)))
    
    return failed_imports


def test_shap_functionality():
    """Test SHAP functionality specifically"""
    print("\n🎯 Testing SHAP functionality...")
    try:
        import shap
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        
        # Simple test data
        X = np.random.random((10, 5))
        y = np.random.random(10)
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:3])
        
        print(f"   ✅ SHAP TreeExplainer working (output shape: {shap_values.shape})")
        return True
        
    except Exception as e:
        print(f"   ❌ SHAP functionality test failed: {str(e)}")
        return False


def main():
    """Main fix process"""
    print("=" * 60)
    print("🚀 NICEGOLD ProjectP - NumPy Compatibility Fix")
    print("=" * 60)
    
    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(script_dir, "requirements.txt")
    
    print(f"📂 Working directory: {script_dir}")
    print(f"📄 Requirements file: {requirements_path}")
    
    # Check current state
    print("\n📋 STEP 1: Current Environment Analysis")
    check_numpy_version()
    initial_failures = test_problematic_imports()
    
    if not initial_failures:
        print("✅ All imports already working! No fix needed.")
        test_shap_functionality()
        return
    
    print(f"\n⚠️ Found {len(initial_failures)} failed imports:")
    for name, error in initial_failures:
        print(f"   - {name}: {error[:100]}...")
    
    # Uninstall problematic packages
    print("\n📋 STEP 2: Uninstalling Problematic Packages")
    packages_to_uninstall = ["numpy", "pandas", "scikit-learn", "shap", "scipy"]
    
    for package in packages_to_uninstall:
        run_command(f"pip uninstall -y {package}", f"Uninstalling {package}")
    
    # Install NumPy 1.26.4 first
    print("\n📋 STEP 3: Installing Compatible NumPy")
    run_command("pip install numpy==1.26.4", "Installing NumPy 1.26.4")
    
    # Verify NumPy installation
    new_numpy_version = check_numpy_version()
    if not new_numpy_version or not new_numpy_version.startswith("1.26"):
        print("❌ Failed to install NumPy 1.26.4")
        return
    
    # Install all requirements
    print("\n📋 STEP 4: Installing All Requirements")
    if os.path.exists(requirements_path):
        run_command(f"pip install -r \"{requirements_path}\"", "Installing from requirements.txt")
    else:
        print(f"❌ Requirements file not found: {requirements_path}")
        return
    
    # Final verification
    print("\n📋 STEP 5: Final Verification")
    final_failures = test_problematic_imports()
    shap_success = test_shap_functionality()
    
    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "numpy_version": check_numpy_version(),
        "initial_failures": len(initial_failures),
        "final_failures": len(final_failures),
        "shap_working": shap_success,
        "success": len(final_failures) == 0 and shap_success
    }
    
    # Save report
    report_path = os.path.join(script_dir, "numpy_fix_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 Fix Report saved to: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    if report["success"]:
        print("🎉 SUCCESS: NumPy compatibility fix completed!")
        print("✅ All imports working")
        print("✅ SHAP functionality verified")
        print("🚀 ProjectP is ready for production!")
    else:
        print("⚠️ PARTIAL SUCCESS: Some issues remain")
        if final_failures:
            print(f"❌ {len(final_failures)} imports still failing")
        if not shap_success:
            print("❌ SHAP functionality not working")
    print("=" * 60)


if __name__ == "__main__":
    main()
