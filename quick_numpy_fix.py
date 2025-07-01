#!/usr/bin/env python3
"""
🔧 NICEGOLD Quick NumPy Compatibility Fix
========================================

Simple and effective fix for NumPy 2.x compatibility issues with SHAP.
Ensures NumPy 1.26.4 is properly installed for enterprise compliance.
"""

import subprocess
import sys
import os

def main():
    print("🔧 NICEGOLD NumPy Compatibility Quick Fix")
    print("=" * 45)
    
    # Set CPU-only environment
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Step 1: Force install correct NumPy version
    print("📦 Installing NumPy 1.26.4 for SHAP compatibility...")
    try:
        # Force reinstall NumPy 1.26.4
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "--force-reinstall", "--no-deps", "numpy==1.26.4"
        ], check=True, capture_output=True)
        print("✅ NumPy 1.26.4 installed")
    except Exception as e:
        print(f"❌ NumPy installation failed: {e}")
        return False
    
    # Step 2: Install other key packages
    print("📦 Installing core packages...")
    key_packages = [
        "pandas==2.2.3",
        "scikit-learn==1.5.2", 
        "shap==0.45.0",
        "optuna==3.5.0"
    ]
    
    for package in key_packages:
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True)
            print(f"✅ {package} installed")
        except Exception as e:
            print(f"⚠️  {package} install warning: {e}")
    
    # Step 3: Test compatibility
    print("\n🧪 Testing compatibility...")
    try:
        import numpy as np
        import shap
        print(f"✅ NumPy {np.__version__}")
        print(f"✅ SHAP {shap.__version__}")
        
        # Quick SHAP test
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=20, n_features=3, random_state=42)
        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:2])
        
        print("✅ SHAP functionality verified")
        print("🎉 NumPy compatibility fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready to run: python ProjectP.py")
    else:
        print("\n❌ Manual intervention required")
    sys.exit(0 if success else 1)
