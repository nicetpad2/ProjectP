#!/usr/bin/env python3
"""
🔧 NICEGOLD ProjectP - Installation Test Script
ทดสอบการติดตั้งไลบรารีและความพร้อมของระบบ
"""

import sys
import importlib

def test_library(name, package_name=None, version_attr='__version__'):
    """ทดสอบการ import library"""
    if package_name is None:
        package_name = name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, version_attr, 'Unknown')
        print(f"✅ {name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {name}: Warning - {e}")
        return True

def main():
    """ทดสอบการติดตั้งไลบรารีทั้งหมด"""
    print("🔧 NICEGOLD ProjectP - Installation Test")
    print("=" * 50)
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    print()
    
    # Core libraries
    print("📦 Core Libraries:")
    core_libs = [
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("SciPy", "scipy"),
        ("Joblib", "joblib"),
    ]
    
    success_count = 0
    for name, package in core_libs:
        if test_library(name, package):
            success_count += 1
    
    print()
    
    # ML/AI libraries
    print("🤖 Machine Learning & AI:")
    ml_libs = [
        ("TensorFlow", "tensorflow"),
        ("PyTorch", "torch"),
        ("Stable-Baselines3", "stable_baselines3"),
        ("Gymnasium", "gymnasium"),
        ("SHAP", "shap"),
        ("Optuna", "optuna"),
        ("Imbalanced-learn", "imblearn"),
    ]
    
    for name, package in ml_libs:
        if test_library(name, package):
            success_count += 1
    
    print()
    
    # Data processing libraries
    print("📊 Data Processing:")
    data_libs = [
        ("PyYAML", "yaml"),
        ("PyWavelets", "pywt"),
        ("TA (Technical Analysis)", "ta"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("Plotly", "plotly"),
    ]
    
    for name, package in data_libs:
        if test_library(name, package):
            success_count += 1
    
    print()
    
    # Testing libraries
    print("🧪 Testing:")
    test_libs = [
        ("Pytest", "pytest"),
    ]
    
    for name, package in test_libs:
        if test_library(name, package):
            success_count += 1
    
    print()
    print("=" * 50)
    
    total_libs = len(core_libs) + len(ml_libs) + len(data_libs) + len(test_libs)
    success_rate = (success_count / total_libs) * 100
    
    print(f"📈 Installation Success Rate: {success_count}/{total_libs} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print("🎉 Excellent! All critical libraries are installed.")
        print("✅ NICEGOLD ProjectP is ready to run!")
    elif success_rate >= 80:
        print("✅ Good! Most libraries are installed.")
        print("⚠️ Some optional libraries may be missing.")
    else:
        print("❌ Warning! Many libraries are missing.")
        print("🔧 Please install missing libraries before running ProjectP.")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
