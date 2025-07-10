#!/usr/bin/env python3
"""
🧪 NICEGOLD Enterprise ProjectP - Installation Test
Test all critical libraries for proper installation
"""

def test_installations():
    """Test all critical library installations"""
    
    print("🔬 NICEGOLD ENTERPRISE - LIBRARY INSTALLATION TEST")
    print("=" * 60)
    
    tested_libraries = []
    failed_libraries = []
    
    # Test Core Data Science Libraries
    print("\n📊 CORE DATA SCIENCE LIBRARIES:")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        tested_libraries.append(f"NumPy {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy: {e}")
        failed_libraries.append("NumPy")
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
        tested_libraries.append(f"Pandas {pd.__version__}")
    except Exception as e:
        print(f"❌ Pandas: {e}")
        failed_libraries.append("Pandas")
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        tested_libraries.append(f"Scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"❌ Scikit-learn: {e}")
        failed_libraries.append("Scikit-learn")
    
    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__}")
        tested_libraries.append(f"SciPy {scipy.__version__}")
    except Exception as e:
        print(f"❌ SciPy: {e}")
        failed_libraries.append("SciPy")
    
    try:
        import joblib
        print(f"✅ Joblib {joblib.__version__}")
        tested_libraries.append(f"Joblib {joblib.__version__}")
    except Exception as e:
        print(f"❌ Joblib: {e}")
        failed_libraries.append("Joblib")
    
    # Test Deep Learning Libraries
    print("\n🧠 DEEP LEARNING LIBRARIES:")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        tested_libraries.append(f"TensorFlow {tf.__version__}")
    except Exception as e:
        print(f"❌ TensorFlow: {e}")
        failed_libraries.append("TensorFlow")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        tested_libraries.append(f"PyTorch {torch.__version__}")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        failed_libraries.append("PyTorch")
    
    # Test ML Enhancement Libraries
    print("\n🎯 ML ENHANCEMENT LIBRARIES:")
    try:
        import shap
        print(f"✅ SHAP {shap.__version__}")
        tested_libraries.append(f"SHAP {shap.__version__}")
    except Exception as e:
        print(f"❌ SHAP: {e}")
        failed_libraries.append("SHAP")
    
    try:
        import optuna
        print(f"✅ Optuna {optuna.__version__}")
        tested_libraries.append(f"Optuna {optuna.__version__}")
    except Exception as e:
        print(f"❌ Optuna: {e}")
        failed_libraries.append("Optuna")
    
    # Test Reinforcement Learning Libraries
    print("\n🤖 REINFORCEMENT LEARNING LIBRARIES:")
    try:
        import stable_baselines3
        print(f"✅ Stable-Baselines3 {stable_baselines3.__version__}")
        tested_libraries.append(f"Stable-Baselines3 {stable_baselines3.__version__}")
    except Exception as e:
        print(f"❌ Stable-Baselines3: {e}")
        failed_libraries.append("Stable-Baselines3")
    
    try:
        import gymnasium
        print(f"✅ Gymnasium {gymnasium.__version__}")
        tested_libraries.append(f"Gymnasium {gymnasium.__version__}")
    except Exception as e:
        print(f"❌ Gymnasium: {e}")
        failed_libraries.append("Gymnasium")
    
    # Test Additional Libraries
    print("\n📈 ADDITIONAL LIBRARIES:")
    try:
        import ta
        print(f"✅ TA (Technical Analysis) {ta.__version__}")
        tested_libraries.append(f"TA {ta.__version__}")
    except Exception as e:
        print(f"❌ TA: {e}")
        failed_libraries.append("TA")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        tested_libraries.append(f"OpenCV {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV: {e}")
        failed_libraries.append("OpenCV")
    
    try:
        import PIL
        print(f"✅ Pillow {PIL.__version__}")
        tested_libraries.append(f"Pillow {PIL.__version__}")
    except Exception as e:
        print(f"❌ Pillow: {e}")
        failed_libraries.append("Pillow")
    
    try:
        import pywt
        print(f"✅ PyWavelets {pywt.__version__}")
        tested_libraries.append(f"PyWavelets {pywt.__version__}")
    except Exception as e:
        print(f"❌ PyWavelets: {e}")
        failed_libraries.append("PyWavelets")
    
    try:
        import yaml
        print(f"✅ PyYAML")
        tested_libraries.append("PyYAML")
    except Exception as e:
        print(f"❌ PyYAML: {e}")
        failed_libraries.append("PyYAML")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        tested_libraries.append(f"Matplotlib {matplotlib.__version__}")
    except Exception as e:
        print(f"❌ Matplotlib: {e}")
        failed_libraries.append("Matplotlib")
    
    try:
        import seaborn
        print(f"✅ Seaborn {seaborn.__version__}")
        tested_libraries.append(f"Seaborn {seaborn.__version__}")
    except Exception as e:
        print(f"❌ Seaborn: {e}")
        failed_libraries.append("Seaborn")
    
    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
        tested_libraries.append(f"Plotly {plotly.__version__}")
    except Exception as e:
        print(f"❌ Plotly: {e}")
        failed_libraries.append("Plotly")
    
    try:
        import psutil
        print(f"✅ Psutil {psutil.__version__}")
        tested_libraries.append(f"Psutil {psutil.__version__}")
    except Exception as e:
        print(f"❌ Psutil: {e}")
        failed_libraries.append("Psutil")
    
    # Test Core NICEGOLD Libraries
    print("\n🏢 CORE NICEGOLD COMPATIBILITY TEST:")
    try:
        # Test core features that NICEGOLD requires
        import numpy as np
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier
        import tensorflow.keras as keras
        print("✅ Core NICEGOLD features available")
        tested_libraries.append("NICEGOLD Core Features")
    except Exception as e:
        print(f"❌ NICEGOLD Core Features: {e}")
        failed_libraries.append("NICEGOLD Core Features")
    
    # Summary Report
    print("\n" + "=" * 60)
    print("📋 INSTALLATION SUMMARY REPORT")
    print("=" * 60)
    
    print(f"\n✅ SUCCESSFULLY INSTALLED: {len(tested_libraries)} libraries")
    for lib in tested_libraries:
        print(f"   • {lib}")
    
    if failed_libraries:
        print(f"\n❌ FAILED INSTALLATIONS: {len(failed_libraries)} libraries")
        for lib in failed_libraries:
            print(f"   • {lib}")
    else:
        print(f"\n🎉 ALL LIBRARIES INSTALLED SUCCESSFULLY!")
    
    success_rate = (len(tested_libraries) / (len(tested_libraries) + len(failed_libraries))) * 100
    print(f"\n📊 SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🏆 EXCELLENT! NICEGOLD PROJECT IS READY TO RUN!")
    elif success_rate >= 75:
        print("⚠️  GOOD! Some optional libraries missing but core functionality available.")
    else:
        print("🚨 CRITICAL ISSUES! Please reinstall missing libraries.")
    
    return success_rate, tested_libraries, failed_libraries

if __name__ == "__main__":
    test_installations()
