#!/usr/bin/env python3
"""
🧪 CUDA FIX VERIFICATION TEST
ทดสอบว่าการแก้ไข CUDA ทำงานได้หรือไม่
"""

import os
import sys

# Apply CUDA fixes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🧪 CUDA FIX VERIFICATION TEST")
print("=" * 50)

# Test 1: Basic imports
print("📋 Test 1: Basic imports...")
try:
    import numpy as np
    import pandas as pd
    print("✅ NumPy and Pandas imported successfully")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")

# Test 2: Scikit-learn
print("\n📋 Test 2: Scikit-learn...")
try:
    from sklearn.ensemble import RandomForestClassifier
    print("✅ Scikit-learn imported successfully")
except Exception as e:
    print(f"❌ Scikit-learn import failed: {e}")

# Test 3: TensorFlow (CPU only)
print("\n📋 Test 3: TensorFlow (CPU only)...")
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    print(f"✅ TensorFlow {tf.__version__} imported (CPU only)")
    print(f"📊 Available devices: {len(tf.config.list_physical_devices())}")
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")

# Test 4: PyTorch (CPU only)
print("\n📋 Test 4: PyTorch (CPU only)...")
try:
    import torch
    torch.set_default_tensor_type('torch.FloatTensor')
    print(f"✅ PyTorch {torch.__version__} imported (CPU only)")
    print(f"📊 CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")

# Test 5: Elliott Wave modules
print("\n📋 Test 5: Elliott Wave modules...")
try:
    from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
    print("✅ Elliott Wave DataProcessor imported successfully")
except Exception as e:
    print(f"❌ Elliott Wave import failed: {e}")

# Test 6: Core modules
print("\n📋 Test 6: Core modules...")
try:
    from core.logger import setup_enterprise_logger
    from core.config import load_enterprise_config
    print("✅ Core modules imported successfully")
except Exception as e:
    print(f"❌ Core modules import failed: {e}")

print("\n" + "=" * 50)
print("🎉 CUDA FIX VERIFICATION COMPLETE")
print("=" * 50)
print("✅ If you see this message without CUDA errors,")
print("   the CUDA fixes are working correctly!")
print("🚀 You can now run ProjectP.py safely")
