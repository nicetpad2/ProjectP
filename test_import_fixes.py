#!/usr/bin/env python3
"""
🔧 QUICK FIX TEST
ทดสอบการแก้ไข import pandas และปัญหาอื่นๆ
"""

import sys
import os
import warnings
from pathlib import Path

# Add project root to path
project_root = '/content/drive/MyDrive/ProjectP'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("🔧 Testing import fixes...")

# Test 1: Import pandas
try:
    import pandas as pd
    import numpy as np
    print("✅ pandas and numpy imports: PASSED")
except Exception as e:
    print(f"❌ pandas/numpy imports: FAILED - {str(e)}")

# Test 2: Import Menu 1
try:
    from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
    print("✅ Menu 1 import: PASSED")
except Exception as e:
    print(f"❌ Menu 1 import: FAILED - {str(e)}")
    
# Test 3: Initialize Menu 1
try:
    config = {
        'data': {'csv_file': '/content/drive/MyDrive/ProjectP/datacsv/XAUUSD_M1.csv'},
        'cnn_lstm': {'epochs': 1},
        'dqn': {'episodes': 1},
        'feature_selection': {'n_features': 3}
    }
    menu = Menu1ElliottWaveFixed(config=config)
    print("✅ Menu 1 initialization: PASSED")
except Exception as e:
    print(f"❌ Menu 1 initialization: FAILED - {str(e)}")
    
# Test 4: Check run_full_pipeline method
try:
    if hasattr(menu, 'run_full_pipeline'):
        print("✅ run_full_pipeline method: FOUND")
    else:
        print("❌ run_full_pipeline method: NOT FOUND")
except:
    print("⚠️ run_full_pipeline method: CANNOT CHECK")

print("\n🎯 Import fixes testing completed!")
