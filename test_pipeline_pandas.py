#!/usr/bin/env python3
"""
🧪 PIPELINE QUICK TEST
ทดสอบ pipeline แบบง่ายเพื่อตรวจจับปัญหา pd is not defined
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

print("🧪 Testing pipeline with pandas fix...")

try:
    # Import pandas first
    import pandas as pd
    import numpy as np
    print("✅ pandas/numpy imports: PASSED")
    
    # Import Menu 1
    from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
    print("✅ Menu 1 import: PASSED")
    
    # Initialize with minimal config
    config = {
        'data': {
            'csv_file': '/content/drive/MyDrive/ProjectP/datacsv/XAUUSD_M1.csv'
        },
        'cnn_lstm': {
            'epochs': 1,
            'batch_size': 16
        },
        'dqn': {
            'episodes': 2,
            'state_size': 5
        },
        'feature_selection': {
            'n_features': 5,
            'target_auc': 0.7
        }
    }
    
    # Initialize menu
    menu = Menu1ElliottWaveFixed(config=config)
    print("✅ Menu 1 initialization: PASSED")
    
    # Try to run pipeline (limited scope)
    print("🚀 Testing pipeline run...")
    try:
        results = menu.run_full_pipeline()
        print(f"✅ Pipeline execution: PASSED")
        print(f"AUC Score: {results.get('performance_analysis', {}).get('overall_performance', {}).get('key_metrics', {}).get('auc', 'N/A')}")
    except Exception as e:
        print(f"❌ Pipeline execution: FAILED - {str(e)}")
        
        # Check if it's the pandas issue
        if "'pd' is not defined" in str(e):
            print("🔍 CONFIRMED: pandas import issue detected!")
        else:
            print("🔍 Different issue found")
    
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    
    # Check if it's the pandas issue
    if "'pd' is not defined" in str(e):
        print("🔍 CONFIRMED: pandas import issue in initialization!")

print("\n🎯 Pipeline testing completed!")
