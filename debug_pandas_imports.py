#!/usr/bin/env python3
"""
🔍 SPECIFIC PANDAS DEBUG
ตรวจจับปัญหา pd is not defined ในแต่ละโมดูล
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

# Import pandas globally
import pandas as pd
import numpy as np

print("🔍 Debugging pandas import issues...")

# Test each module individually
modules_to_test = [
    ('elliott_wave_modules.data_processor', 'ElliottWaveDataProcessor'),
    ('elliott_wave_modules.cnn_lstm_engine', 'CNNLSTMElliottWave'),
    ('elliott_wave_modules.dqn_agent', 'DQNReinforcementAgent'),
    ('elliott_wave_modules.feature_selector', 'EnterpriseShapOptunaFeatureSelector'),
    ('elliott_wave_modules.pipeline_orchestrator', 'ElliottWavePipelineOrchestrator'),
    ('elliott_wave_modules.performance_analyzer', 'ElliottWavePerformanceAnalyzer'),
    ('elliott_wave_modules.enterprise_ml_protection', 'EnterpriseMLProtectionSystem')
]

for module_name, class_name in modules_to_test:
    try:
        print(f"Testing {module_name}...")
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        
        # Try to create instance
        instance = cls()
        print(f"✅ {class_name}: PASSED")
        
    except NameError as e:
        if "'pd' is not defined" in str(e):
            print(f"❌ {class_name}: PANDAS IMPORT ISSUE - {str(e)}")
        else:
            print(f"❌ {class_name}: OTHER NAME ERROR - {str(e)}")
    except Exception as e:
        print(f"⚠️ {class_name}: OTHER ERROR - {str(e)}")

print("\n🔍 Testing complete! Check for pandas import issues above.")
