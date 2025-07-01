#!/usr/bin/env python3
"""
🔍 DEBUG: ตรวจสอบปัญหา 'pd' is not defined ในระบบ Pipeline
หาแหล่งที่มาของปัญหาและแก้ไขให้สมบูรณ์แบบ
"""

import sys
import os
from pathlib import Path

# Add project path
sys.path.append('/content/drive/MyDrive/ProjectP')

print("🔍 DEBUGGING: 'pd' is not defined ปัญหา")
print("="*60)

# Test 1: Import เฉพาะโมดูลหลัก
print("\n📦 TEST 1: Import โมดูลหลักทั้งหมด")
try:
    from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
    print("✅ data_processor imported successfully")
except Exception as e:
    print(f"❌ data_processor import failed: {e}")

try:
    from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave
    print("✅ cnn_lstm_engine imported successfully")
except Exception as e:
    print(f"❌ cnn_lstm_engine import failed: {e}")

try:
    from elliott_wave_modules.dqn_agent import DQNReinforcementAgent
    print("✅ dqn_agent imported successfully")
except Exception as e:
    print(f"❌ dqn_agent import failed: {e}")

try:
    from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
    print("✅ feature_selector imported successfully")
except Exception as e:
    print(f"❌ feature_selector import failed: {e}")

try:
    from elliott_wave_modules.pipeline_orchestrator import ElliottWavePipelineOrchestrator
    print("✅ pipeline_orchestrator imported successfully")
except Exception as e:
    print(f"❌ pipeline_orchestrator import failed: {e}")

try:
    from elliott_wave_modules.performance_analyzer import ElliottWavePerformanceAnalyzer
    print("✅ performance_analyzer imported successfully")
except Exception as e:
    print(f"❌ performance_analyzer import failed: {e}")

try:
    from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
    print("✅ enterprise_ml_protection imported successfully")
except Exception as e:
    print(f"❌ enterprise_ml_protection import failed: {e}")

# Test 2: Initialize components
print("\n🔧 TEST 2: Initialize components")
try:
    paths = {
        'data': '/content/drive/MyDrive/ProjectP/datacsv',
        'outputs': '/content/drive/MyDrive/ProjectP/outputs',
        'logs': '/content/drive/MyDrive/ProjectP/logs'
    }
    
    data_processor = ElliottWaveDataProcessor(paths)
    print("✅ ElliottWaveDataProcessor initialized")
except Exception as e:
    print(f"❌ ElliottWaveDataProcessor init failed: {e}")

try:
    cnn_lstm_engine = CNNLSTMElliottWave(paths)
    print("✅ CNNLSTMElliottWave initialized")
except Exception as e:
    print(f"❌ CNNLSTMElliottWave init failed: {e}")

try:
    dqn_agent = DQNReinforcementAgent(paths)
    print("✅ DQNReinforcementAgent initialized")
except Exception as e:
    print(f"❌ DQNReinforcementAgent init failed: {e}")

try:
    feature_selector = EnterpriseShapOptunaFeatureSelector(paths)
    print("✅ EnterpriseShapOptunaFeatureSelector initialized")
except Exception as e:
    print(f"❌ EnterpriseShapOptunaFeatureSelector init failed: {e}")

# Test 3: Import Menu 1
print("\n📋 TEST 3: Import Menu 1")
try:
    from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
    print("✅ Menu1ElliottWaveFixed imported successfully")
    
    menu1 = Menu1ElliottWaveFixed()
    print("✅ Menu1ElliottWaveFixed initialized successfully")
except Exception as e:
    print(f"❌ Menu1ElliottWaveFixed failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎯 SUMMARY: Testing Complete")
print("="*60)
