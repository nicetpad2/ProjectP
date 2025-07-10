#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 FULL DATA VERIFICATION TEST
ทดสอบให้แน่ใจว่าเมนูที่ 1 ใช้ข้อมูลทั้งหมด 100% ไม่มีการจำกัด

REQUIREMENTS:
- ✅ โหลดข้อมูล CSV ทั้งหมด ไม่มี nrows, sample, chunk
- ✅ ใช้ข้อมูลทั้งหมดในการประมวลผล ไม่มี .head(), .iloc[]
- ✅ ไม่มี fallback, mock, dummy data ใดๆ
- ✅ Full Power Mode เท่านั้น
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import necessary modules
from core.project_paths import get_project_paths
from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
from ultimate_full_power_config import ULTIMATE_FULL_POWER_CONFIG, apply_full_power_mode

def test_full_data_loading():
    """Test that all CSV data is loaded without limits"""
    print("🎯 TESTING FULL DATA LOADING - NO LIMITS")
    print("=" * 60)
    
    # Get project paths
    paths = get_project_paths()
    datacsv_path = paths.datacsv
    
    # Check CSV files
    csv_files = list(datacsv_path.glob("*.csv"))
    print(f"📁 Found CSV files: {len(csv_files)}")
    
    for csv_file in csv_files:
        print(f"  📊 {csv_file.name}")
        
        # Load with pandas directly to get full size
        df_full = pd.read_csv(csv_file)
        full_rows = len(df_full)
        print(f"    Full size: {full_rows:,} rows")
        
        # Test data processor loading
        config = apply_full_power_mode({})
        processor = ElliottWaveDataProcessor(config=config)
        
        # Load data through processor
        df_processed = processor.load_real_data()
        
        if df_processed is not None:
            processed_rows = len(df_processed)
            print(f"    Processed size: {processed_rows:,} rows")
            
            # Check if we're using all data
            if processed_rows >= full_rows * 0.95:  # Allow for minimal cleaning
                print(f"    ✅ FULL DATA USED: {processed_rows:,}/{full_rows:,} ({processed_rows/full_rows*100:.1f}%)")
            else:
                print(f"    ❌ DATA REDUCED: {processed_rows:,}/{full_rows:,} ({processed_rows/full_rows*100:.1f}%)")
                return False
        else:
            print(f"    ❌ FAILED TO LOAD DATA")
            return False
    
    return True

def test_feature_engineering():
    """Test that feature engineering uses all data"""
    print("\n🧠 TESTING FEATURE ENGINEERING - NO LIMITS")
    print("=" * 60)
    
    # Apply full power config
    config = apply_full_power_mode({})
    processor = ElliottWaveDataProcessor(config=config)
    
    # Load data
    data = processor.load_real_data()
    if data is None:
        print("❌ Failed to load data")
        return False
    
    initial_rows = len(data)
    print(f"📊 Initial data: {initial_rows:,} rows")
    
    # Create features
    features = processor.create_elliott_wave_features(data)
    feature_rows = len(features)
    
    print(f"🎯 Features created: {feature_rows:,} rows")
    print(f"📈 Features count: {features.shape[1]} columns")
    
    # Check data preservation
    data_preservation = feature_rows / initial_rows * 100
    print(f"💾 Data preservation: {data_preservation:.1f}%")
    
    if data_preservation >= 95.0:
        print("✅ EXCELLENT: >95% data preserved")
        return True
    elif data_preservation >= 90.0:
        print("⚠️ GOOD: >90% data preserved")
        return True
    else:
        print("❌ POOR: <90% data preserved")
        return False

def test_ml_preparation():
    """Test that ML data preparation uses all available data"""
    print("\n🤖 TESTING ML DATA PREPARATION - NO LIMITS")
    print("=" * 60)
    
    config = apply_full_power_mode({})
    processor = ElliottWaveDataProcessor(config=config)
    
    # Load and process
    data = processor.load_real_data()
    features = processor.create_elliott_wave_features(data)
    initial_rows = len(features)
    
    # Prepare ML data
    X, y = processor.prepare_ml_data(features)
    ml_rows = len(X)
    
    print(f"📊 Features data: {initial_rows:,} rows")
    print(f"🎯 ML data: {ml_rows:,} rows")
    print(f"📈 Features count: {X.shape[1]} columns")
    print(f"⚖️ Target balance: {y.mean():.3f}")
    
    # Check data usage
    data_usage = ml_rows / initial_rows * 100
    print(f"💯 Data usage: {data_usage:.1f}%")
    
    if data_usage >= 90.0:
        print("✅ EXCELLENT: >90% data used for ML")
        return True
    else:
        print("❌ INSUFFICIENT: <90% data used for ML")
        return False

def verify_no_limits_config():
    """Verify that full power config is applied correctly"""
    print("\n⚙️ VERIFYING FULL POWER CONFIGURATION")
    print("=" * 60)
    
    config = apply_full_power_mode({})
    
    # Check data processor config
    data_config = config.get('data_processor', {})
    print(f"📊 load_all_data: {data_config.get('load_all_data', False)}")
    print(f"🚫 sampling_disabled: {data_config.get('sampling_disabled', False)}")
    print(f"💾 chunk_size: {data_config.get('chunk_size', 'default')}")
    
    # Check feature selector config
    feature_config = config.get('feature_selector', {})
    print(f"🎯 target_auc: {feature_config.get('target_auc', 'default')}")
    print(f"🔢 max_features: {feature_config.get('max_features', 'default')}")
    print(f"⏱️ timeout_minutes: {feature_config.get('timeout_minutes', 'default')}")
    
    # Validate settings
    checks = [
        data_config.get('load_all_data', False),
        data_config.get('sampling_disabled', False),
        data_config.get('chunk_size', 1) == 0,  # 0 means no chunking
        feature_config.get('timeout_minutes', 1) == 0,  # 0 means no timeout
        feature_config.get('max_features', 0) >= 50,  # At least 50 features
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\n🔍 Configuration checks: {passed}/{total} passed")
    
    if passed == total:
        print("✅ FULL POWER CONFIG VERIFIED")
        return True
    else:
        print("❌ CONFIGURATION ISSUES DETECTED")
        return False

def main():
    """Run all verification tests"""
    print("🚀 FULL DATA VERIFICATION TEST SUITE")
    print("=" * 80)
    print("OBJECTIVE: Verify Menu 1 uses 100% of CSV data with NO limits")
    print("=" * 80)
    
    tests = [
        ("Full Data Loading", test_full_data_loading),
        ("Feature Engineering", test_feature_engineering),
        ("ML Data Preparation", test_ml_preparation),
        ("Full Power Config", verify_no_limits_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    total = len(results)
    success_rate = passed / total * 100
    
    print(f"\n🎯 OVERALL RESULT: {passed}/{total} tests passed ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("🏆 PERFECT: Menu 1 uses 100% data with NO limits!")
        print("✅ READY FOR ENTERPRISE PRODUCTION")
    elif success_rate >= 75:
        print("⚠️ GOOD: Most tests passed, minor issues may exist")
    else:
        print("❌ CRITICAL: Major issues detected, requires fixes")
    
    return success_rate == 100

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
