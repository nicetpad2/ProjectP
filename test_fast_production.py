#!/usr/bin/env python3
"""
🚀 NICEGOLD Enterprise - Fast Production Test
ทดสอบระบบแบบเร็วสำหรับการใช้งานจริง

เน้นการทำงานที่เร็ว มีประสิทธิภาพ และใช้งานได้จริง
"""

import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
project_root = Path("/mnt/data/projects/ProjectP")
sys.path.insert(0, str(project_root))

def test_fast_feature_selection():
    """ทดสอบ Feature Selection แบบเร็ว"""
    print("🧠 Testing Fast Feature Selection...")
    
    try:
        from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
        from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
        import pandas as pd
        import numpy as np
        
        # Initialize data processor
        data_processor = ElliottWaveDataProcessor()
        
        # Load sample of real data for fast testing
        print("📊 Loading sample data...")
        df = data_processor.load_real_data()
        
        if df is None or len(df) == 0:
            print("❌ No data loaded")
            return False
            
        # Use small sample for fast testing
        sample_size = min(5000, len(df))  # Maximum 5000 rows
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"✅ Using sample: {len(df_sample):,} rows")
        
        # Create features quickly
        print("⚙️ Creating basic features...")
        X, y = data_processor.create_elliott_wave_features_fast(df_sample)
        
        if X is None or len(X) == 0:
            print("❌ No features created")
            return False
            
        print(f"✅ Features created: {X.shape}")
        
        # Initialize fast feature selector
        print("🎯 Initializing Fast Feature Selector...")
        feature_selector = EnterpriseShapOptunaFeatureSelector(
            target_auc=0.65,    # Lower target for speed
            max_features=15,    # Fewer features
            n_trials=20,        # Fewer trials
            timeout=120         # 2 minutes max
        )
        
        # Select features
        print("🚀 Running Fast Feature Selection...")
        start_time = time.time()
        
        selected_features, results = feature_selector.select_features(X, y)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Feature Selection completed in {duration:.1f} seconds")
        print(f"📊 Selected {len(selected_features)} features")
        print(f"🎯 AUC Score: {results.get('best_auc', 0):.3f}")
        
        # Display selected features
        print("\n🎯 Selected Features:")
        for i, feature in enumerate(selected_features[:10], 1):
            print(f"  {i:2d}. {feature}")
        
        if len(selected_features) > 10:
            print(f"  ... and {len(selected_features) - 10} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in feature selection test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fast_pipeline():
    """ทดสอบ Pipeline แบบเร็ว"""
    print("\n🌊 Testing Fast Elliott Wave Pipeline...")
    
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        
        # Initialize with fast configuration
        config = {
            'fast_mode': True,
            'sample_size': 3000,
            'target_auc': 0.65,
            'n_trials': 15,
            'timeout': 180
        }
        
        menu1 = Menu1ElliottWave(config=config)
        
        print("🚀 Running Fast Pipeline...")
        start_time = time.time()
        
        result = menu1.run_fast_pipeline()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result and result.get('success'):
            print(f"✅ Fast Pipeline completed in {duration:.1f} seconds")
            print(f"📊 Performance: {result.get('performance', {})}")
            return True
        else:
            print(f"❌ Pipeline failed: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in pipeline test: {e}")
        return False

def main():
    """Main test function"""
    print("🏢 NICEGOLD ENTERPRISE - FAST PRODUCTION TEST")
    print("=" * 60)
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    total_start = time.time()
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Fast Feature Selection
    total_tests += 1
    if test_fast_feature_selection():
        tests_passed += 1
    
    # Test 2: Fast Pipeline
    total_tests += 1
    if test_fast_pipeline():
        tests_passed += 1
    
    total_end = time.time()
    total_duration = total_end - total_start
    
    print("\n" + "=" * 60)
    print("🎯 FAST PRODUCTION TEST RESULTS")
    print("=" * 60)
    print(f"✅ Tests Passed: {tests_passed}/{total_tests}")
    print(f"⏰ Total Duration: {total_duration:.1f} seconds")
    print(f"🎯 Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
    else:
        print("⚠️ Some tests failed - Review and optimize")
    
    print(f"⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
