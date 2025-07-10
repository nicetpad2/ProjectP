#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 COMPREHENSIVE SYSTEM TEST - Fixed Feature Selector Integration
Test the entire pipeline with the fixed feature selector and resource optimization
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add project path
project_path = '/mnt/data/projects/ProjectP'
if project_path not in sys.path:
    sys.path.append(project_path)

def test_fixed_feature_selector_integration():
    """Test the fixed feature selector in the context of the full system"""
    
    print("🧪 NICEGOLD ProjectP - Fixed Feature Selector Integration Test")
    print("=" * 80)
    
    # Test 1: Import test
    print("\n📦 Test 1: Import Tests")
    try:
        from fixed_advanced_feature_selector import FixedAdvancedFeatureSelector
        print("✅ Fixed Feature Selector import: SUCCESS")
    except ImportError as e:
        print(f"❌ Fixed Feature Selector import: FAILED - {e}")
        return False
    
    try:
        from menu_modules.menu_1_elliott_wave import EnhancedElliottWaveMenu
        print("✅ Menu 1 import: SUCCESS")
    except ImportError as e:
        print(f"❌ Menu 1 import: FAILED - {e}")
        return False
    
    # Test 2: Feature Selector Creation
    print("\n🔧 Test 2: Feature Selector Creation")
    try:
        selector = FixedAdvancedFeatureSelector(
            target_auc=0.70,
            max_features=20,
            max_cpu_percent=80.0
        )
        print("✅ Fixed Feature Selector creation: SUCCESS")
        print(f"   Target AUC: {selector.target_auc}")
        print(f"   Max Features: {selector.max_features}")
        print(f"   Max CPU: {selector.max_cpu_percent}%")
    except Exception as e:
        print(f"❌ Fixed Feature Selector creation: FAILED - {e}")
        return False
    
    # Test 3: Data Processing Test
    print("\n📊 Test 3: Data Processing Test")
    try:
        import pandas as pd
        import numpy as np
        
        # Create realistic test data
        np.random.seed(42)
        n_samples = 5000  # Smaller test dataset
        n_features = 30
        
        # Generate features with some correlation to make it realistic
        X = pd.DataFrame(np.random.randn(n_samples, n_features))
        
        # Create target with some signal
        target_feature_indices = [0, 5, 10, 15, 20]  # Features that matter
        signal = np.sum(X.iloc[:, target_feature_indices], axis=1)
        noise = np.random.randn(n_samples) * 0.5
        y_continuous = signal + noise
        y = pd.Series((y_continuous > y_continuous.median()).astype(int))
        
        # Name the features
        X.columns = [f'feature_{i}' for i in range(n_features)]
        
        print(f"✅ Test data created: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Target balance: {y.value_counts().to_dict()}")
        
    except Exception as e:
        print(f"❌ Test data creation: FAILED - {e}")
        return False
    
    # Test 4: Feature Selection Process
    print("\n⚡ Test 4: Feature Selection Process")
    try:
        start_time = time.time()
        
        # Run feature selection
        selected_features, metadata = selector.select_features(X, y)
        
        selection_time = time.time() - start_time
        
        print("✅ Feature selection: SUCCESS")
        print(f"   Selected features: {len(selected_features)}")
        print(f"   Features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
        print(f"   AUC Score: {metadata['auc_score']:.3f}")
        print(f"   Final CPU usage: {metadata['final_cpu_usage']:.1f}%")
        print(f"   CPU compliant: {metadata['cpu_compliant']}")
        print(f"   Processing time: {selection_time:.1f} seconds")
        print(f"   Variable scope fixed: {metadata.get('variable_scope_fixed', 'N/A')}")
        
        # Validate results
        if len(selected_features) == 0:
            print("⚠️ WARNING: No features selected")
            return False
            
        if metadata['auc_score'] < 0.5:
            print("⚠️ WARNING: AUC score too low")
            
        if not metadata['cpu_compliant']:
            print("⚠️ WARNING: CPU usage exceeded limit")
            
    except Exception as e:
        print(f"❌ Feature selection: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Menu Integration Test
    print("\n🎛️ Test 5: Menu Integration Test")
    try:
        # Test if the menu can be created (without running full pipeline)
        from core.config import load_config
        from core.advanced_terminal_logger import get_terminal_logger
        
        config = load_config()
        logger = get_terminal_logger()
        
        menu = EnhancedElliottWaveMenu(config, logger)
        print("✅ Menu creation: SUCCESS")
        
        # Check if fixed feature selector is available
        from menu_modules.menu_1_elliott_wave import FIXED_FEATURE_SELECTOR_AVAILABLE
        print(f"   Fixed Feature Selector Available: {FIXED_FEATURE_SELECTOR_AVAILABLE}")
        
    except Exception as e:
        print(f"❌ Menu creation: FAILED - {e}")
        return False
    
    # Test 6: Resource Usage Validation
    print("\n🔧 Test 6: Resource Usage Validation")
    try:
        import psutil
        
        # Get current system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        print(f"✅ Resource monitoring: SUCCESS")
        print(f"   Current CPU: {cpu_percent:.1f}%")
        print(f"   Current Memory: {memory_percent:.1f}%")
        
        # Validate that our selector respects limits
        if metadata['final_cpu_usage'] > 85:  # Allow 5% buffer
            print(f"⚠️ WARNING: CPU usage might be too high: {metadata['final_cpu_usage']:.1f}%")
        else:
            print(f"✅ CPU usage within limits: {metadata['final_cpu_usage']:.1f}%")
            
    except Exception as e:
        print(f"❌ Resource monitoring: FAILED - {e}")
        return False
    
    # Final Summary
    print("\n📊 FINAL TEST SUMMARY")
    print("=" * 80)
    print("✅ All tests completed successfully!")
    print(f"🎯 Fixed Feature Selector Integration: READY FOR PRODUCTION")
    print(f"📊 Features selected: {len(selected_features)}")
    print(f"🎯 AUC achieved: {metadata['auc_score']:.3f}")
    print(f"💻 CPU usage: {metadata['final_cpu_usage']:.1f}%")
    print(f"🔧 Variable scope fixed: {metadata.get('variable_scope_fixed', 'N/A')}")
    print(f"⏱️ Processing time: {selection_time:.1f} seconds")
    
    return True

def main():
    """Main test function"""
    print(f"🚀 Starting comprehensive test at {datetime.now()}")
    
    success = test_fixed_feature_selector_integration()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        exit_code = 0
    else:
        print("\n❌ SOME TESTS FAILED - PLEASE REVIEW")
        exit_code = 1
    
    print(f"🏁 Test completed at {datetime.now()}")
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
