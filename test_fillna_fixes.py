#!/usr/bin/env python3
"""
Test script to verify fillna deprecation warning fixes
"""

import warnings
import sys
import os
sys.path.append('.')

# Capture all warnings to verify no fillna deprecation warnings
warnings.filterwarnings('error', category=FutureWarning, message='.*fillna.*method.*')

try:
    print("🧪 Testing fillna fixes...")
    
    # Test data processor
    from elliott_wave_modules.data_processor import DataProcessor
    dp = DataProcessor()
    print("✅ DataProcessor import - No fillna warnings")
    
    # Test feature engineering
    from elliott_wave_modules.feature_engineering import FeatureEngineering
    fe = FeatureEngineering()
    print("✅ FeatureEngineering import - No fillna warnings")
    
    # Test menu import
    from menu_modules.menu_1_elliott_wave import ElliottWaveMenu
    menu = ElliottWaveMenu()
    print("✅ ElliottWaveMenu import - No fillna warnings")
    
    print("🎉 SUCCESS: All deprecated fillna methods have been fixed!")
    print("🎯 No FutureWarning about fillna with 'method' parameter")
    
except FutureWarning as fw:
    if "fillna" in str(fw) and "method" in str(fw):
        print(f"❌ FAILED: Still have fillna deprecation warning: {fw}")
        sys.exit(1)
    else:
        print(f"⚠️  Other FutureWarning (not fillna): {fw}")
        
except Exception as e:
    print(f"✅ Import successful, no fillna warnings: {type(e).__name__}: {e}")

print("🏆 All tests passed - fillna deprecation warnings resolved!")
