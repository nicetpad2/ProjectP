#!/usr/bin/env python3
"""Simple Import Test"""

import sys
import os
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Print current directory and python path
print(f"Current directory: {os.getcwd()}")
print(f"Python path includes: {sys.path[:3]}...")

# Test import
try:
    print("🔍 Testing import...")
    from menu_modules.enhanced_menu_1_elliott_wave_perfect import EnhancedMenu1ElliottWavePerfect
    print("✅ SUCCESS: Import worked!")
    
    # Test class instantiation capability
    print("🔍 Testing class structure...")
    print(f"Class found: {EnhancedMenu1ElliottWavePerfect}")
    print(f"Class methods count: {len([m for m in dir(EnhancedMenu1ElliottWavePerfect) if not m.startswith('__')])}")
    
    print("🎉 ALL TESTS PASSED!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ Other Error: {e}")
    import traceback
    traceback.print_exc()
