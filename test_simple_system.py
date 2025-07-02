#!/usr/bin/env python3
"""
🧪 Simple System Test
Test basic system functionality without heavy monitoring
"""

import sys
import os
import warnings

# Suppress all warnings first
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def test_basic_imports():
    """Test basic package imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except Exception as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test data file availability"""
    print("\n📊 Testing data files...")
    
    data_dir = "datacsv"
    required_files = ["XAUUSD_M1.csv", "XAUUSD_M15.csv"]
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"✅ {file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} not found")
            return False
    
    return True

def test_core_modules():
    """Test core module imports"""
    print("\n🔧 Testing core modules...")
    
    try:
        sys.path.append('.')
        from core.config import load_enterprise_config
        print("✅ Core config module")
    except Exception as e:
        print(f"❌ Core config failed: {e}")
        return False
    
    try:
        from core.project_paths import ProjectPaths
        print("✅ Project paths module")
    except Exception as e:
        print(f"❌ Project paths failed: {e}")
        return False
    
    return True

def test_simple_menu():
    """Test simple menu system"""
    print("\n🎛️ Testing simple menu...")
    
    try:
        from core.menu_system import MenuSystem
        menu = MenuSystem()
        print("✅ Menu system initialized")
        return True
    except Exception as e:
        print(f"❌ Menu system failed: {e}")
        return False

def main():
    """Run simple system test"""
    print("🏢 NICEGOLD Enterprise - Simple System Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Files", test_data_files),
        ("Core Modules", test_core_modules),
        ("Simple Menu", test_simple_menu)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    main()
