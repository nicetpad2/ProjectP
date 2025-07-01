#!/usr/bin/env python3
"""
🧪 QUICK TEST: Fixed Menu 1 Elliott Wave
ทดสอบการแก้ไขปัญหาและการปรับปรุง AUC performance
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_syntax():
    """ทดสอบ syntax"""
    print("🧪 Testing syntax...")
    import ast
    try:
        with open('menu_modules/menu_1_elliott_wave.py', 'r') as f:
            code = f.read()
        ast.parse(code)
        print("✅ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_import():
    """ทดสอบการ import"""
    print("\n🧪 Testing import...")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
        print("✅ Import successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_initialization():
    """ทดสอบการ initialize"""
    print("\n🧪 Testing initialization...")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
        menu = Menu1ElliottWaveFixed()
        print("✅ Initialization successful")
        
        # Test menu info
        info = menu.get_menu_info()
        print(f"📋 Menu: {info['name']}")
        print(f"📋 Version: {info['version']}")
        print(f"📋 Status: {info['status']}")
        
        return True
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_selector():
    """ทดสอบ Enhanced Feature Selector"""
    print("\n🧪 Testing Enhanced Feature Selector...")
    try:
        from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
        
        selector = EnterpriseShapOptunaFeatureSelector(
            target_auc=0.70,
            max_features=30,
            n_trials=5,  # Quick test
            timeout=60
        )
        print("✅ Feature Selector initialized successfully")
        print(f"📊 Target AUC: {selector.target_auc}")
        print(f"📊 Max Features: {selector.max_features}")
        print(f"📊 Trials: {selector.n_trials}")
        print(f"📊 Timeout: {selector.timeout}s")
        
        return True
    except Exception as e:
        print(f"❌ Feature Selector failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔥 TESTING FIXED MENU 1 ELLIOTT WAVE")
    print("=" * 50)
    
    tests = [
        test_syntax,
        test_import,
        test_initialization,
        test_feature_selector
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    success_count = sum(results)
    total_tests = len(results)
    
    print(f"\n🏆 RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED - System is ready!")
        return True
    else:
        print("❌ Some tests failed - Check errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
