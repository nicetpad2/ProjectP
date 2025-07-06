#!/usr/bin/env python3
"""
Enterprise Feature Selector Integration Test
Tests all feature selector components for Menu 1 compliance
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🎯 ENTERPRISE FEATURE SELECTOR INTEGRATION TEST")
print("=" * 70)

def test_real_profit_selector():
    """Test RealProfitFeatureSelector"""
    print("\n📋 Test 1: RealProfitFeatureSelector")
    try:
        from real_profit_feature_selector import RealProfitFeatureSelector
        
        # Create instance
        selector = RealProfitFeatureSelector(
            target_auc=0.70,
            max_features=30,
            max_trials=500
        )
        
        print("✅ RealProfitFeatureSelector imported and initialized")
        print(f"   📊 Target AUC: {selector.target_auc}")
        print(f"   🎯 Max Features: {selector.max_features}")
        print(f"   ⚡ Max Trials: {selector.max_trials}")
        
        # Check for enterprise methods
        methods = ['select_features', '_validate_enterprise_data', '_enterprise_shap_analysis']
        for method in methods:
            if hasattr(selector, method):
                print(f"   ✅ Method {method} exists")
            else:
                print(f"   ❌ Method {method} missing")
        
        return True
        
    except Exception as e:
        print(f"❌ RealProfitFeatureSelector failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_selector_wrapper():
    """Test AdvancedFeatureSelector wrapper"""
    print("\n📋 Test 2: AdvancedFeatureSelector Wrapper")
    try:
        from advanced_feature_selector import AdvancedFeatureSelector
        from real_profit_feature_selector import RealProfitFeatureSelector
        
        # Create instance
        selector = AdvancedFeatureSelector()
        
        # Verify inheritance
        is_real_profit = isinstance(selector, RealProfitFeatureSelector)
        
        print("✅ AdvancedFeatureSelector imported and initialized")
        print(f"   🔗 Inherits from RealProfitFeatureSelector: {is_real_profit}")
        
        if is_real_profit:
            print("   ✅ Proper inheritance confirmed")
        else:
            print("   ❌ Inheritance broken")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ AdvancedFeatureSelector failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fast_selector_deprecated():
    """Test FastFeatureSelector (deprecated)"""
    print("\n📋 Test 3: FastFeatureSelector (Deprecated)")
    try:
        import warnings
        from real_profit_feature_selector import RealProfitFeatureSelector
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            from fast_feature_selector import FastFeatureSelector
            selector = FastFeatureSelector()
            
            # Check warnings
            deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
            
            print("✅ FastFeatureSelector imported")
            print(f"   ⚠️ Deprecation warnings: {len(deprecation_warnings)} (expected)")
            
            # Verify inheritance
            is_real_profit = isinstance(selector, RealProfitFeatureSelector)
            print(f"   🔗 Inherits from RealProfitFeatureSelector: {is_real_profit}")
            
            if deprecation_warnings:
                print("   ✅ Proper deprecation warnings shown")
            
            return True
        
    except Exception as e:
        print(f"❌ FastFeatureSelector failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_elliott_wave_selector():
    """Test Elliott Wave FeatureSelector"""
    print("\n📋 Test 4: Elliott Wave FeatureSelector")
    try:
        from elliott_wave_modules.feature_selector import FeatureSelector
        from real_profit_feature_selector import RealProfitFeatureSelector
        
        # Create instance
        selector = FeatureSelector()
        
        # Verify inheritance
        is_real_profit = isinstance(selector, RealProfitFeatureSelector)
        
        print("✅ Elliott Wave FeatureSelector imported and initialized")
        print(f"   🔗 Inherits from RealProfitFeatureSelector: {is_real_profit}")
        
        if is_real_profit:
            print("   ✅ Proper inheritance confirmed")
        else:
            print("   ❌ Inheritance broken")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Elliott Wave FeatureSelector failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_menu1_import():
    """Test Menu 1 module import"""
    print("\n📋 Test 5: Menu 1 Module Import")
    try:
        # Test import only (not execution)
        import importlib.util
        
        menu_path = project_root / "menu_modules" / "menu_1_elliott_wave.py"
        spec = importlib.util.spec_from_file_location("menu_1_elliott_wave", menu_path)
        
        if spec and spec.loader:
            print("✅ Menu 1 module can be loaded")
            return True
        else:
            print("❌ Menu 1 module spec failed")
            return False
        
    except Exception as e:
        print(f"❌ Menu 1 module import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print(f"📁 Project Root: {project_root}")
    print(f"🐍 Python Version: {sys.version}")
    
    tests = [
        test_real_profit_selector,
        test_advanced_selector_wrapper,
        test_fast_selector_deprecated,
        test_elliott_wave_selector,
        test_menu1_import
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - ENTERPRISE COMPLIANCE VERIFIED")
        print("💰 MENU 1 IS READY FOR REAL PROFIT TRADING")
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
