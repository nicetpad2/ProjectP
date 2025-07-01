#!/usr/bin/env python3
"""
🧪 QUICK TEST - Menu 1 AttributeError Fix
Test if generate_report fix works
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_menu1_import():
    """ทดสอบการ import Menu 1"""
    print("🧪 Testing Menu 1 import...")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        print("✅ Import successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_menu1_initialization():
    """ทดสอบการ initialize Menu 1"""
    print("\n🧪 Testing Menu 1 initialization...")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        menu = Menu1ElliottWave()
        print("✅ Menu 1 initialized successfully")
        
        # Check output_manager methods
        print("\n📋 Available methods in output_manager:")
        methods = [method for method in dir(menu.output_manager) if not method.startswith('_')]
        for method in methods:
            print(f"  • {method}")
        
        # Check if generate_report method exists (should NOT exist)
        if hasattr(menu.output_manager, 'generate_report'):
            print("❌ generate_report method still exists!")
            return False
        
        # Check if save_report method exists (should exist)
        if hasattr(menu.output_manager, 'save_report'):
            print("✅ save_report method exists")
        else:
            print("❌ save_report method missing!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_format_report_content():
    """ทดสอบ _format_report_content method"""
    print("\n🧪 Testing _format_report_content method...")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        menu = Menu1ElliottWave()
        
        # Test data
        test_content = {
            "📊 Data Summary": {
                "Total Rows": "1,553",
                "Selected Features": 10,
                "Data Source": "REAL Market Data"
            },
            "🧠 Model Performance": {
                "CNN-LSTM AUC": "0.8500",
                "DQN Total Reward": "125.50",
                "Target AUC ≥ 0.70": "✅ ACHIEVED"
            }
        }
        
        formatted_report = menu._format_report_content(test_content)
        print("✅ _format_report_content works correctly")
        print("\n📄 Sample formatted report:")
        print(formatted_report[:300] + "..." if len(formatted_report) > 300 else formatted_report)
        return True
        
    except Exception as e:
        print(f"❌ _format_report_content failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 STARTING MENU 1 ATTRIBUTEERROR FIX TESTS")
    print("=" * 60)
    
    tests = [
        test_menu1_import,
        test_menu1_initialization, 
        test_format_report_content
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! AttributeError fix successful!")
    else:
        print("❌ Some tests failed. Fix needed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
