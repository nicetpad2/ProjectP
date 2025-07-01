#!/usr/bin/env python3
"""
🔧 TEST MENU 1 TEXT ERROR FIX
ทดสอบการแก้ไขปัญหา 'Text' is not defined
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_menu1_import_with_rich():
    """ทดสอบการ import Menu 1 พร้อม Rich"""
    print("🧪 Testing Menu 1 import with Rich components...")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        print("✅ Import successful")
        return True
    except NameError as e:
        if "'Text' is not defined" in str(e):
            print(f"❌ Text error still exists: {e}")
            return False
        else:
            print(f"❌ Other NameError: {e}")
            return False
    except Exception as e:
        print(f"❌ Other error: {e}")
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
        
        # Test _display_pipeline_overview method
        print("\n🧪 Testing _display_pipeline_overview...")
        menu._display_pipeline_overview()
        print("✅ _display_pipeline_overview works")
        
        return True
        
    except NameError as e:
        if "'Text' is not defined" in str(e):
            print(f"❌ Text error in initialization: {e}")
            return False
        else:
            print(f"❌ Other NameError in initialization: {e}")
            return False
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_display_methods():
    """ทดสอบ display methods"""
    print("\n🧪 Testing display methods...")
    try:
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        menu = Menu1ElliottWave()
        
        # Test _display_results method
        print("Testing _display_results...")
        menu._display_results()
        print("✅ _display_results works")
        
        return True
        
    except NameError as e:
        if "'Text' is not defined" in str(e):
            print(f"❌ Text error in display methods: {e}")
            return False
        else:
            print(f"❌ Other NameError in display methods: {e}")
            return False
    except Exception as e:
        print(f"❌ Display methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_format_report_content():
    """ทดสอบ _format_report_content method ที่แก้ไขก่อนหน้า"""
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
                "DQN Total Reward": "125.50"
            }
        }
        
        formatted_report = menu._format_report_content(test_content)
        print("✅ _format_report_content works correctly")
        return True
        
    except Exception as e:
        print(f"❌ _format_report_content failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 STARTING MENU 1 TEXT ERROR FIX TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_menu1_import_with_rich),
        ("Initialization Test", test_menu1_initialization),
        ("Display Methods Test", test_display_methods),
        ("Format Report Test", test_format_report_content)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append(result)
        print(f"Result: {'✅ PASSED' if result else '❌ FAILED'}")
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY:")
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Text error fix successful!")
        print("🌊 Menu 1 is ready for full pipeline execution!")
    else:
        print("❌ Some tests failed. Additional fixes needed.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
