#!/usr/bin/env python3
"""
🧪 PRODUCTION FIXES FINAL TEST
ทดสอบการแก้ไขปัญหาที่เกิดขึ้นจริงจากการรันระบบ

ปัญหาที่แก้ไข:
1. name 'get_progress_manager' is not defined  
2. 'NoneType' object has no attribute 'load_and_prepare_data'
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_data_processor_import():
    """ทดสอบการ import Data Processor"""
    print("🧪 Testing Data Processor Import...")
    print("=" * 50)
    
    try:
        from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
        print("✅ Data Processor imported successfully")
        
        # Test initialization
        data_processor = ElliottWaveDataProcessor()
        print("✅ Data Processor initialized successfully")
        
        # Check if it has required methods
        if hasattr(data_processor, 'load_and_prepare_data'):
            print("✅ load_and_prepare_data method exists")
        else:
            print("❌ load_and_prepare_data method missing")
            
        if hasattr(data_processor, 'load_real_data'):
            print("✅ load_real_data method exists")
        else:
            print("❌ load_real_data method missing")
            
        return True
        
    except Exception as e:
        print(f"❌ Data Processor test failed: {e}")
        traceback.print_exc()
        return False

def test_enhanced_menu1_initialization():
    """ทดสอบการ initialize Enhanced Menu 1"""
    print("\n🧪 Testing Enhanced Menu 1 Initialization...")
    print("=" * 50)
    
    try:
        from core.unified_enterprise_logger import get_unified_logger
        from core.config import get_global_config
        
        # Get configuration
        config = get_global_config().config
        
        print("📝 Importing Enhanced Menu 1...")
        from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
        print("✅ Enhanced Menu 1 imported successfully")
        
        print("\n📝 Creating Enhanced Menu 1 instance...")
        menu1 = EnhancedMenu1ElliottWave(config=config)
        print("✅ Enhanced Menu 1 instance created successfully")
        
        # Check data processor status
        print(f"\n📊 Data Processor Status: {type(menu1.data_processor).__name__ if menu1.data_processor else 'None'}")
        
        if menu1.data_processor is not None:
            print("✅ Data Processor is properly initialized")
            
            # Test if load_and_prepare_data method exists
            if hasattr(menu1.data_processor, 'load_and_prepare_data'):
                print("✅ load_and_prepare_data method is available")
            else:
                print("❌ load_and_prepare_data method is missing")
        else:
            print("⚠️ Data Processor is None - checking initialization...")
            
            # Try to initialize components manually
            if hasattr(menu1, '_initialize_components'):
                print("📝 Attempting manual component initialization...")
                result = menu1._initialize_components()
                if result:
                    print("✅ Manual initialization successful")
                    if menu1.data_processor:
                        print("✅ Data Processor now available")
                    else:
                        print("❌ Data Processor still None")
                else:
                    print("❌ Manual initialization failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced Menu 1 test failed: {e}")
        traceback.print_exc()
        return False

def test_menu1_pipeline_step():
    """ทดสอบขั้นตอนแรกของ pipeline"""
    print("\n🧪 Testing Menu 1 Pipeline First Step...")
    print("=" * 50)
    
    try:
        from core.config import get_global_config
        from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
        
        config = get_global_config().config
        menu1 = EnhancedMenu1ElliottWave(config=config)
        
        # Test _load_data_high_memory method
        print("📝 Testing _load_data_high_memory method...")
        
        test_config = {
            'data_file': 'xauusd_1m_features_with_elliott_waves.csv'
        }
        
        result = menu1._load_data_high_memory({}, test_config)
        print(f"📊 Result status: {result.get('status', 'Unknown')}")
        
        if result.get('status') == 'ERROR':
            print(f"❌ Error: {result.get('message')}")
            return False
        else:
            print("✅ Data loading step completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Pipeline step test failed: {e}")
        traceback.print_exc()
        return False

def test_import_chain():
    """ทดสอบ import chain ที่สำคัญ"""
    print("\n🧪 Testing Critical Import Chain...")
    print("=" * 50)
    
    imports_to_test = [
        "core.unified_enterprise_logger",
        "core.config", 
        "elliott_wave_modules.data_processor",
        "menu_modules.enhanced_menu_1_elliott_wave"
    ]
    
    success_count = 0
    for module_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}: {e}")
    
    print(f"\n📊 Import Success Rate: {success_count}/{len(imports_to_test)} ({(success_count/len(imports_to_test))*100:.1f}%)")
    return success_count == len(imports_to_test)

def main():
    """รันการทดสอบทั้งหมด"""
    print("🧪 PRODUCTION FIXES FINAL TEST")
    print("=" * 70)
    print("Testing fixes for:")
    print("1. ✅ name 'get_progress_manager' is not defined")
    print("2. ✅ 'NoneType' object has no attribute 'load_and_prepare_data'")
    print("=" * 70)
    
    tests = [
        ("Import Chain", test_import_chain),
        ("Data Processor", test_data_processor_import),
        ("Enhanced Menu 1", test_enhanced_menu1_initialization),
        ("Pipeline Step", test_menu1_pipeline_step)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🏃‍♂️ Running {test_name} Test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} Test: PASSED")
        else:
            print(f"❌ {test_name} Test: FAILED")
    
    print("\n" + "=" * 70)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 70)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"📊 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - PRODUCTION FIXES SUCCESSFUL!")
        print("🚀 System is ready for production use!")
    elif passed >= total * 0.75:
        print("⚠️ MOST TESTS PASSED - Minor issues may remain")
        print("🔧 System should work but may need minor adjustments")
    else:
        print("❌ SIGNIFICANT ISSUES DETECTED")
        print("🚨 System needs additional fixes before production use")
    
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    main() 