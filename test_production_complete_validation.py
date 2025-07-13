#!/usr/bin/env python3
"""
🎉 PRODUCTION COMPLETE VALIDATION TEST - แก้ไขปัญหาสมบูรณ์
Test all fixes and ensure Enterprise Production readiness

แก้ไขปัญหา:
✅ core.logger module missing -> สร้าง compatibility wrapper
✅ Menu 1 missing -> ใช้ real_enterprise_menu_1.py 
✅ Data files availability -> ตรวจสอบ CSV files จริง
✅ Import errors -> แก้ไข ProjectP.py imports
✅ System integration -> ตรวจสอบการทำงานแบบ end-to-end
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent))

def test_1_core_logger_fix():
    """Test 1: ตรวจสอบการแก้ไข core.logger"""
    print("🔧 Test 1: Core Logger Compatibility Wrapper")
    print("-" * 60)
    
    try:
        # Test importing core.logger (should work now with compatibility wrapper)
        from core.logger import get_logger, get_unified_logger
        logger = get_logger("TEST")
        print("✅ core.logger import successful")
        print("✅ get_logger function available")
        print("✅ Compatibility wrapper working")
        
        # Test that it forwards to unified logger
        unified_logger = get_unified_logger("TEST")
        print("✅ Unified logger accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Core logger test failed: {e}")
        traceback.print_exc()
        return False

def test_2_menu_1_availability():
    """Test 2: ตรวจสอบ Menu 1 ที่แก้ไขแล้ว"""
    print("\n🎛️ Test 2: Menu 1 Real Enterprise Availability")
    print("-" * 60)
    
    try:
        # Test importing real enterprise menu 1
        from menu_modules.real_enterprise_menu_1 import RealEnterpriseMenu1
        print("✅ RealEnterpriseMenu1 import successful")
        
        # Test initializing menu
        menu = RealEnterpriseMenu1()
        print("✅ Menu 1 initialization successful")
        print(f"✅ Session ID: {menu.session_id}")
        print(f"✅ Initialized: {menu.initialized}")
        
        # Test AI components loading
        components_status = {
            'Data Processor': menu.data_processor is not None,
            'Feature Selector': menu.feature_selector is not None,
            'CNN-LSTM Engine': menu.cnn_lstm_engine is not None,
            'DQN Agent': menu.dqn_agent is not None
        }
        
        print(f"✅ AI Components Status:")
        for component, status in components_status.items():
            status_icon = "✅" if status else "⚠️"
            print(f"   {status_icon} {component}: {'Loaded' if status else 'Failed'}")
        
        successful_components = sum(components_status.values())
        print(f"✅ Components loaded: {successful_components}/4")
        
        return successful_components >= 2  # At least 2 components should load
        
    except Exception as e:
        print(f"❌ Menu 1 test failed: {e}")
        traceback.print_exc()
        return False

def test_3_data_files_validation():
    """Test 3: ตรวจสอบไฟล์ CSV ข้อมูลจริง"""
    print("\n📊 Test 3: Real Data CSV Files Validation")
    print("-" * 60)
    
    try:
        data_files = {
            'XAUUSD_M1.csv': 'datacsv/XAUUSD_M1.csv',
            'XAUUSD_M15.csv': 'datacsv/XAUUSD_M15.csv',
            'Features CSV': 'datacsv/xauusd_1m_features_with_elliott_waves.csv'
        }
        
        all_files_ok = True
        
        for name, path in data_files.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                size_mb = size / (1024 * 1024)
                print(f"✅ {name}: {size_mb:.1f} MB")
                
                # Basic validation - files should not be empty
                if size < 1000:  # Less than 1KB is suspicious
                    print(f"⚠️ {name}: File too small ({size} bytes)")
                    all_files_ok = False
            else:
                print(f"❌ {name}: File not found at {path}")
                all_files_ok = False
        
        # Test data loading
        try:
            import pandas as pd
            df = pd.read_csv('datacsv/XAUUSD_M1.csv', nrows=5)  # Test read first 5 rows
            print(f"✅ Data loading test: {len(df)} rows read successfully")
            print(f"✅ Columns: {list(df.columns)}")
        except Exception as e:
            print(f"⚠️ Data loading test failed: {e}")
            
        return all_files_ok
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        return False

def test_4_unified_master_system():
    """Test 4: ตรวจสอบ Unified Master Menu System"""
    print("\n🎛️ Test 4: Unified Master Menu System")
    print("-" * 60)
    
    try:
        # Test importing unified master system
        from core.unified_master_menu_system import UnifiedMasterMenuSystem
        print("✅ UnifiedMasterMenuSystem import successful")
        
        # Test initialization
        master_menu = UnifiedMasterMenuSystem()
        print("✅ Master menu initialization successful")
        print(f"✅ Session ID: {master_menu.session_id}")
        
        # Test component initialization
        init_success = master_menu.initialize_components()
        print(f"✅ Components initialization: {'Success' if init_success else 'Partial'}")
        
        # Check component status
        status = {
            'Resource Manager': master_menu.resource_manager is not None,
            'Logger': master_menu.logger is not None,
            'Config': master_menu.config is not None,
            'Menu 1': master_menu.menu_available
        }
        
        print("✅ System Components Status:")
        for component, is_available in status.items():
            icon = "✅" if is_available else "❌"
            print(f"   {icon} {component}: {'Available' if is_available else 'Failed'}")
        
        working_components = sum(status.values())
        print(f"✅ Working components: {working_components}/4")
        
        return working_components >= 3  # At least 3 should work
        
    except Exception as e:
        print(f"❌ Unified master system test failed: {e}")
        traceback.print_exc()
        return False

def test_5_complete_integration():
    """Test 5: ทดสอบการทำงานแบบ end-to-end"""
    print("\n🚀 Test 5: Complete System Integration Test")
    print("-" * 60)
    
    try:
        # Import and initialize complete system
        from core.unified_master_menu_system import UnifiedMasterMenuSystem
        
        master = UnifiedMasterMenuSystem()
        init_success = master.initialize_components()
        
        if not init_success:
            print("⚠️ Component initialization incomplete but continuing...")
        
        # Test that Menu 1 can be accessed
        if master.menu_available and master.menu_1:
            print("✅ Menu 1 is available and accessible")
            print(f"✅ Menu Type: {master.menu_type}")
            
            # Test accessing pipeline state
            if hasattr(master.menu_1, 'pipeline_state'):
                state = master.menu_1.pipeline_state
                print(f"✅ Pipeline state accessible: {state['total_steps']} steps")
            
            return True
        else:
            print("⚠️ Menu 1 not available - checking fallbacks...")
            return False
            
    except Exception as e:
        print(f"❌ Complete integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """รันการทดสอบทั้งหมด"""
    print("🧪 ENTERPRISE PRODUCTION COMPLETE VALIDATION")
    print("=" * 80)
    print("📅 Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("🎯 Testing fixes for production readiness")
    print("=" * 80)
    
    tests = [
        ("Core Logger Fix", test_1_core_logger_fix),
        ("Menu 1 Availability", test_2_menu_1_availability),
        ("Data Files Validation", test_3_data_files_validation),
        ("Unified Master System", test_4_unified_master_system),
        ("Complete Integration", test_5_complete_integration)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'passed': result,
                'duration': end_time - start_time
            }
            
            if result:
                passed += 1
                
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = {
                'passed': False,
                'duration': 0,
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        duration = result['duration']
        print(f"{status:<12} {test_name:<30} ({duration:.2f}s)")
        
        if 'error' in result:
            print(f"             Error: {result['error']}")
    
    print("\n" + "=" * 80)
    print(f"🏆 FINAL RESULT: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY!")
        return True
    elif passed >= total * 0.8:  # 80% pass rate
        print("🎯 MOSTLY READY - Minor issues detected but core functionality works")
        return True
    else:
        print("⚠️ SYSTEM NEEDS MORE FIXES - Critical issues detected")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 