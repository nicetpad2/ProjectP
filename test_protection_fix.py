#!/usr/bin/env python3
"""
🎯 TEST ENTERPRISE ML PROTECTION SYSTEM CONFIGURATION FIX
ทดสอบการแก้ไขปัญหา configuration ของ Enterprise ML Protection System
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_protection_system_fix():
    """ทดสอบการแก้ไขปัญหา EnterpriseMLProtectionSystem"""
    
    print("🎯 Testing EnterpriseMLProtectionSystem Configuration Fix...")
    print("=" * 60)
    
    try:
        # Test 1: Import
        print("1. Testing import...")
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        print("✅ Import successful")
        
        # Test 2: Basic initialization
        print("\n2. Testing basic initialization...")
        protection1 = EnterpriseMLProtectionSystem()
        print("✅ Basic initialization successful")
        print(f"   Default config: {protection1.protection_config}")
        
        # Test 3: Initialization with empty config
        print("\n3. Testing initialization with empty config...")
        protection2 = EnterpriseMLProtectionSystem(config={})
        print("✅ Empty config initialization successful")
        
        # Test 4: Initialization with ml_protection config
        print("\n4. Testing initialization with ml_protection config...")
        config = {
            'ml_protection': {
                'overfitting_threshold': 0.10,
                'noise_threshold': 0.03,
                'leak_detection_window': 50
            }
        }
        protection3 = EnterpriseMLProtectionSystem(config=config)
        print("✅ ML protection config initialization successful")
        print(f"   Updated config: {protection3.protection_config}")
        
        # Test 5: Configuration validation
        print("\n5. Testing configuration validation...")
        validation = protection3.validate_configuration()
        print(f"✅ Configuration validation: {validation}")
        print(f"   Protection status: {protection3.get_protection_status()}")
        
        # Test 6: Menu 1 integration test
        print("\n6. Testing Menu 1 integration...")
        try:
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
            
            # Test with minimal config
            menu_config = {
                'elliott_wave': {
                    'target_auc': 0.70,
                    'max_features': 20
                },
                'ml_protection': {
                    'overfitting_threshold': 0.12,
                    'noise_threshold': 0.04
                }
            }
            
            # This should not fail with config error anymore
            menu1 = Menu1ElliottWave(config=menu_config)
            print("✅ Menu 1 initialization successful")
            print(f"   ML Protection status: {menu1.ml_protection.get_protection_status()}")
            
        except Exception as e:
            print(f"❌ Menu 1 integration test failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 7: Config update functionality
        print("\n7. Testing config update functionality...")
        protection3.update_protection_config({
            'overfitting_threshold': 0.08,
            'custom_threshold': 0.95
        })
        print("✅ Config update successful")
        print(f"   New config: {protection3.get_protection_config()}")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Enterprise ML Protection System fix successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_system_integration():
    """ทดสอบการ integrate กับระบบหลัก"""
    
    print("\n🔗 Testing System Integration...")
    print("=" * 60)
    
    try:
        # Test importing main system components
        from core.project_paths import get_project_paths
        from core.config import load_enterprise_config
        
        print("✅ Core imports successful")
        
        # Load actual system config
        config = load_enterprise_config()
        print(f"✅ System config loaded")
        
        # Test with actual system config
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        protection = EnterpriseMLProtectionSystem(config=config)
        print(f"✅ Protection system with actual config: {protection.get_protection_status()}")
        
        print("\n🎉 System integration test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ System integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    print("🛡️ ENTERPRISE ML PROTECTION SYSTEM - CONFIGURATION FIX TEST")
    print("=" * 70)
    
    test1_success = test_protection_system_fix()
    test2_success = test_system_integration()
    
    overall_success = test1_success and test2_success
    
    print("\n" + "=" * 70)
    print("📋 FINAL TEST RESULTS:")
    print(f"Protection System Fix: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"System Integration: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"Overall Result: {'🎉 SUCCESS' if overall_success else '💥 FAILURE'}")
    
    if overall_success:
        print("\n✅ Enterprise ML Protection System is now ready for production!")
        print("   - Configuration parameter support added")
        print("   - Menu 1 integration fixed")
        print("   - Validation and status checking implemented")
        print("   - All tests passed successfully")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())
