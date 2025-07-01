#!/usr/bin/env python3
"""
🔧 ENTERPRISE ML PROTECTION SYSTEM - DEBUG TEST
ทดสอบและ debug ปัญหา EnterpriseMLProtectionSystem
"""

import sys
import traceback

def test_enterprise_protection():
    """ทดสอบ EnterpriseMLProtectionSystem"""
    
    print("🔧 Testing EnterpriseMLProtectionSystem...")
    print("=" * 50)
    
    try:
        print("Step 1: Importing EnterpriseMLProtectionSystem...")
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        print("✅ Import successful")
        
        print("\nStep 2: Testing basic initialization...")
        protection = EnterpriseMLProtectionSystem()
        print("✅ Basic initialization successful")
        
        print("\nStep 3: Testing initialization with config...")
        config = {
            'ml_protection': {
                'overfitting_threshold': 0.1,
                'noise_threshold': 0.03
            }
        }
        protection_with_config = EnterpriseMLProtectionSystem(config=config)
        print("✅ Initialization with config successful")
        print(f"Config: {protection_with_config.protection_config}")
        
        print("\nStep 4: Testing validation...")
        validation = protection_with_config.validate_configuration()
        print(f"✅ Validation result: {validation}")
        
        print("\nStep 5: Testing status...")
        status = protection_with_config.get_protection_status()
        print(f"✅ Protection status: {status}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_menu1_integration():
    """ทดสอบการ integrate กับ Menu 1"""
    
    print("\n🌊 Testing Menu 1 Integration...")
    print("=" * 50)
    
    try:
        print("Step 1: Importing Menu1ElliottWave...")
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        print("✅ Import successful")
        
        print("\nStep 2: Creating test config...")
        config = {
            'elliott_wave': {
                'target_auc': 0.70,
                'max_features': 20
            },
            'ml_protection': {
                'overfitting_threshold': 0.12,
                'noise_threshold': 0.04
            }
        }
        print("✅ Test config created")
        
        print("\nStep 3: Initializing Menu 1...")
        menu1 = Menu1ElliottWave(config=config)
        print("✅ Menu 1 initialization successful")
        
        print("\nStep 4: Checking ML protection integration...")
        if hasattr(menu1, 'ml_protection'):
            print("✅ ml_protection attribute exists")
            status = menu1.ml_protection.get_protection_status()
            print(f"✅ Protection status: {status}")
        else:
            print("❌ ml_protection attribute missing")
            return False
            
        print("\n🎉 Menu 1 integration test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🛡️ ENTERPRISE ML PROTECTION SYSTEM - DEBUG TEST")
    print("=" * 60)
    
    test1 = test_enterprise_protection()
    test2 = test_menu1_integration()
    
    print("\n" + "=" * 60)
    print("📋 FINAL RESULTS:")
    print(f"EnterpriseMLProtectionSystem: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Menu 1 Integration: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print("\n🎉 ALL TESTS PASSED! System is ready!")
        return 0
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
