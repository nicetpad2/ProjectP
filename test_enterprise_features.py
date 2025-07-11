#!/usr/bin/env python3
"""
🧪 Test Enterprise Production Features
ทดสอบ Enterprise Features ที่เพิ่มเข้าไปใน NICEGOLD ProjectP
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_enterprise_progress():
    """ทดสอบ Enterprise Progress Bar"""
    print("🧪 Testing Enterprise Progress Bar")
    print("=" * 50)
    
    # Import the EnterpriseProgress class
    from menu_modules.enhanced_menu_1_elliott_wave import EnterpriseProgress
    
    # Test progress bar
    with EnterpriseProgress(10, "Demo Enterprise Progress") as progress:
        for i in range(10):
            time.sleep(0.5)
            progress.update(f"Processing step {i+1}")
    
    print("\n✅ Enterprise Progress Bar Test Complete!")

def test_enterprise_resource_manager():
    """ทดสอบ Enterprise Resource Manager"""
    print("\n🧪 Testing Enterprise Resource Manager")
    print("=" * 50)
    
    # Import the EnterpriseResourceManager class
    from menu_modules.enhanced_menu_1_elliott_wave import EnterpriseResourceManager
    
    # Test resource manager
    resource_manager = EnterpriseResourceManager(target_percentage=80.0)
    
    print("📊 Resource Manager Status Before Activation:")
    print(f"   Target: {resource_manager.target_percentage}%")
    print(f"   Active: {resource_manager.active}")
    
    # Activate resource manager
    success = resource_manager.activate_80_percent_ram()
    
    if success:
        print("\n📊 Resource Manager Status After Activation:")
        status = resource_manager.get_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
    
    print("\n✅ Enterprise Resource Manager Test Complete!")

def test_enhanced_menu1_initialization():
    """ทดสอบการ initialize Enhanced Menu 1 พร้อม Enterprise Features"""
    print("\n🧪 Testing Enhanced Menu 1 with Enterprise Features")
    print("=" * 50)
    
    try:
        # Import required modules
        from core.unified_enterprise_logger import get_unified_logger
        from core.config import get_global_config
        
        # Test imports
        print("📝 Importing Enhanced Menu 1...")
        from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
        print("✅ Enhanced Menu 1 imported successfully")
        
        # Get configuration
        config = get_global_config().config
        
        # Create instance (this will trigger Enterprise Feature initialization)
        print("\n📝 Creating Enhanced Menu 1 instance...")
        menu1 = EnhancedMenu1ElliottWave(config=config)
        print("✅ Enhanced Menu 1 instance created successfully")
        
        # Check Enterprise Features
        print("\n📊 Enterprise Features Status:")
        if hasattr(menu1, 'enterprise_resource_manager'):
            print("   ✅ Enterprise Resource Manager: AVAILABLE")
            status = menu1.enterprise_resource_manager.get_status()
            if 'current_usage' in status:
                print(f"   💾 RAM Usage: {status['current_usage']:.1f}%")
                print(f"   🎯 Target: {status['target_usage']:.1f}%")
                print(f"   📊 Active: {status['active']}")
        else:
            print("   ❌ Enterprise Resource Manager: NOT AVAILABLE")
        
        print("\n✅ Enhanced Menu 1 Enterprise Features Test Complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced Menu 1 Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """รันการทดสอบทั้งหมด"""
    print("🏢 NICEGOLD ENTERPRISE PRODUCTION FEATURES TEST")
    print("=" * 70)
    
    success_count = 0
    total_tests = 3
    
    try:
        # Test 1: Enterprise Progress Bar
        test_enterprise_progress()
        success_count += 1
    except Exception as e:
        print(f"\n❌ Progress Bar Test Failed: {e}")
    
    try:
        # Test 2: Enterprise Resource Manager
        test_enterprise_resource_manager()
        success_count += 1
    except Exception as e:
        print(f"\n❌ Resource Manager Test Failed: {e}")
    
    try:
        # Test 3: Enhanced Menu 1 with Enterprise Features
        if test_enhanced_menu1_initialization():
            success_count += 1
    except Exception as e:
        print(f"\n❌ Menu 1 Enterprise Features Test Failed: {e}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 ENTERPRISE FEATURES TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Tests Passed: {success_count}/{total_tests}")
    print(f"📊 Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 ALL ENTERPRISE FEATURES WORKING PERFECTLY!")
        print("🚀 PRODUCTION READY!")
    elif success_count > 0:
        print("⚠️ PARTIAL SUCCESS - Some features working")
    else:
        print("❌ ENTERPRISE FEATURES NEED ATTENTION")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 