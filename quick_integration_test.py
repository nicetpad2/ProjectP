#!/usr/bin/env python3
"""
🔍 QUICK RESOURCE INTEGRATION TEST
ทดสอบการ integration ของ Resource Management อย่างรวดเร็ว
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_integration():
    """ทดสอบการ integration อย่างรวดเร็ว"""
    print("🔍 QUICK RESOURCE INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Test UnifiedMasterMenuSystem
        print("1. Testing UnifiedMasterMenuSystem...")
        from core.unified_master_menu_system import UnifiedMasterMenuSystem
        menu = UnifiedMasterMenuSystem()
        print("   ✅ UnifiedMasterMenuSystem imported")
        
        success = menu.initialize_components()
        print(f"   📊 Components initialized: {success}")
        print(f"   🧠 Resource Manager: {'✅ ACTIVE' if menu.resource_manager else '❌ NOT FOUND'}")
        print(f"   🌊 Menu 1: {'✅ READY' if menu.menu_1 else '❌ NOT READY'}")
        
        # Test Menu 1 directly
        print("\n2. Testing Enhanced Menu 1...")
        from menu_modules.enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
        menu1 = EnhancedMenu1ElliottWave()
        print("   ✅ Enhanced Menu 1 imported")
        print(f"   🧠 Unified RM: {'✅ ACTIVE' if menu1.resource_manager else '❌ NOT FOUND'}")
        print(f"   🏢 Enterprise RM: {'✅ ACTIVE' if hasattr(menu1, 'enterprise_resource_manager') else '❌ NOT FOUND'}")
        
        # Test Resource Managers directly
        print("\n3. Testing Resource Managers...")
        
        # Test Unified Resource Manager
        try:
            from core.unified_resource_manager import get_unified_resource_manager
            rm = get_unified_resource_manager()
            print("   ✅ Unified Resource Manager: WORKING")
        except Exception as e:
            print(f"   ❌ Unified Resource Manager: {e}")
        
        # Test High Memory Resource Manager
        try:
            from core.high_memory_resource_manager import get_high_memory_resource_manager
            hrm = get_high_memory_resource_manager()
            print("   ✅ High Memory Resource Manager: WORKING")
        except Exception as e:
            print(f"   ❌ High Memory Resource Manager: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 INTEGRATION SUMMARY:")
        
        overall_score = 0
        total_tests = 5
        
        if menu.resource_manager:
            print("✅ Master Menu RM Integration: PASS")
            overall_score += 1
        else:
            print("❌ Master Menu RM Integration: FAIL")
            
        if menu.menu_1:
            print("✅ Menu 1 Availability: PASS")
            overall_score += 1
        else:
            print("❌ Menu 1 Availability: FAIL")
            
        if hasattr(menu1, 'resource_manager') and menu1.resource_manager:
            print("✅ Menu 1 Unified RM: PASS")
            overall_score += 1
        else:
            print("❌ Menu 1 Unified RM: FAIL")
            
        if hasattr(menu1, 'enterprise_resource_manager'):
            print("✅ Menu 1 Enterprise RM: PASS")
            overall_score += 1
        else:
            print("❌ Menu 1 Enterprise RM: FAIL")
            
        if success:
            print("✅ Overall System Integration: PASS")
            overall_score += 1
        else:
            print("❌ Overall System Integration: FAIL")
        
        percentage = (overall_score / total_tests) * 100
        print(f"\n🎯 INTEGRATION SCORE: {overall_score}/{total_tests} ({percentage:.0f}%)")
        
        if percentage >= 80:
            print("🎉 EXCELLENT - Resource Management is well integrated!")
        elif percentage >= 60:
            print("⚠️ GOOD - Most systems integrated, some improvements needed")
        else:
            print("❌ NEEDS WORK - Significant integration issues detected")
            
        return overall_score >= 4
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_integration() 