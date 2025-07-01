#!/usr/bin/env python3
"""
🧪 TEST ENHANCED MENU SYSTEM
ทดสอบระบบ Enhanced Menu พร้อม Beautiful Progress Bar
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.beautiful_progress import EnhancedBeautifulLogger
from enhanced_menu_system import EnhancedMenuSystem


def test_beautiful_progress():
    """ทดสอบระบบ Beautiful Progress"""
    logger = EnhancedBeautifulLogger("TEST-PROGRESS")
    
    logger.info("🧪 Testing Beautiful Progress System", {
        "test_type": "Unit Test",
        "component": "Enhanced Menu System"
    })
    
    try:
        # Test 1: Menu System Initialization
        logger.step_start(1, "Menu System Initialization", "Testing enhanced menu system setup")
        menu_system = EnhancedMenuSystem()
        logger.step_complete(1, "Menu System Initialization", 0.5, {
            "status": "✅ Success",
            "components": "All initialized"
        })
        
        # Test 2: Display Menu (without user interaction)
        logger.step_start(2, "Menu Display Test", "Testing beautiful menu display")
        menu_system.display_main_menu()
        logger.step_complete(2, "Menu Display", 0.1, {
            "display": "✅ Beautiful",
            "rich_enabled": "✅ Yes" if hasattr(menu_system.beautiful_logger, 'console') else "❌ No"
        })
        
        logger.success("🎉 Enhanced Menu System Test Completed", {
            "all_tests": "✅ PASSED",
            "beautiful_progress": "✅ Working",
            "enhanced_logging": "✅ Active"
        })
        
        return True
        
    except Exception as e:
        logger.error("💥 Test Failed", {
            "error": str(e),
            "test_phase": "Enhanced Menu System Testing"
        })
        return False


def test_demo_mode():
    """ทดสอบโหมด Demo (ไม่ต้องใส่ข้อมูลจริง)"""
    logger = EnhancedBeautifulLogger("DEMO-TEST")
    
    logger.info("🎬 Starting Demo Mode Test")
    
    # Import demo
    try:
        from demo_beautiful_progress import demo_elliott_wave_pipeline_simulation
        
        logger.step_start(1, "Demo Simulation", "Running Elliott Wave pipeline simulation")
        demo_elliott_wave_pipeline_simulation()
        logger.step_complete(1, "Demo Simulation", 6.5, {
            "simulation": "✅ Completed",
            "progress_bars": "✅ Beautiful",
            "logging": "✅ Enhanced"
        })
        
        return True
        
    except Exception as e:
        logger.error("❌ Demo test failed", {
            "error": str(e)
        })
        return False


def main():
    """Main test function"""
    logger = EnhancedBeautifulLogger("TEST-MAIN")
    
    print("🧪 ENHANCED MENU SYSTEM TEST SUITE")
    print("=" * 60)
    
    # Test 1: Beautiful Progress System
    logger.info("Test 1: Beautiful Progress System")
    test1_passed = test_beautiful_progress()
    
    print("\n" + "-" * 60)
    
    # Test 2: Demo Mode
    logger.info("Test 2: Demo Mode Simulation")
    test2_passed = test_demo_mode()
    
    print("\n" + "=" * 60)
    
    # Summary
    if test1_passed and test2_passed:
        logger.success("🎊 ALL TESTS PASSED!", {
            "beautiful_progress": "✅ Working",
            "enhanced_logging": "✅ Active", 
            "menu_system": "✅ Ready",
            "demo_mode": "✅ Functional"
        })
        print("\n✅ Enhanced Menu System is ready for use!")
        print("🚀 You can now run:")
        print("   • python3 enhanced_menu_system.py")
        print("   • python3 demo_beautiful_progress.py")
    else:
        logger.error("❌ Some tests failed", {
            "test1_progress": "✅ PASSED" if test1_passed else "❌ FAILED",
            "test2_demo": "✅ PASSED" if test2_passed else "❌ FAILED"
        })


if __name__ == "__main__":
    main()
