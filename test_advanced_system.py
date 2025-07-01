#!/usr/bin/env python3
"""
Test Advanced Logger System
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_logger():
    """Test the advanced logger"""
    try:
        print("🔍 Testing Advanced Logger...")
        
        # Import advanced logger
        from core.advanced_logger import get_advanced_logger
        print("✅ Advanced logger import successful")
        
        # Create logger instance
        logger = get_advanced_logger("TEST_LOGGER")
        print("✅ Logger instance created")
        
        # Test different log levels
        logger.info("This is an info message")
        logger.success("This is a success message")
        logger.warning("This is a warning message")
        logger.debug("This is a debug message")
        
        # Test process tracking
        logger.start_process_tracking("test_process", "Test Process", 3)
        logger.update_process_progress("test_process", 1, "Step 1 completed")
        logger.update_process_progress("test_process", 2, "Step 2 completed")
        logger.update_process_progress("test_process", 3, "Step 3 completed")
        logger.complete_process("test_process", True)
        
        # Display performance summary
        logger.display_performance_summary()
        
        print("\n🎉 Advanced Logger Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_menu_1():
    """Test Menu 1 Advanced"""
    try:
        print("\n🌊 Testing Menu 1 Advanced...")
        
        # Import Menu 1 Advanced
        from menu_modules.menu_1_elliott_wave_advanced import Menu1ElliottWaveAdvanced
        print("✅ Menu 1 Advanced import successful")
        
        # Create menu instance
        menu = Menu1ElliottWaveAdvanced()
        print("✅ Menu 1 instance created")
        
        # Get menu info
        info = menu.get_menu_info()
        print(f"✅ Menu Info: {info['menu_name']}")
        
        print("🎉 Menu 1 Advanced Test Completed!")
        return True
        
    except Exception as e:
        print(f"❌ Menu 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Advanced System Tests...")
    print("="*60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test logger
    if test_logger():
        tests_passed += 1
    
    # Test Menu 1
    if test_menu_1():
        tests_passed += 1
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
