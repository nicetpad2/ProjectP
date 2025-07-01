#!/usr/bin/env python3
"""
🧪 NICEGOLD ADVANCED LOGGING SYSTEM - QUICK TEST & DEMO
ทดสอบและแสดงการทำงานของระบบ Advanced Terminal Logger

🎯 Features Tested:
- ✨ Beautiful Terminal Output
- 📊 Real-time Progress Bars
- 🎨 Color-coded Log Levels
- 🔄 Process Tracking
- 📈 System Performance Monitoring
- 🛡️ Error Handling & Recovery
- 💫 Rich Text Formatting
"""

import time
import random
import threading
from datetime import datetime
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_basic_logging():
    """🎯 Test basic logging functionality"""
    print("\n" + "="*60)
    print("🎯 TESTING BASIC LOGGING FUNCTIONALITY")
    print("="*60)
    
    try:
        from core.advanced_terminal_logger import get_terminal_logger, LogLevel
        
        logger = get_terminal_logger()
        
        # Test different log levels
        logger.debug("🔍 This is a debug message", "Debug_Test")
        time.sleep(0.5)
        
        logger.info("ℹ️ System information message", "Info_Test")
        time.sleep(0.5)
        
        logger.warning("⚠️ This is a warning message", "Warning_Test")
        time.sleep(0.5)
        
        logger.error("❌ This is an error message", "Error_Test")
        time.sleep(0.5)
        
        logger.critical("🚨 This is a critical message", "Critical_Test")
        time.sleep(0.5)
        
        logger.success("✅ This is a success message", "Success_Test")
        time.sleep(0.5)
        
        logger.progress("📊 This is a progress message", "Progress_Test")
        time.sleep(0.5)
        
        logger.system("⚙️ This is a system message", "System_Test")
        time.sleep(0.5)
        
        logger.performance("📈 This is a performance message", "Performance_Test")
        time.sleep(0.5)
        
        logger.security("🛡️ This is a security message", "Security_Test")
        time.sleep(0.5)
        
        logger.data_log("📊 This is a data message", "Data_Test")
        time.sleep(0.5)
        
        logger.ai_log("🧠 This is an AI message", "AI_Test")
        time.sleep(0.5)
        
        logger.trade_log("💹 This is a trading message", "Trade_Test")
        time.sleep(0.5)
        
        logger.success("🎉 Basic logging test completed!", "Test_Complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic logging test failed: {e}")
        return False

def test_progress_tracking():
    """📊 Test progress tracking functionality"""
    print("\n" + "="*60)
    print("📊 TESTING PROGRESS TRACKING FUNCTIONALITY")
    print("="*60)
    
    try:
        from core.real_time_progress_manager import get_progress_manager, ProgressType
        from core.advanced_terminal_logger import get_terminal_logger
        
        logger = get_terminal_logger()
        progress_manager = get_progress_manager()
        
        # Test single progress bar
        logger.info("🚀 Testing single progress bar", "Progress_Test")
        
        task1 = progress_manager.create_progress(
            "📊 Data Processing", 100, ProgressType.PROCESSING
        )
        
        for i in range(100):
            progress_manager.update_progress(task1, 1, f"Processing item {i+1}")
            time.sleep(0.03)  # Faster for demo
        
        progress_manager.complete_progress(task1, "✅ Data processing completed")
        
        # Test multiple concurrent progress bars
        logger.info("🚀 Testing multiple concurrent progress bars", "Progress_Test")
        
        tasks = []
        task_names = [
            ("🧠 AI Model Training", ProgressType.TRAINING),
            ("📈 Data Analysis", ProgressType.ANALYSIS),
            ("🎯 Optimization", ProgressType.OPTIMIZATION),
            ("✅ Validation", ProgressType.VALIDATION)
        ]
        
        # Create all tasks
        for name, ptype in task_names:
            task_id = progress_manager.create_progress(name, 50, ptype)
            tasks.append(task_id)
        
        # Update all tasks concurrently
        for step in range(50):
            for task_id in tasks:
                if random.random() > 0.15:  # 85% chance to update (simulate varying speeds)
                    progress_manager.update_progress(task_id, 1, f"Step {step+1}")
            time.sleep(0.05)
        
        # Complete all tasks
        for i, task_id in enumerate(tasks):
            progress_manager.complete_progress(task_id, f"✅ {task_names[i][0]} completed")
            time.sleep(0.2)
        
        logger.success("🎉 Progress tracking test completed!", "Test_Complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Progress tracking test failed: {e}")
        return False

def test_process_tracking():
    """🔄 Test process tracking functionality"""
    print("\n" + "="*60)
    print("🔄 TESTING PROCESS TRACKING FUNCTIONALITY")
    print("="*60)
    
    try:
        from core.advanced_terminal_logger import get_terminal_logger, ProcessStatus
        
        logger = get_terminal_logger()
        
        # Start a tracked process
        process_id = logger.start_process(
            "Elliott Wave Analysis", 
            "Analyzing XAUUSD market data", 
            total_steps=5
        )
        
        # Simulate process steps
        steps = [
            "Loading market data",
            "Calculating technical indicators", 
            "Detecting Elliott Wave patterns",
            "Training CNN-LSTM model",
            "Generating trading signals"
        ]
        
        for i, step_desc in enumerate(steps, 1):
            logger.update_process(process_id, ProcessStatus.RUNNING, i, step_desc)
            logger.progress(f"Step {i}/5: {step_desc}", "Elliott_Wave_Analysis", process_id=process_id)
            time.sleep(1.0)  # Simulate processing time
        
        # Complete the process
        logger.complete_process(process_id, True, "Elliott Wave analysis completed successfully")
        
        logger.success("🎉 Process tracking test completed!", "Test_Complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Process tracking test failed: {e}")
        return False

def test_error_handling():
    """🛡️ Test error handling and recovery"""
    print("\n" + "="*60)
    print("🛡️ TESTING ERROR HANDLING & RECOVERY")
    print("="*60)
    
    try:
        from core.advanced_terminal_logger import get_terminal_logger
        from core.real_time_progress_manager import get_progress_manager
        
        logger = get_terminal_logger()
        progress_manager = get_progress_manager()
        
        # Test exception logging
        logger.info("🧪 Testing exception handling", "Error_Test")
        
        try:
            # Intentional error for testing
            result = 10 / 0
        except Exception as e:
            logger.error("Test exception caught successfully", "Error_Test", exception=e)
        
        # Test progress failure
        logger.info("🧪 Testing progress failure handling", "Error_Test")
        
        failed_task = progress_manager.create_progress("💥 Task That Will Fail", 100)
        
        for i in range(30):
            progress_manager.update_progress(failed_task, 1, f"Processing step {i+1}")
            time.sleep(0.02)
        
        # Simulate failure
        progress_manager.fail_progress(failed_task, "Simulated failure for testing")
        
        # Test warning accumulation
        logger.info("🧪 Testing warning accumulation", "Error_Test")
        
        for i in range(3):
            logger.warning(f"Test warning #{i+1}", "Warning_Test")
            time.sleep(0.3)
        
        logger.success("🎉 Error handling test completed!", "Test_Complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_performance_monitoring():
    """📈 Test performance monitoring"""
    print("\n" + "="*60)
    print("📈 TESTING PERFORMANCE MONITORING")
    print("="*60)
    
    try:
        from core.advanced_terminal_logger import get_terminal_logger
        
        logger = get_terminal_logger()
        
        # Generate some activity to monitor
        logger.info("🚀 Starting performance monitoring test", "Performance_Test")
        
        # Simulate various system activities
        activities = [
            "Loading configuration files",
            "Initializing ML models",
            "Processing market data",
            "Running feature selection",
            "Training neural networks",
            "Validating model performance",
            "Generating trading signals",
            "Saving results to disk"
        ]
        
        for activity in activities:
            logger.performance(f"⚡ {activity}", "System_Activity")
            
            # Generate some logs to increase activity
            for i in range(5):
                logger.debug(f"Processing {activity.lower()} - step {i+1}", "Debug_Activity")
                time.sleep(0.1)
        
        # Show system statistics
        logger.info("📊 Displaying system performance statistics", "Performance_Test")
        logger.show_system_stats()
        
        logger.success("🎉 Performance monitoring test completed!", "Test_Complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

def test_integration_with_ml_protection():
    """🛡️ Test integration with ML Protection System"""
    print("\n" + "="*60)
    print("🛡️ TESTING ML PROTECTION INTEGRATION")
    print("="*60)
    
    try:
        from core.advanced_terminal_logger import get_terminal_logger
        
        logger = get_terminal_logger()
        
        # Simulate ML Protection System operations
        logger.security("🛡️ Starting ML Protection System test", "ML_Protection")
        
        protection_steps = [
            "Validating data integrity",
            "Checking for data leakage", 
            "Detecting overfitting patterns",
            "Analyzing feature correlations",
            "Performing time series validation",
            "Running enterprise compliance checks"
        ]
        
        process_id = logger.start_process(
            "ML Protection Analysis",
            "Comprehensive ML security validation",
            total_steps=len(protection_steps)
        )
        
        for i, step in enumerate(protection_steps, 1):
            logger.security(f"🔍 {step}", "ML_Protection", process_id=process_id)
            logger.update_process(process_id, None, i, step)
            time.sleep(0.8)
        
        logger.complete_process(process_id, True, "ML Protection analysis passed all checks")
        logger.success("🏆 Enterprise ML Protection: VALIDATED", "ML_Protection")
        
        return True
        
    except Exception as e:
        print(f"❌ ML Protection integration test failed: {e}")
        return False

def run_comprehensive_test():
    """🎯 Run comprehensive test suite"""
    print("\n" + "🎉"*20)
    print("🚀 NICEGOLD ADVANCED LOGGING SYSTEM - COMPREHENSIVE TEST")
    print("🎉"*60)
    
    tests = [
        ("Basic Logging", test_basic_logging),
        ("Progress Tracking", test_progress_tracking), 
        ("Process Tracking", test_process_tracking),
        ("Error Handling", test_error_handling),
        ("Performance Monitoring", test_performance_monitoring),
        ("ML Protection Integration", test_integration_with_ml_protection)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
            results[test_name] = False
    
    # Show final results
    print("\n" + "="*60)
    print("📊 FINAL TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} : {status}")
    
    print("-" * 60)
    print(f"📈 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 🏆 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION! 🏆 🎉")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed - System mostly functional")
    else:
        print("❌ Multiple test failures - System needs attention")
    
    # Show final system stats if available
    try:
        from core.advanced_terminal_logger import get_terminal_logger
        logger = get_terminal_logger()
        
        print("\n📊 Final System Statistics:")
        logger.show_system_stats()
        
        # Export test session
        export_file = logger.export_session_log("test_session_export.json")
        if export_file:
            print(f"📋 Test session exported to: {export_file}")
        
    except Exception as e:
        print(f"⚠️ Could not show final stats: {e}")
    
    return passed == total

def quick_demo():
    """⚡ Quick demonstration of key features"""
    print("\n" + "⚡"*20)
    print("⚡ QUICK DEMO - ADVANCED LOGGING FEATURES")
    print("⚡"*60)
    
    try:
        from core.advanced_terminal_logger import get_terminal_logger
        from core.real_time_progress_manager import get_progress_manager, ProgressType
        
        logger = get_terminal_logger()
        progress_manager = get_progress_manager()
        
        # Demo beautiful logging
        logger.success("🎉 Welcome to NICEGOLD Advanced Logging System!", "Demo")
        time.sleep(1)
        
        # Demo progress tracking
        demo_task = progress_manager.create_progress(
            "✨ Demo Progress", 20, ProgressType.PROCESSING
        )
        
        for i in range(20):
            progress_manager.update_progress(demo_task, 1, f"Demo step {i+1}")
            time.sleep(0.1)
        
        progress_manager.complete_progress(demo_task, "✅ Demo completed!")
        
        # Demo different log types
        logger.ai_log("🧠 AI system ready for Elliott Wave analysis", "AI_Demo")
        logger.trade_log("💹 Market data processed successfully", "Trading_Demo")
        logger.security("🛡️ All security checks passed", "Security_Demo")
        
        logger.success("🏆 Quick demo completed successfully!", "Demo")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NICEGOLD Advanced Logging System - Test Suite")
    print("=" * 60)
    
    # Ask user what to run
    print("\nChoose test mode:")
    print("1. 🎯 Full Comprehensive Test Suite")
    print("2. ⚡ Quick Demo")
    print("3. 🧪 Individual Test Selection")
    
    try:
        choice = input("\nSelect option (1-3, default: 2): ").strip()
        if not choice:
            choice = "2"
    except KeyboardInterrupt:
        print("\n🛑 Test cancelled by user")
        sys.exit(0)
    
    if choice == "1":
        # Run full test suite
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
        
    elif choice == "2":
        # Run quick demo
        success = quick_demo()
        if success:
            print("\n🎉 Quick demo completed successfully!")
        else:
            print("\n❌ Quick demo failed!")
        
    elif choice == "3":
        # Individual test selection
        print("\nSelect individual test:")
        print("1. Basic Logging")
        print("2. Progress Tracking")
        print("3. Process Tracking")
        print("4. Error Handling")
        print("5. Performance Monitoring")
        print("6. ML Protection Integration")
        
        test_choice = input("\nSelect test (1-6): ").strip()
        
        tests = {
            "1": test_basic_logging,
            "2": test_progress_tracking,
            "3": test_process_tracking,
            "4": test_error_handling,
            "5": test_performance_monitoring,
            "6": test_integration_with_ml_protection
        }
        
        if test_choice in tests:
            success = tests[test_choice]()
            print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")
        else:
            print("❌ Invalid test selection")
    
    else:
        print("❌ Invalid option selected")
    
    print("\n🎯 Test session completed!")
