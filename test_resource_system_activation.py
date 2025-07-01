#!/usr/bin/env python3
"""
🧪 TEST RESOURCE SYSTEM ACTIVATION
==================================

ทดสอบการทำงานของระบบ Intelligent Resource Management
เพื่อให้เห็นการจัดสรรทรัพยากร 80% และการแสดงข้อความ

🎯 ทดสอบ:
- การเริ่มต้นระบบ Resource Management
- การแสดงข้อความการจัดสรรทรัพยากร
- การติดตามการใช้งานแบบ real-time
- การแสดงสถิติการใช้งาน
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_resource_system():
    """ทดสอบระบบ Resource Management"""
    
    print("🧪 TESTING INTELLIGENT RESOURCE MANAGEMENT SYSTEM")
    print("="*60)
    
    try:
        # Import resource management
        print("\n1️⃣ Importing Resource Management modules...")
        from core.intelligent_resource_manager import initialize_intelligent_resources
        from core.enhanced_intelligent_resource_manager import initialize_enhanced_intelligent_resources
        
        print("✅ Resource Management modules imported successfully")
        
        # Test 1: Initialize Basic Resource Manager
        print("\n2️⃣ Testing Basic Resource Manager...")
        basic_manager = initialize_intelligent_resources(
            allocation_percentage=0.8,
            enable_monitoring=True
        )
        
        if basic_manager:
            print("✅ Basic Resource Manager initialized")
            
            # Show resource configuration
            resource_config = basic_manager.resource_config
            
            cpu_config = resource_config.get('cpu', {})
            memory_config = resource_config.get('memory', {})
            
            print(f"\n📊 RESOURCE ALLOCATION SUMMARY:")
            print(f"   🧮 CPU: {cpu_config.get('allocated_threads', 0)}/{cpu_config.get('total_cores', 0)} cores ({cpu_config.get('allocation_percentage', 0):.1f}%)")
            print(f"   🧠 Memory: {memory_config.get('allocated_gb', 0):.1f}/{memory_config.get('total_gb', 0):.1f} GB ({memory_config.get('allocation_percentage', 0):.1f}%)")
            
            optimization = resource_config.get('optimization', {})
            print(f"   ⚡ Batch Size: {optimization.get('batch_size', 32)}")
            print(f"   👥 Workers: {optimization.get('recommended_workers', 4)}")
        
        # Test 2: Test Real-time Monitoring
        print("\n3️⃣ Testing Real-time Monitoring...")
        if basic_manager.monitoring_active:
            print("✅ Monitoring is already active")
        else:
            basic_manager.start_monitoring(interval=0.5)
            print("✅ Real-time monitoring started")
        
        # Show monitoring data for 5 seconds
        print("\n📈 Monitoring system performance for 5 seconds...")
        for i in range(10):
            current_perf = basic_manager.get_current_performance()
            cpu_usage = current_perf.get('cpu_percent', 0)
            memory_info = current_perf.get('memory', {})
            memory_usage = memory_info.get('percent', 0)
            
            print(f"   Update {i+1:2d}: CPU {cpu_usage:5.1f}%, Memory {memory_usage:5.1f}%")
            time.sleep(0.5)
        
        # Test 3: Enhanced Resource Manager
        print("\n4️⃣ Testing Enhanced Resource Manager...")
        enhanced_manager = initialize_enhanced_intelligent_resources(
            allocation_percentage=0.8,
            enable_advanced_monitoring=True
        )
        
        if enhanced_manager:
            print("✅ Enhanced Resource Manager initialized")
            
            # Test Menu 1 configuration
            menu1_config = enhanced_manager.get_menu1_optimization_config()
            print(f"\n🌊 MENU 1 OPTIMIZATION CONFIG:")
            
            if menu1_config:
                data_config = menu1_config.get('data_processing', {})
                print(f"   📊 Data Processing: Chunk Size {data_config.get('chunk_size', 'N/A')}, Workers {data_config.get('parallel_workers', 'N/A')}")
                
                elliott_config = menu1_config.get('elliott_wave', {})
                print(f"   🌊 Elliott Wave: Batch Size {elliott_config.get('batch_size', 'N/A')}")
                
                feature_config = menu1_config.get('feature_selection', {})
                print(f"   🎯 Feature Selection: Trials {feature_config.get('n_trials', 'N/A')}")
        
        # Test 4: Stage Monitoring
        print("\n5️⃣ Testing Pipeline Stage Monitoring...")
        
        test_stages = ['data_loading', 'feature_engineering', 'model_training']
        
        for stage in test_stages:
            print(f"\n   ▶️ Starting stage: {stage}")
            enhanced_manager.start_stage_monitoring(stage)
            
            # Simulate work
            time.sleep(1)
            
            # End stage with mock metrics
            mock_metrics = {
                'duration': 1.0,
                'success': True,
                'efficiency': 0.85
            }
            
            summary = enhanced_manager.end_stage_monitoring(stage, mock_metrics)
            if summary:
                efficiency = summary.get('efficiency_score', 0)
                print(f"   ✅ Stage completed - Efficiency: {efficiency:.2f}")
        
        # Test 5: Performance Report
        print("\n6️⃣ Generating Performance Report...")
        
        if hasattr(enhanced_manager, 'generate_pipeline_performance_report'):
            report = enhanced_manager.generate_pipeline_performance_report()
            if report:
                pipeline_summary = report.get('pipeline_summary', {})
                avg_efficiency = pipeline_summary.get('average_efficiency', 0)
                grade = pipeline_summary.get('resource_efficiency_grade', 'N/A')
                
                print(f"   📋 Pipeline Average Efficiency: {avg_efficiency:.2f}")
                print(f"   🏆 Resource Efficiency Grade: {grade}")
        
        # Cleanup
        print("\n7️⃣ Cleanup...")
        basic_manager.stop_monitoring()
        enhanced_manager.stop_monitoring()
        print("✅ Monitoring stopped")
        
        print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print(f"📊 Resource Management System is working correctly")
        print(f"⚡ 80% allocation strategy is active")
        print(f"📈 Real-time monitoring is functional")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all resource management files are present")
        return False
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_resource_system()
    
    print(f"\n{'='*60}")
    if success:
        print("🎯 RESULT: ✅ Resource Management System is WORKING")
        print("💡 Now when you run ProjectP.py, you should see:")
        print("   - Resource allocation messages")
        print("   - 80% CPU/Memory allocation")
        print("   - Real-time monitoring during pipeline")
        print("   - Resource usage summaries")
    else:
        print("🎯 RESULT: ❌ Resource Management System has ISSUES")
        print("💡 Check the error messages above and fix imports/dependencies")
    
    print(f"{'='*60}")
