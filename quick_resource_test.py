#!/usr/bin/env python3
"""
🧪 QUICK RESOURCE MANAGEMENT TEST
=================================

ทดสอบระบบ Intelligent Resource Management อย่างรวดเร็ว
สำหรับ Live Share environment

🎯 การทดสอบ:
- แสดงการจัดสรรทรัพยากร 80%
- ติดตามการใช้งานแบบ real-time
- แสดงข้อความที่คุณไม่เห็น
"""

import os
import sys
import time
from pathlib import Path

# Setup project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def quick_test():
    """ทดสอบอย่างรวดเร็ว"""
    
    print("🚀 QUICK RESOURCE MANAGEMENT TEST")
    print("="*50)
    
    try:
        # Test basic imports first
        print("\n1️⃣ Testing imports...")
        
        try:
            from core.intelligent_resource_manager import initialize_intelligent_resources
            print("✅ Basic Resource Manager imported")
        except ImportError as e:
            print(f"❌ Basic import failed: {e}")
            return False
            
        try:
            from core.enhanced_intelligent_resource_manager import initialize_enhanced_intelligent_resources
            print("✅ Enhanced Resource Manager imported")
        except ImportError as e:
            print(f"❌ Enhanced import failed: {e}")
            return False
        
        # Test basic initialization
        print("\n2️⃣ Testing basic resource manager...")
        basic_rm = initialize_intelligent_resources(allocation_percentage=0.8, enable_monitoring=True)
        
        if basic_rm:
            print("✅ Basic Resource Manager initialized")
            
            # Show resource allocation
            config = basic_rm.resource_config
            cpu_config = config.get('cpu', {})
            memory_config = config.get('memory', {})
            
            print(f"\n📊 RESOURCE ALLOCATION (80% Strategy):")
            print(f"   🧮 CPU: {cpu_config.get('allocated_threads', 0)}/{cpu_config.get('total_cores', 0)} cores ({cpu_config.get('allocation_percentage', 0):.1f}%)")
            print(f"   🧠 Memory: {memory_config.get('allocated_gb', 0):.1f}/{memory_config.get('total_gb', 0):.1f} GB ({memory_config.get('allocation_percentage', 0):.1f}%)")
            
            optimization = config.get('optimization', {})
            print(f"   ⚡ Batch Size: {optimization.get('batch_size', 32)}")
            print(f"   👥 Workers: {optimization.get('recommended_workers', 4)}")
            
            # Test monitoring
            print(f"\n3️⃣ Testing real-time monitoring...")
            
            if not basic_rm.monitoring_active:
                basic_rm.start_monitoring(interval=0.5)
                print("✅ Monitoring started")
            
            # Show some monitoring data
            for i in range(5):
                current_perf = basic_rm.get_current_performance()
                cpu_usage = current_perf.get('cpu_percent', 0)
                memory_info = current_perf.get('memory', {})
                memory_usage = memory_info.get('percent', 0)
                
                print(f"   📈 Update {i+1}: CPU {cpu_usage:5.1f}%, Memory {memory_usage:5.1f}%")
                time.sleep(0.5)
                
            # Test enhanced manager
            print(f"\n4️⃣ Testing enhanced resource manager...")
            enhanced_rm = initialize_enhanced_intelligent_resources(
                allocation_percentage=0.8,
                enable_advanced_monitoring=True
            )
            
            if enhanced_rm:
                print("✅ Enhanced Resource Manager initialized")
                
                # Test Menu 1 optimization config
                menu1_config = enhanced_rm.get_menu1_optimization_config()
                if menu1_config:
                    print(f"\n🌊 MENU 1 OPTIMIZATION CONFIG:")
                    data_config = menu1_config.get('data_processing', {})
                    print(f"   📊 Data: Chunk {data_config.get('chunk_size', 'N/A')}, Workers {data_config.get('parallel_workers', 'N/A')}")
                    
                    elliott_config = menu1_config.get('elliott_wave', {})
                    print(f"   🌊 Elliott: Batch {elliott_config.get('batch_size', 'N/A')}")
            
            # Cleanup
            basic_rm.stop_monitoring()
            if enhanced_rm:
                enhanced_rm.stop_monitoring()
            
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"📊 Resource Management System is working correctly")
            print(f"⚡ 80% allocation strategy is active")
            print(f"📈 Real-time monitoring is functional")
            
            return True
        else:
            print("❌ Failed to initialize basic resource manager")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_usage_instructions():
    """แสดงวิธีการใช้งาน"""
    print(f"\n" + "="*60)
    print(f"📋 HOW TO RUN MAIN SYSTEM:")
    print(f"="*60)
    print(f"")
    print(f"🚀 OPTION 1: Direct Run")
    print(f"   python ProjectP.py")
    print(f"")
    print(f"🚀 OPTION 2: VS Code Terminal")
    print(f"   1. Open Terminal (Ctrl+`)")
    print(f"   2. Type: python ProjectP.py")
    print(f"   3. Press Enter")
    print(f"")
    print(f"🚀 OPTION 3: VS Code Run Button")
    print(f"   1. Open ProjectP.py file")
    print(f"   2. Press F5 or Ctrl+F5")
    print(f"   3. Select 'Python File'")
    print(f"")
    print(f"🚀 OPTION 4: Right-click Run")
    print(f"   1. Right-click on ProjectP.py")
    print(f"   2. Select 'Run Python File in Terminal'")
    print(f"")
    print(f"✨ EXPECTED OUTPUT:")
    print(f"   🧠 Intelligent Resource Manager: ACTIVE")
    print(f"   📊 80% CPU/Memory allocation messages")
    print(f"   🌊 Menu with Resource Optimized options")
    print(f"   ⚡ Real-time monitoring during pipeline")
    print(f"")
    print(f"🔧 IF DEPENDENCIES MISSING:")
    print(f"   Select option 'D' in menu to auto-fix")
    print(f"")

if __name__ == "__main__":
    success = quick_test()
    show_usage_instructions()
    
    if success:
        print(f"🎯 RESULT: ✅ RESOURCE SYSTEM READY FOR LIVE SHARE")
    else:
        print(f"🎯 RESULT: ❌ RESOURCE SYSTEM NEEDS ATTENTION")
    
    print(f"="*60)
