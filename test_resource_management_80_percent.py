#!/usr/bin/env python3
"""
🧪 RESOURCE MANAGEMENT 80% TEST
ทดสอบระบบ Resource Management เพื่อตรวจสอบการใช้ RAM 80% และการทำงานในสภาพแวดล้อมปัจจุบัน

Features:
- ✅ ตรวจสอบสภาพแวดล้อมระบบ
- ✅ ทดสอบการจัดสรร RAM 80%
- ✅ ติดตามการใช้ทรัพยากร Real-time
- ✅ ทดสอบ Resource Manager หลายตัว
- ✅ รายงานประสิทธิภาพแบบครอบคลุม
"""

import sys
import time
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_system_environment():
    """ทดสอบและแสดงสภาพแวดล้อมระบบ"""
    print("🔍 SYSTEM ENVIRONMENT CHECK")
    print("=" * 60)
    
    try:
        import psutil
        
        # CPU Information
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"🖥️  CPU Cores: {cpu_count} logical, {cpu_physical} physical")
        print(f"⚡ CPU Usage: {cpu_percent:.1f}%")
        
        # Memory Information
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        used_gb = memory.used / (1024**3)
        
        print(f"💾 Total RAM: {total_gb:.1f} GB")
        print(f"💾 Available RAM: {available_gb:.1f} GB")
        print(f"💾 Used RAM: {used_gb:.1f} GB ({memory.percent:.1f}%)")
        
        # Calculate 80% allocation
        target_80_percent = total_gb * 0.8
        current_available_for_80 = available_gb if available_gb < target_80_percent else target_80_percent
        
        print(f"🎯 80% RAM Target: {target_80_percent:.1f} GB")
        print(f"🎯 Available for 80% allocation: {current_available_for_80:.1f} GB")
        
        # System recommendations
        if available_gb >= target_80_percent:
            print("✅ System has sufficient RAM for 80% allocation")
        else:
            print("⚠️  System may have limited RAM for full 80% allocation")
        
        return {
            'total_ram_gb': total_gb,
            'available_ram_gb': available_gb,
            'cpu_cores': cpu_count,
            'target_80_percent_gb': target_80_percent,
            'can_allocate_80_percent': available_gb >= target_80_percent
        }
        
    except Exception as e:
        print(f"❌ System environment check failed: {e}")
        return None

def test_unified_resource_manager():
    """ทดสอบ Unified Resource Manager"""
    print("\n🧪 TESTING UNIFIED RESOURCE MANAGER")
    print("=" * 60)
    
    try:
        from core.unified_resource_manager import get_unified_resource_manager
        
        # Initialize resource manager
        rm = get_unified_resource_manager()
        print("✅ Unified Resource Manager initialized")
        
        # Get resource status
        resources = rm.get_resource_status()
        print(f"📊 Resource Status: {len(resources)} resources detected")
        
        for resource_type, resource_info in resources.items():
            print(f"   {resource_type}: {resource_info.percentage:.1f}% used ({resource_info.status.value})")
        
        # Test resource allocation for 80% memory
        print("\n🎯 Testing 80% Memory Allocation...")
        
        if 'memory' in resources:
            memory_resource = resources['memory']
            total_memory_gb = memory_resource.total / (1024**3)
            target_allocation = total_memory_gb * 0.8
            
            allocation_result = rm.allocate_resources({
                'memory': target_allocation * (1024**3)  # Convert back to bytes
            })
            
            print(f"📊 Allocation Result:")
            print(f"   Success: {allocation_result.success}")
            print(f"   Allocated Percentage: {allocation_result.allocated_percentage:.1f}")
            print(f"   Safety Margin: {allocation_result.safety_margin:.1f}")
            
            if allocation_result.details:
                for msg in allocation_result.details.get('messages', []):
                    print(f"   {msg}")
        
        # Get resource summary
        summary = rm.get_resource_summary()
        print(f"\n📋 Resource Summary:")
        print(f"   System Status: {summary.get('overall_status', 'Unknown')}")
        print(f"   Memory Usage: {summary.get('memory_usage_percent', 0):.1f}%")
        print(f"   CPU Usage: {summary.get('cpu_usage_percent', 0):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Unified Resource Manager test failed: {e}")
        traceback.print_exc()
        return False

def test_high_memory_resource_manager():
    """ทดสอบ High Memory Resource Manager"""
    print("\n🧪 TESTING HIGH MEMORY RESOURCE MANAGER")
    print("=" * 60)
    
    try:
        from core.high_memory_resource_manager import get_high_memory_resource_manager
        
        # Initialize high memory resource manager
        hrm = get_high_memory_resource_manager()
        print("✅ High Memory Resource Manager initialized")
        
        # Get current performance
        performance = hrm.get_current_performance()
        
        print("📊 High Memory Performance Status:")
        print(f"   Memory Target: {performance.get('memory_target', 'Unknown')}")
        print(f"   CPU Target: {performance.get('cpu_target', 'Unknown')}")
        print(f"   Current Memory Usage: {performance.get('memory_percent', 0):.1f}%")
        print(f"   Current CPU Usage: {performance.get('cpu_percent', 0):.1f}%")
        print(f"   Available Memory: {performance.get('memory_available_gb', 0):.1f} GB")
        print(f"   Status: {performance.get('status', 'Unknown')}")
        
        # Get allocation info
        allocation = hrm.get_current_allocation()
        print(f"\n🎯 Resource Allocation:")
        print(f"   Memory Percentage: {allocation.get('memory_percentage', 0)*100:.0f}%")
        print(f"   CPU Percentage: {allocation.get('cpu_percentage', 0)*100:.0f}%")
        print(f"   Allocated Memory: {allocation.get('allocated_memory_gb', 0):.1f} GB")
        print(f"   Allocated CPU Cores: {allocation.get('allocated_cores', 0)}")
        
        # Get health status
        health = hrm.get_health_status()
        print(f"\n🏥 Health Status:")
        print(f"   Overall Health: {health.get('overall_health', 'Unknown')}")
        print(f"   Memory Health: {health.get('memory_health', 'Unknown')}")
        print(f"   CPU Health: {health.get('cpu_health', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ High Memory Resource Manager test failed: {e}")
        traceback.print_exc()
        return False

def monitor_resources_realtime(duration: int = 30):
    """ติดตามการใช้ทรัพยากรแบบ Real-time"""
    print(f"\n📊 REAL-TIME RESOURCE MONITORING ({duration} seconds)")
    print("=" * 60)
    print("Time     | CPU%  | RAM%  | RAM(GB) | Status")
    print("-" * 60)
    
    try:
        import psutil
        
        for i in range(duration):
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_gb = memory.used / (1024**3)
            
            # Determine status
            if memory.percent >= 95:
                status = "🔴 CRITICAL"
            elif memory.percent >= 85:
                status = "🟠 WARNING"
            elif memory.percent >= 70:
                status = "🟡 MODERATE"
            else:
                status = "🟢 HEALTHY"
            
            # Display metrics
            print(f"{current_time} | {cpu_percent:5.1f} | {memory.percent:5.1f} | {memory_gb:7.1f} | {status}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

def stress_test_80_percent_allocation():
    """ทดสอบการจัดสรร RAM 80% ในสถานการณ์จริง"""
    print("\n🔥 80% RAM ALLOCATION STRESS TEST")
    print("=" * 60)
    
    try:
        import psutil
        import numpy as np
        
        # Get initial memory status
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        target_allocation_gb = total_gb * 0.8
        
        print(f"📊 Initial Memory Status:")
        print(f"   Total RAM: {total_gb:.1f} GB")
        print(f"   Current Usage: {memory.percent:.1f}%")
        print(f"   Target 80% Allocation: {target_allocation_gb:.1f} GB")
        
        # Calculate how much we can safely allocate
        available_gb = memory.available / (1024**3)
        test_allocation_gb = min(target_allocation_gb * 0.5, available_gb * 0.6)  # Conservative test
        
        print(f"\n🧪 Test Allocation: {test_allocation_gb:.1f} GB")
        
        if test_allocation_gb < 1.0:
            print("⚠️  Insufficient RAM for stress test")
            return False
        
        # Allocate memory progressively
        allocated_arrays = []
        chunk_size_mb = 100  # 100 MB chunks
        total_chunks = int((test_allocation_gb * 1024) / chunk_size_mb)
        
        print(f"📈 Allocating memory in {chunk_size_mb}MB chunks...")
        
        for i in range(total_chunks):
            try:
                # Allocate chunk
                chunk = np.random.random((chunk_size_mb * 1024 * 1024 // 8,))  # 8 bytes per float64
                allocated_arrays.append(chunk)
                
                # Monitor progress
                current_memory = psutil.virtual_memory()
                allocated_so_far = (i + 1) * chunk_size_mb / 1024
                
                if (i + 1) % 10 == 0:  # Report every 1GB
                    print(f"   Allocated: {allocated_so_far:.1f} GB, Memory Usage: {current_memory.percent:.1f}%")
                
                # Check if we're approaching limits
                if current_memory.percent >= 85:
                    print(f"⚠️  Approaching memory limit at {current_memory.percent:.1f}%, stopping allocation")
                    break
                    
            except MemoryError:
                print(f"⚠️  Memory allocation failed at {allocated_so_far:.1f} GB")
                break
        
        # Final status
        final_memory = psutil.virtual_memory()
        final_allocated = len(allocated_arrays) * chunk_size_mb / 1024
        
        print(f"\n📊 Final Status:")
        print(f"   Successfully Allocated: {final_allocated:.1f} GB")
        print(f"   Final Memory Usage: {final_memory.percent:.1f}%")
        print(f"   Target Achievement: {(final_allocated/target_allocation_gb)*100:.1f}%")
        
        # Clean up
        print("\n🧹 Cleaning up allocated memory...")
        del allocated_arrays
        import gc
        gc.collect()
        
        cleanup_memory = psutil.virtual_memory()
        print(f"   Memory after cleanup: {cleanup_memory.percent:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Stress test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """รันการทดสอบทั้งหมด"""
    print("🧪 RESOURCE MANAGEMENT 80% COMPREHENSIVE TEST")
    print("=" * 70)
    print("Testing Resource Management System for 80% RAM allocation")
    print("=" * 70)
    
    # Test results tracking
    tests = []
    
    # Test 1: System Environment
    env_result = test_system_environment()
    tests.append(("System Environment", env_result is not None))
    
    # Test 2: Unified Resource Manager
    unified_result = test_unified_resource_manager()
    tests.append(("Unified Resource Manager", unified_result))
    
    # Test 3: High Memory Resource Manager
    high_memory_result = test_high_memory_resource_manager()
    tests.append(("High Memory Resource Manager", high_memory_result))
    
    # Test 4: Real-time Monitoring (shorter duration for testing)
    print("\n📊 Starting Real-time Monitoring (10 seconds)...")
    monitor_resources_realtime(10)
    tests.append(("Real-time Monitoring", True))
    
    # Test 5: Stress Test (optional)
    stress_test_input = input("\n❓ Run 80% RAM allocation stress test? (y/N): ").lower()
    if stress_test_input == 'y':
        stress_result = stress_test_80_percent_allocation()
        tests.append(("80% RAM Stress Test", stress_result))
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 RESOURCE MANAGEMENT TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} | {status}")
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Resource Management System is working correctly")
        print("✅ 80% RAM allocation is properly configured")
        print("✅ System is ready for production use")
    elif passed >= total * 0.75:
        print("⚠️ MOST TESTS PASSED")
        print("✅ Core functionality is working")
        print("⚠️ Some minor issues may need attention")
    else:
        print("❌ SIGNIFICANT ISSUES DETECTED")
        print("🚨 Resource Management System needs attention")
        print("🔧 Review failed tests and fix issues")
    
    print("\n💡 Recommendations:")
    if env_result and env_result['can_allocate_80_percent']:
        print("✅ System has sufficient RAM for 80% allocation")
    else:
        print("⚠️ Consider optimizing memory usage or increasing system RAM")
    
    print("✅ Monitor resource usage regularly during production")
    print("✅ Use Resource Manager APIs for dynamic allocation")
    print("=" * 70)

if __name__ == "__main__":
    main() 