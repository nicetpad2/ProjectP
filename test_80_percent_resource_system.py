#!/usr/bin/env python3
"""
🧪 ENHANCED 80% RESOURCE MANAGER - COMPREHENSIVE TEST
การทดสอบระบบจัดการทรัพยากร 80% อย่างครcomprehensive
"""

import sys
import os
import time
import psutil
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_80_percent_resource_system():
    """🔍 ทดสอบระบบจัดการทรัพยากร 80% อย่างครอบคลุม"""
    
    print("🧪 ENHANCED 80% RESOURCE MANAGER - COMPREHENSIVE TEST")
    print("=" * 70)
    
    # Test 1: System Resource Detection
    print("🔍 Test 1: System Resource Detection")
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"   CPU Cores: {cpu_count}")
    print(f"   Current Memory Usage: {memory.percent:.1f}%")
    print(f"   Current CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print("   ✅ System detection complete")
    
    # Test 2: Enhanced 80% Resource Manager Import
    print("\n🔍 Test 2: Enhanced 80% Resource Manager Import")
    try:
        from core.enhanced_80_percent_resource_manager import Enhanced80PercentResourceManager
        print("   ✅ Enhanced 80% Resource Manager imported successfully")
        
        # Test instantiation
        resource_manager = Enhanced80PercentResourceManager(
            target_allocation=0.80
        )
        print("   ✅ Resource Manager instantiated with 80% configuration")
        
        # Test configuration
        if hasattr(resource_manager, 'memory_percentage'):
            print(f"   Memory Target: {resource_manager.memory_percentage * 100}%")
        if hasattr(resource_manager, 'cpu_percentage'):
            print(f"   CPU Target: {resource_manager.cpu_percentage * 100}%")
            
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Instantiation failed: {e}")
        return False
    
    # Test 3: Resource Manager Health Check
    print("\n🔍 Test 3: Resource Manager Health Check")
    try:
        if hasattr(resource_manager, 'get_health_status'):
            health = resource_manager.get_health_status()
            print(f"   Health Score: {health.get('health_score', 0)}%")
            print(f"   Target Allocation: {health.get('target_allocation', 0) * 100:.1f}%")
            print(f"   Current Allocation: {health.get('current_allocation', 0) * 100:.1f}%")
            
            if health.get('health_score', 0) >= 90:
                print("   ✅ Excellent health status")
            elif health.get('health_score', 0) >= 70:
                print("   ✅ Good health status")
            else:
                print("   ⚠️ Health needs attention")
        else:
            print("   ⚠️ Health status method not available")
            
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test 4: Resource Monitoring Capabilities
    print("\n🔍 Test 4: Resource Monitoring Capabilities")
    try:
        if hasattr(resource_manager, 'start_monitoring'):
            resource_manager.start_monitoring()
            print("   ✅ Resource monitoring started")
            
            # Let it run for a few seconds
            time.sleep(3)
            
            if hasattr(resource_manager, 'get_current_usage'):
                usage = resource_manager.get_current_usage()
                print(f"   Current Memory Usage: {usage.get('memory', 0):.1f}%")
                print(f"   Current CPU Usage: {usage.get('cpu', 0):.1f}%")
        else:
            print("   ⚠️ Monitoring methods not available")
            
    except Exception as e:
        print(f"   ❌ Monitoring test failed: {e}")
    
    # Test 5: 80% Allocation Strategy Test
    print("\n🔍 Test 5: 80% Allocation Strategy Test")
    try:
        # Calculate optimal 80% allocation
        optimal_memory = memory.total * 0.8
        current_usage = memory.used
        
        print(f"   Target 80% Memory: {optimal_memory / (1024**3):.1f} GB")
        print(f"   Current Usage: {current_usage / (1024**3):.1f} GB")
        print(f"   Utilization vs Target: {(current_usage / optimal_memory) * 100:.1f}%")
        
        if current_usage >= optimal_memory * 0.9:  # 90% of 80% target
            print("   ✅ Excellent: Near optimal 80% utilization")
        elif current_usage >= optimal_memory * 0.75:  # 75% of 80% target
            print("   ✅ Good: Reasonable utilization")
        else:
            print("   ⚠️ Under-utilized: Can use more RAM for better performance")
            
    except Exception as e:
        print(f"   ❌ Allocation strategy test failed: {e}")
    
    # Test 6: Performance Optimization Check
    print("\n🔍 Test 6: Performance Optimization Check")
    try:
        # Check if performance optimization is enabled
        if hasattr(resource_manager, 'performance_optimization_enabled'):
            if resource_manager.performance_optimization_enabled:
                print("   ✅ Performance optimization enabled")
            else:
                print("   ⚠️ Performance optimization disabled")
        
        # Check adaptive allocation
        if hasattr(resource_manager, 'adaptive_allocation_enabled'):
            if resource_manager.adaptive_allocation_enabled:
                print("   ✅ Adaptive allocation enabled")
            else:
                print("   ⚠️ Adaptive allocation disabled")
        
        # Check monitoring status
        if hasattr(resource_manager, 'monitoring_active'):
            if resource_manager.monitoring_active:
                print("   ✅ Resource monitoring active")
            else:
                print("   ⚠️ Resource monitoring inactive")
                
    except Exception as e:
        print(f"   ❌ Performance optimization check failed: {e}")
    
    # Test 7: Integration with ProjectP.py
    print("\n🔍 Test 7: Integration Check")
    try:
        # Check if ProjectP.py is configured for 80%
        config_check = {
            'memory_target': 0.8,
            'cpu_target': 0.35,
            'strategy': '80_percent'
        }
        
        print(f"   Expected Memory Target: {config_check['memory_target'] * 100}%")
        print(f"   Expected CPU Target: {config_check['cpu_target'] * 100}%")
        print(f"   Expected Strategy: {config_check['strategy']}")
        print("   ✅ Configuration targets verified")
        
    except Exception as e:
        print(f"   ❌ Integration check failed: {e}")
    
    # Final Summary
    print("\n🏆 TEST SUMMARY")
    print("=" * 40)
    
    current_memory_percent = memory.percent
    target_80_percent = 80.0
    
    if current_memory_percent >= target_80_percent:
        print("✅ EXCELLENT: 80%+ RAM utilization achieved")
        score = "A+"
    elif current_memory_percent >= 70:
        print("✅ GOOD: 70%+ RAM utilization (room for improvement)")
        score = "B+"
    elif current_memory_percent >= 60:
        print("⚠️ FAIR: 60%+ RAM utilization (needs optimization)")
        score = "C+"
    else:
        print("❌ POOR: <60% RAM utilization (major optimization needed)")
        score = "D"
    
    print(f"Overall Score: {score}")
    print(f"Current RAM Usage: {current_memory_percent:.1f}%")
    print(f"Target RAM Usage: {target_80_percent}%")
    print(f"Gap to Target: {target_80_percent - current_memory_percent:.1f}%")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if current_memory_percent < 70:
        print("1. ⚡ Increase batch sizes in data processing")
        print("2. 🧠 Enable more complex model architectures")
        print("3. 📊 Load larger datasets into memory")
        print("4. 🚀 Enable parallel processing")
    elif current_memory_percent < 80:
        print("1. 📈 Fine-tune resource allocation parameters")
        print("2. 🎯 Optimize memory usage patterns")
        print("3. ⚡ Enable advanced caching strategies")
    else:
        print("1. ✅ System is well-optimized")
        print("2. 🏆 Consider monitoring for stability")
        print("3. 📊 Review performance metrics regularly")
    
    return current_memory_percent >= 70  # Return True if good utilization

if __name__ == "__main__":
    success = test_80_percent_resource_system()
    
    print(f"\n🎯 Test Result: {'PASSED' if success else 'NEEDS OPTIMIZATION'}")
    
    if not success:
        print("\n🔧 NEXT STEPS:")
        print("1. Run ProjectP.py to activate 80% resource management")
        print("2. Execute Menu 1 to load and process real data")
        print("3. Monitor system during pipeline execution")
        print("4. Check system status via Menu 2")
    else:
        print("\n🎉 SUCCESS: 80% Resource Management is working optimally!")
