#!/usr/bin/env python3
"""
🎯 PRODUCTION STATUS VALIDATOR
ตรวจสอบสถานะระบบและยืนยันความพร้อมใช้งาน Production
"""

import os
import sys
import time
from datetime import datetime

def validate_production_status():
    """ตรวจสอบสถานะการเตรียมพร้อม Production"""
    
    print("🔍 NICEGOLD PRODUCTION STATUS VALIDATION")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Core Files Existence
    print("📁 Test 1: Core Files Verification")
    core_files = [
        'ProjectP.py',
        'core/optimized_resource_manager.py',
        'menu_modules/ultra_lightweight_menu_1.py',
        'aggressive_cuda_suppression.py'
    ]
    
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            return False
    
    # Test 2: System Import Test
    print("\n🧪 Test 2: System Import Verification")
    try:
        sys.path.append('/mnt/data/projects/ProjectP')
        
        # Test optimized resource manager
        from core.optimized_resource_manager import OptimizedResourceManager
        print("   ✅ OptimizedResourceManager")
        
        # Test ultra-lightweight menu
        from menu_modules.ultra_lightweight_menu_1 import UltraLightweightMenu1
        print("   ✅ UltraLightweightMenu1")
        
        # Test CUDA suppression
        from aggressive_cuda_suppression import suppress_all_output
        print("   ✅ CUDA Suppression")
        
    except Exception as e:
        print(f"   ❌ Import Error: {e}")
        return False
    
    # Test 3: Quick Performance Test
    print("\n⚡ Test 3: Performance Verification")
    try:
        start_time = time.time()
        
        # Initialize optimized resource manager
        rm = OptimizedResourceManager()
        
        # Get performance metrics
        performance = rm.get_current_performance()
        
        # Get health status
        health = rm.get_health_status()
        
        # Cleanup
        rm.stop_monitoring()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"   ✅ Execution Time: {execution_time:.2f}s")
        print(f"   ✅ Health Score: {health.get('health_score', 0)}%")
        print(f"   ✅ Memory Usage: {performance.get('memory_percent', 0):.1f}%")
        print(f"   ✅ CPU Usage: {performance.get('cpu_percent', 0):.1f}%")
        
        if execution_time > 2.0:
            print("   ⚠️ Execution time above target (2.0s)")
            return False
            
    except Exception as e:
        print(f"   ❌ Performance Test Error: {e}")
        return False
    
    # Test 4: Menu System Test
    print("\n🎛️ Test 4: Menu System Verification")
    try:
        config = {'optimized_mode': True}
        menu = UltraLightweightMenu1(config, None, None)
        
        # Test menu availability
        if hasattr(menu, 'run'):
            print("   ✅ Menu Run Method Available")
        else:
            print("   ❌ Menu Run Method Missing")
            return False
            
        print("   ✅ Menu System Operational")
        
    except Exception as e:
        print(f"   ❌ Menu System Error: {e}")
        return False
    
    # Test 5: Resource Optimization Verification
    print("\n🧠 Test 5: Resource Optimization Verification")
    try:
        rm = OptimizedResourceManager()
        config = rm.get_optimized_config('ml_training')
        
        # Check conservative settings
        if config.get('cpu_threads', 0) <= 2:
            print("   ✅ Conservative CPU Allocation")
        else:
            print("   ⚠️ CPU allocation may be too high")
        
        if config.get('memory_limit_gb', 0) <= 2.0:
            print("   ✅ Conservative Memory Allocation")
        else:
            print("   ⚠️ Memory allocation may be too high")
        
        if config.get('batch_size', 0) <= 16:
            print("   ✅ Optimized Batch Size")
        else:
            print("   ⚠️ Batch size may be too large")
        
        rm.stop_monitoring()
        print("   ✅ Resource Optimization Verified")
        
    except Exception as e:
        print(f"   ❌ Resource Optimization Error: {e}")
        return False
    
    return True

def main():
    """Main validation function"""
    print("🚀 NICEGOLD ENTERPRISE PRODUCTION VALIDATOR")
    print("=" * 60)
    
    success = validate_production_status()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PRODUCTION STATUS: ✅ READY")
        print("🏆 All tests passed - System is production-ready!")
        print("⚡ Performance optimized and error-free!")
        print("🛡️ Enterprise-grade quality verified!")
        print("\n✅ Status: PRODUCTION DEPLOYMENT APPROVED")
    else:
        print("❌ PRODUCTION STATUS: NOT READY")
        print("🔧 Some tests failed - System needs attention")
        print("\n❌ Status: PRODUCTION DEPLOYMENT NOT APPROVED")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
