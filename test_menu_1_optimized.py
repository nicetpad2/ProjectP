#!/usr/bin/env python3
"""
🧪 MENU 1 OPTIMIZED TESTING SCRIPT
Test Menu 1 (Full Pipeline) with optimized components
"""

import os
import sys
import time
import psutil
from pathlib import Path

# Add project to path
sys.path.insert(0, '/mnt/data/projects/ProjectP')

# Set optimized environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONIOENCODING'] = 'utf-8'

def test_menu_1_optimized():
    """Test optimized Menu 1 performance"""
    print("🧪 TESTING MENU 1 OPTIMIZED PERFORMANCE")
    print("=" * 60)
    
    # Baseline measurements
    start_memory = psutil.virtual_memory().used / (1024**3)
    start_time = time.time()
    
    print(f"📊 Baseline Memory: {start_memory:.2f} GB")
    print()
    
    try:
        # Test optimized Menu 1 import
        print("📦 Testing optimized Menu 1 import...")
        
        from menu_modules.optimized_menu_1_elliott_wave import OptimizedMenu1ElliottWave
        
        import_time = time.time() - start_time
        import_memory = psutil.virtual_memory().used / (1024**3)
        import_delta = import_memory - start_memory
        
        print(f"✅ Import time: {import_time:.2f}s")
        print(f"📈 Memory delta: +{import_delta:.3f} GB")
        
        # Test Menu 1 initialization
        print("\n🔧 Testing optimized Menu 1 initialization...")
        init_start = time.time()
        
        # Initialize with minimal config
        config = {'optimized_mode': True, 'conservative_allocation': True}
        menu_1 = OptimizedMenu1ElliottWave(config, None, None)
        
        init_time = time.time() - init_start
        init_memory = psutil.virtual_memory().used / (1024**3)
        total_delta = init_memory - start_memory
        
        print(f"✅ Init time: {init_time:.2f}s")
        print(f"📈 Total memory delta: +{total_delta:.3f} GB")
        
        # Performance summary
        print("\n🎯 PERFORMANCE SUMMARY")
        print("=" * 40)
        print(f"📦 Import time: {import_time:.2f}s (target: <3s)")
        print(f"🔧 Init time: {init_time:.2f}s (target: <2s)")
        print(f"💾 Memory usage: {total_delta*1000:.0f}MB (target: <200MB)")
        
        # Performance scoring
        import_score = "✅" if import_time < 3.0 else "⚠️"
        init_score = "✅" if init_time < 2.0 else "⚠️"
        memory_score = "✅" if total_delta < 0.2 else "⚠️"
        
        print(f"\n📊 PERFORMANCE SCORES")
        print(f"   {import_score} Import Speed: {'PASS' if import_time < 3.0 else 'NEEDS IMPROVEMENT'}")
        print(f"   {init_score} Init Speed: {'PASS' if init_time < 2.0 else 'NEEDS IMPROVEMENT'}")
        print(f"   {memory_score} Memory Usage: {'PASS' if total_delta < 0.2 else 'NEEDS IMPROVEMENT'}")
        
        # Test basic functionality
        print("\n🧪 Testing basic functionality...")
        try:
            # Test that menu has required methods
            assert hasattr(menu_1, 'run'), "Menu missing 'run' method"
            assert hasattr(menu_1, 'config'), "Menu missing 'config' attribute"
            print("✅ Basic functionality test: PASS")
            
            return {
                'success': True,
                'import_time': import_time,
                'init_time': init_time,
                'memory_usage_gb': total_delta,
                'performance_summary': {
                    'import_ok': import_time < 3.0,
                    'init_ok': init_time < 2.0,
                    'memory_ok': total_delta < 0.2
                }
            }
            
        except Exception as func_e:
            print(f"❌ Functionality test failed: {func_e}")
            return {'success': False, 'error': str(func_e)}
        
    except Exception as e:
        print(f"❌ Menu 1 test failed: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main test function"""
    print("🚀 MENU 1 OPTIMIZED TESTING")
    print("=" * 60)
    print(f"📅 Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️ System: {os.uname().sysname}")
    print()
    
    # Run test
    result = test_menu_1_optimized()
    
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if result['success']:
        print("✅ MENU 1 OPTIMIZED: TEST PASSED")
        
        perf = result['performance_summary']
        overall_score = sum([perf['import_ok'], perf['init_ok'], perf['memory_ok']])
        
        print(f"📊 Overall Performance: {overall_score}/3 tests passed")
        
        if overall_score == 3:
            print("🏆 EXCELLENT: All performance targets met!")
        elif overall_score == 2:
            print("🎯 GOOD: Most performance targets met")
        else:
            print("⚠️ NEEDS IMPROVEMENT: Performance optimization required")
            
        print(f"⏱️ Total time: {result['import_time'] + result['init_time']:.2f}s")
        print(f"💾 Memory footprint: {result['memory_usage_gb']*1000:.0f}MB")
        
    else:
        print("❌ MENU 1 OPTIMIZED: TEST FAILED")
        print(f"💥 Error: {result['error']}")
    
    print("\n🎯 Menu 1 optimization testing complete!")

if __name__ == "__main__":
    main()
