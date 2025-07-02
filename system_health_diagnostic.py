#!/usr/bin/env python3
"""
🔧 SYSTEM HEALTH DIAGNOSTIC AND OPTIMIZATION ANALYSIS
================================================

Comprehensive diagnostic to assess:
- Current resource usage patterns
- Error/warning counts in system health dashboard
- Memory and CPU consumption during Menu 1 operations
- Identify specific bottlenecks and optimization opportunities
"""

import os
import sys
import warnings
import psutil
import gc
import time
from pathlib import Path

# Force optimized environment
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')

def check_system_baseline():
    """Check baseline system resources before any operations"""
    print("🔍 BASELINE SYSTEM RESOURCE ASSESSMENT")
    print("=" * 60)
    
    # CPU baseline
    cpu_count = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"💻 CPU: {cpu_count} cores, {cpu_usage:.1f}% usage")
    
    # Memory baseline
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    memory_used_gb = memory.used / (1024**3)
    memory_percent = memory.percent
    print(f"🧠 Memory: {memory_used_gb:.1f}/{memory_gb:.1f} GB ({memory_percent:.1f}%)")
    
    # Process count
    process_count = len(list(psutil.process_iter()))
    print(f"⚙️ Processes: {process_count} running")
    
    return {
        'cpu_cores': cpu_count,
        'cpu_usage': cpu_usage,
        'memory_total_gb': memory_gb,
        'memory_used_gb': memory_used_gb,
        'memory_percent': memory_percent,
        'process_count': process_count
    }

def test_menu_1_resource_usage():
    """Test Menu 1 resource consumption patterns"""
    print("\n🎯 MENU 1 RESOURCE CONSUMPTION TEST")
    print("=" * 60)
    
    baseline_memory = psutil.virtual_memory().used / (1024**3)
    baseline_cpu = psutil.cpu_percent()
    
    try:
        # Test importing Menu 1
        print("📦 Testing Menu 1 import...")
        start_time = time.time()
        
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        
        import_time = time.time() - start_time
        import_memory = psutil.virtual_memory().used / (1024**3)
        import_memory_delta = import_memory - baseline_memory
        
        print(f"✅ Import time: {import_time:.2f}s")
        print(f"📈 Memory after import: +{import_memory_delta:.2f} GB")
        
        # Test Menu 1 initialization
        print("🔧 Testing Menu 1 initialization...")
        init_start = time.time()
        
        menu_1 = Menu1ElliottWave({}, None, None)
        
        init_time = time.time() - init_start
        init_memory = psutil.virtual_memory().used / (1024**3)
        init_memory_delta = init_memory - import_memory
        
        print(f"✅ Init time: {init_time:.2f}s")
        print(f"📈 Memory after init: +{init_memory_delta:.2f} GB")
        
        total_memory_delta = init_memory - baseline_memory
        print(f"🎯 Total Menu 1 memory impact: +{total_memory_delta:.2f} GB")
        
        return {
            'import_time': import_time,
            'init_time': init_time,
            'memory_impact_gb': total_memory_delta,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Menu 1 test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def check_logging_system_overhead():
    """Check logging system resource overhead"""
    print("\n📊 LOGGING SYSTEM OVERHEAD ANALYSIS")
    print("=" * 60)
    
    baseline_memory = psutil.virtual_memory().used / (1024**3)
    
    try:
        # Test advanced logging import
        print("📦 Testing advanced logging import...")
        start_time = time.time()
        
        from core.advanced_terminal_logger import get_terminal_logger
        from core.real_time_progress_manager import get_progress_manager
        
        import_time = time.time() - start_time
        import_memory = psutil.virtual_memory().used / (1024**3)
        import_memory_delta = import_memory - baseline_memory
        
        print(f"✅ Logging import time: {import_time:.2f}s")
        print(f"📈 Memory impact: +{import_memory_delta:.2f} GB")
        
        # Test logger initialization
        logger = get_terminal_logger()
        progress_manager = get_progress_manager()
        
        init_memory = psutil.virtual_memory().used / (1024**3)
        total_memory_delta = init_memory - baseline_memory
        
        print(f"🎯 Total logging overhead: +{total_memory_delta:.2f} GB")
        
        return {
            'import_time': import_time,
            'memory_overhead_gb': total_memory_delta,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Logging system test failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def identify_optimization_opportunities():
    """Identify specific optimization opportunities"""
    print("\n🎯 OPTIMIZATION OPPORTUNITY ANALYSIS")
    print("=" * 60)
    
    opportunities = []
    
    # Check current memory usage
    memory = psutil.virtual_memory()
    if memory.percent > 70:
        opportunities.append("HIGH_MEMORY_USAGE: Current memory usage is over 70%")
    
    # Check CPU usage
    cpu_usage = psutil.cpu_percent(interval=1)
    if cpu_usage > 50:
        opportunities.append("HIGH_CPU_USAGE: Current CPU usage is over 50%")
    
    # Check for large Python processes
    python_memory = 0
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if 'python' in proc.info['name'].lower():
                python_memory += proc.info['memory_info'].rss
        except:
            continue
    
    python_memory_gb = python_memory / (1024**3)
    if python_memory_gb > 1.0:
        opportunities.append(f"PYTHON_MEMORY_USAGE: Python processes using {python_memory_gb:.2f} GB")
    
    # Check disk space
    disk_usage = psutil.disk_usage('/')
    disk_percent = (disk_usage.used / disk_usage.total) * 100
    if disk_percent > 80:
        opportunities.append(f"DISK_SPACE: Disk usage at {disk_percent:.1f}%")
    
    print(f"🔍 Found {len(opportunities)} optimization opportunities:")
    for i, opp in enumerate(opportunities, 1):
        print(f"   {i}. {opp}")
    
    return opportunities

def main():
    """Main diagnostic function"""
    print("🏥 NICEGOLD SYSTEM HEALTH DIAGNOSTIC")
    print("=" * 60)
    print(f"📅 Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️ System: {os.uname().sysname} {os.uname().release}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print()
    
    # Run all diagnostic tests
    baseline = check_system_baseline()
    menu_1_test = test_menu_1_resource_usage()
    logging_test = check_logging_system_overhead()
    opportunities = identify_optimization_opportunities()
    
    # Generate optimization recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    if menu_1_test.get('memory_impact_gb', 0) > 0.5:
        print("🎯 MENU 1 OPTIMIZATION: High memory usage detected")
        print("   → Recommend: Use optimized_menu_1_elliott_wave.py")
        print("   → Implement: Lazy loading, reduced batch sizes")
    
    if logging_test.get('memory_overhead_gb', 0) > 0.2:
        print("📊 LOGGING OPTIMIZATION: Logging overhead detected")
        print("   → Recommend: Reduce logging verbosity")
        print("   → Implement: Conditional advanced logging")
    
    if baseline['memory_percent'] > 70:
        print("🧠 MEMORY OPTIMIZATION: High baseline memory usage")
        print("   → Recommend: Use optimized_resource_manager.py")
        print("   → Implement: Conservative resource allocation")
    
    print("\n✅ DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("🔧 Ready for optimization implementation!")

if __name__ == "__main__":
    main()
