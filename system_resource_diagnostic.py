#!/usr/bin/env python3
"""
🔍 NICEGOLD ENTERPRISE - COMPLETE SYSTEM RESOURCE DIAGNOSTIC
ระบบตรวจสอบและวิเคราะห์ทรัพยากรระบบแบบสมบูรณ์

Features:
🧠 Memory Analysis (80% target)
⚡ CPU Analysis (cores and usage)
🖥️ GPU Detection and Analysis  
📊 Resource Optimization Recommendations
🛡️ Enterprise-grade Monitoring
"""

import os
import sys
import platform
import subprocess
import psutil
import time
import gc
from datetime import datetime
from typing import Dict, Any, List, Optional

def check_gpu_availability() -> Dict[str, Any]:
    """Comprehensive GPU detection and analysis"""
    gpu_info = {
        'available': False,
        'type': 'none',
        'count': 0,
        'details': {},
        'cuda_available': False,
        'tensorflow_gpu': False,
        'pytorch_gpu': False
    }
    
    print("🖥️ GPU DETECTION AND ANALYSIS")
    print("=" * 50)
    
    # Check NVIDIA GPU with nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu', '--format=csv,nounits,noheader'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split('\n')
            gpu_info['available'] = True
            gpu_info['type'] = 'NVIDIA'
            gpu_info['count'] = len(gpu_lines)
            
            for i, line in enumerate(gpu_lines):
                parts = line.split(', ')
                if len(parts) >= 6:
                    gpu_info['details'][f'gpu_{i}'] = {
                        'name': parts[0].strip(),
                        'memory_total_mb': int(parts[1]),
                        'memory_used_mb': int(parts[2]),
                        'memory_free_mb': int(parts[3]),
                        'temperature_c': int(parts[4]) if parts[4] != '[Not Supported]' else 'N/A',
                        'utilization_percent': int(parts[5]) if parts[5] != '[Not Supported]' else 'N/A'
                    }
                    
            print(f"✅ NVIDIA GPU detected: {gpu_info['count']} device(s)")
            for i, details in gpu_info['details'].items():
                print(f"   📋 {details['name']}: {details['memory_total_mb']}MB total, {details['memory_free_mb']}MB free")
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        print("❌ NVIDIA GPU not detected (nvidia-smi not available)")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info['cuda_available'] = True
            gpu_info['pytorch_gpu'] = True
            cuda_count = torch.cuda.device_count()
            print(f"✅ CUDA available with PyTorch: {cuda_count} device(s)")
            
            for i in range(cuda_count):
                props = torch.cuda.get_device_properties(i)
                print(f"   📋 Device {i}: {props.name} ({props.total_memory//1024//1024}MB)")
        else:
            print("❌ CUDA not available with PyTorch")
    except ImportError:
        print("⚠️ PyTorch not installed - cannot check CUDA")
    
    # Check TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info['tensorflow_gpu'] = True
            print(f"✅ TensorFlow GPU support: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   📋 TF Device {i}: {gpu.name}")
        else:
            print("❌ TensorFlow GPU support not available")
    except ImportError:
        print("⚠️ TensorFlow not installed - cannot check GPU support")
    
    # Check AMD GPU (basic)
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
        if 'AMD' in result.stdout and ('Radeon' in result.stdout or 'GPU' in result.stdout):
            print("✅ AMD GPU detected (basic detection)")
            gpu_info['type'] = 'AMD'
            gpu_info['available'] = True
    except:
        pass
    
    if not gpu_info['available']:
        print("❌ No GPU detected - using CPU-only mode")
        print("💡 This is normal for CPU-only environments like Google Colab CPU runtime")
    
    return gpu_info

def analyze_memory_usage() -> Dict[str, Any]:
    """Comprehensive memory analysis for 80% target"""
    print("\n🧠 MEMORY ANALYSIS (80% TARGET)")
    print("=" * 50)
    
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    # Calculate 80% targets
    target_80_percent = memory.total * 0.8
    safe_allocation = memory.available * 0.8
    
    memory_info = {
        'total_gb': round(memory.total / (1024**3), 2),
        'available_gb': round(memory.available / (1024**3), 2),
        'used_gb': round(memory.used / (1024**3), 2),
        'used_percent': memory.percent,
        'target_80_percent_gb': round(target_80_percent / (1024**3), 2),
        'safe_allocation_gb': round(safe_allocation / (1024**3), 2),
        'can_allocate_80_percent': memory.available > target_80_percent * 0.5,
        'recommendation': {}
    }
    
    print(f"📊 Total Memory: {memory_info['total_gb']} GB")
    print(f"📊 Available Memory: {memory_info['available_gb']} GB")
    print(f"📊 Used Memory: {memory_info['used_gb']} GB ({memory_info['used_percent']:.1f}%)")
    print(f"🎯 80% Target: {memory_info['target_80_percent_gb']} GB")
    print(f"✅ Safe 80% Allocation: {memory_info['safe_allocation_gb']} GB")
    
    # Memory recommendations
    if memory_info['available_gb'] >= 8:
        memory_info['recommendation'] = {
            'status': 'excellent',
            'batch_size': 512,
            'workers': 8,
            'cache_size': 10000,
            'message': 'Excellent memory - can use full 80% allocation'
        }
        print("🏆 EXCELLENT: Sufficient memory for 80% allocation")
    elif memory_info['available_gb'] >= 4:
        memory_info['recommendation'] = {
            'status': 'good',
            'batch_size': 256,
            'workers': 4,
            'cache_size': 5000,
            'message': 'Good memory - can use modified 80% allocation'
        }
        print("✅ GOOD: Adequate memory with optimized 80% allocation")
    else:
        memory_info['recommendation'] = {
            'status': 'limited',
            'batch_size': 128,
            'workers': 2,
            'cache_size': 1000,
            'message': 'Limited memory - conservative allocation recommended'
        }
        print("⚠️ LIMITED: Conservative allocation recommended")
    
    # Swap analysis
    if swap.total > 0:
        print(f"💿 Swap: {round(swap.total/(1024**3), 2)} GB total, {round(swap.used/(1024**3), 2)} GB used ({swap.percent:.1f}%)")
    else:
        print("💿 Swap: Not configured")
    
    return memory_info

def analyze_cpu_usage() -> Dict[str, Any]:
    """Comprehensive CPU analysis"""
    print("\n⚡ CPU ANALYSIS")
    print("=" * 50)
    
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_freq = psutil.cpu_freq()
    
    # Get CPU usage over 3 seconds for accuracy
    print("📊 Measuring CPU usage (3 second sample)...")
    cpu_percent = psutil.cpu_percent(interval=3, percpu=True)
    avg_cpu_percent = sum(cpu_percent) / len(cpu_percent)
    
    cpu_info = {
        'logical_cores': cpu_count_logical,
        'physical_cores': cpu_count_physical,
        'current_usage_percent': round(avg_cpu_percent, 1),
        'per_core_usage': [round(x, 1) for x in cpu_percent],
        'frequency_mhz': round(cpu_freq.current, 0) if cpu_freq else 'Unknown',
        'max_frequency_mhz': round(cpu_freq.max, 0) if cpu_freq else 'Unknown',
        'target_80_percent_cores': max(1, int(cpu_count_logical * 0.8)),
        'recommended_workers': max(1, int(cpu_count_logical * 0.8)),
        'optimization': {}
    }
    
    print(f"🔧 Logical Cores: {cpu_count_logical}")
    print(f"🔧 Physical Cores: {cpu_count_physical}")
    print(f"⚡ Current Usage: {cpu_info['current_usage_percent']}%")
    print(f"📈 Frequency: {cpu_info['frequency_mhz']} MHz (max: {cpu_info['max_frequency_mhz']} MHz)")
    print(f"🎯 80% Target Workers: {cpu_info['target_80_percent_cores']}")
    
    # CPU optimization recommendations
    if avg_cpu_percent < 30:
        cpu_info['optimization'] = {
            'status': 'underutilized',
            'can_increase_load': True,
            'recommended_action': 'Can safely increase to 80% usage',
            'parallel_processing': True
        }
        print("📈 CPU UNDERUTILIZED: Can safely increase workload to 80%")
    elif avg_cpu_percent < 70:
        cpu_info['optimization'] = {
            'status': 'optimal',
            'can_increase_load': True,
            'recommended_action': 'Good for 80% target usage',
            'parallel_processing': True
        }
        print("✅ CPU OPTIMAL: Good for 80% target usage")
    else:
        cpu_info['optimization'] = {
            'status': 'high_usage',
            'can_increase_load': False,
            'recommended_action': 'Monitor and optimize current load',
            'parallel_processing': False
        }
        print("⚠️ CPU HIGH USAGE: Monitor current load before increasing")
    
    return cpu_info

def get_system_information() -> Dict[str, Any]:
    """Get comprehensive system information"""
    print("\n💻 SYSTEM INFORMATION")
    print("=" * 50)
    
    system_info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'environment': 'Unknown'
    }
    
    # Detect environment
    if 'google.colab' in sys.modules:
        system_info['environment'] = 'Google Colab'
    elif 'KAGGLE_CONTAINER_NAME' in os.environ:
        system_info['environment'] = 'Kaggle'
    elif 'JUPYTER_SERVER_ROOT' in os.environ:
        system_info['environment'] = 'Jupyter'
    elif os.path.exists('/.dockerenv'):
        system_info['environment'] = 'Docker'
    else:
        system_info['environment'] = 'Local/Server'
    
    print(f"🖥️ Platform: {system_info['platform']} {system_info['platform_release']}")
    print(f"🏗️ Architecture: {system_info['architecture']}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🌐 Environment: {system_info['environment']}")
    
    return system_info

def test_resource_allocation() -> Dict[str, Any]:
    """Test actual resource allocation capabilities"""
    print("\n🧪 RESOURCE ALLOCATION TEST")
    print("=" * 50)
    
    test_results = {
        'memory_test': {},
        'cpu_test': {},
        'allocation_feasible': False
    }
    
    # Memory allocation test
    print("🧠 Testing memory allocation...")
    try:
        import numpy as np  # Import numpy here
        initial_memory = psutil.virtual_memory()
        
        # Test allocating arrays (conservative test)
        test_arrays = []
        allocated_mb = 0
        target_mb = min(1000, int(initial_memory.available * 0.1 / (1024*1024)))  # Test with 10% of available
        
        for i in range(min(10, target_mb // 100)):
            test_array = np.zeros((100 * 1024 * 1024 // 8,), dtype=np.float64)  # 100MB array
            test_arrays.append(test_array)
            allocated_mb += 100
            
        # Clean up
        del test_arrays
        gc.collect()
        
        test_results['memory_test'] = {
            'allocated_mb': allocated_mb,
            'success': True,
            'message': f'Successfully allocated {allocated_mb}MB'
        }
        print(f"✅ Memory test passed: {allocated_mb}MB allocated and freed")
        
    except Exception as e:
        test_results['memory_test'] = {
            'success': False,
            'error': str(e),
            'message': 'Memory allocation test failed'
        }
        print(f"❌ Memory test failed: {e}")
    
    # CPU utilization test
    print("⚡ Testing CPU utilization...")
    try:
        import threading
        import time
        
        def cpu_worker():
            """Simple CPU-intensive task"""
            end_time = time.time() + 2  # Run for 2 seconds
            while time.time() < end_time:
                _ = sum(i*i for i in range(1000))
        
        # Start worker threads
        threads = []
        cpu_count = psutil.cpu_count(logical=True)
        worker_count = max(1, int(cpu_count * 0.5))  # Use 50% for test
        
        for _ in range(worker_count):
            thread = threading.Thread(target=cpu_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        test_results['cpu_test'] = {
            'workers_started': worker_count,
            'success': True,
            'message': f'Successfully ran {worker_count} CPU workers'
        }
        print(f"✅ CPU test passed: {worker_count} workers completed")
        
    except Exception as e:
        test_results['cpu_test'] = {
            'success': False,
            'error': str(e),
            'message': 'CPU utilization test failed'
        }
        print(f"❌ CPU test failed: {e}")
    
    # Overall feasibility
    test_results['allocation_feasible'] = (
        test_results['memory_test'].get('success', False) and 
        test_results['cpu_test'].get('success', False)
    )
    
    if test_results['allocation_feasible']:
        print("🏆 RESOURCE ALLOCATION FEASIBLE: System ready for 80% utilization")
    else:
        print("⚠️ RESOURCE ALLOCATION CONCERNS: Review system capacity")
    
    return test_results

def generate_optimization_report() -> Dict[str, Any]:
    """Generate comprehensive optimization recommendations"""
    print("\n📋 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 65)
    
    # Get all analysis results
    gpu_info = check_gpu_availability()
    memory_info = analyze_memory_usage()
    cpu_info = analyze_cpu_usage()
    system_info = get_system_information()
    test_results = test_resource_allocation()
    
    # Generate recommendations
    recommendations = {
        'system_status': 'unknown',
        'memory_optimization': {},
        'cpu_optimization': {},
        'gpu_optimization': {},
        'nicegold_config': {},
        'priority_actions': []
    }
    
    # System status assessment
    if (memory_info['available_gb'] >= 4 and 
        cpu_info['logical_cores'] >= 2 and 
        test_results['allocation_feasible']):
        recommendations['system_status'] = 'excellent'
        print("🏆 SYSTEM STATUS: EXCELLENT - Ready for 80% resource utilization")
    elif (memory_info['available_gb'] >= 2 and 
          cpu_info['logical_cores'] >= 2):
        recommendations['system_status'] = 'good'
        print("✅ SYSTEM STATUS: GOOD - Ready with optimizations")
    else:
        recommendations['system_status'] = 'limited'
        print("⚠️ SYSTEM STATUS: LIMITED - Conservative settings recommended")
    
    # Memory optimization
    recommendations['memory_optimization'] = {
        'target_allocation_gb': min(memory_info['safe_allocation_gb'], memory_info['target_80_percent_gb']),
        'batch_size': memory_info['recommendation']['batch_size'],
        'cache_size': memory_info['recommendation']['cache_size'],
        'garbage_collection': 'aggressive' if memory_info['available_gb'] < 4 else 'normal'
    }
    
    # CPU optimization
    recommendations['cpu_optimization'] = {
        'workers': cpu_info['recommended_workers'],
        'parallel_processing': cpu_info['optimization']['parallel_processing'],
        'cpu_intensive_tasks': cpu_info['optimization']['can_increase_load']
    }
    
    # GPU optimization
    if gpu_info['available']:
        recommendations['gpu_optimization'] = {
            'use_gpu': True,
            'gpu_type': gpu_info['type'],
            'cuda_available': gpu_info['cuda_available'],
            'recommendation': 'Use GPU for deep learning tasks'
        }
    else:
        recommendations['gpu_optimization'] = {
            'use_gpu': False,
            'cpu_fallback': True,
            'recommendation': 'Optimize CPU-only operations'
        }
    
    # NICEGOLD-specific configuration
    recommendations['nicegold_config'] = {
        'resource_manager_config': {
            'memory_target': 0.8,
            'cpu_target': 0.8 if cpu_info['optimization']['can_increase_load'] else 0.6,
            'batch_size': recommendations['memory_optimization']['batch_size'],
            'workers': recommendations['cpu_optimization']['workers'],
            'use_gpu': recommendations['gpu_optimization']['use_gpu']
        },
        'elliott_wave_config': {
            'feature_cache_size': recommendations['memory_optimization']['cache_size'],
            'parallel_feature_engineering': recommendations['cpu_optimization']['parallel_processing'],
            'memory_conservative_mode': memory_info['available_gb'] < 4
        }
    }
    
    print(f"\n🎯 RECOMMENDED NICEGOLD CONFIGURATION:")
    print(f"   💾 Memory Target: {recommendations['nicegold_config']['resource_manager_config']['memory_target']*100}%")
    print(f"   ⚡ CPU Target: {recommendations['nicegold_config']['resource_manager_config']['cpu_target']*100}%")
    print(f"   📦 Batch Size: {recommendations['nicegold_config']['resource_manager_config']['batch_size']}")
    print(f"   👥 Workers: {recommendations['nicegold_config']['resource_manager_config']['workers']}")
    print(f"   🖥️ GPU Usage: {recommendations['nicegold_config']['resource_manager_config']['use_gpu']}")
    
    return {
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'gpu_info': gpu_info,
        'memory_info': memory_info,
        'cpu_info': cpu_info,
        'test_results': test_results,
        'recommendations': recommendations
    }

def main():
    """Main diagnostic function"""
    print("🔍 NICEGOLD ENTERPRISE - COMPLETE SYSTEM RESOURCE DIAGNOSTIC")
    print("=" * 65)
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Purpose: Comprehensive analysis for 80% resource utilization")
    
    try:
        # Import required libraries
        import numpy as np
        
        # Run complete diagnostic
        diagnostic_report = generate_optimization_report()
        
        # Save report
        report_filename = f"/content/drive/MyDrive/ProjectP-1/system_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            import json
            with open(report_filename, 'w') as f:
                json.dump(diagnostic_report, f, indent=2, default=str)
            print(f"\n📁 Diagnostic report saved: {report_filename}")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        # Final summary
        print(f"\n🏁 DIAGNOSTIC COMPLETE")
        print("=" * 65)
        status = diagnostic_report['recommendations']['system_status']
        if status == 'excellent':
            print("🎉 SYSTEM READY: Full 80% resource utilization supported!")
        elif status == 'good':
            print("👍 SYSTEM CAPABLE: 80% utilization possible with optimizations")
        else:
            print("🔧 SYSTEM NEEDS OPTIMIZATION: Conservative settings recommended")
        
    except ImportError as e:
        print(f"❌ Required library missing: {e}")
        print("🔧 Please install missing dependencies")
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")

if __name__ == "__main__":
    main()
