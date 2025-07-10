#!/usr/bin/env python3
"""
🎮 NICEGOLD ENTERPRISE GPU RESOURCE MANAGER
Advanced GPU Detection & Resource Management System
Author: NICEGOLD Enterprise AI
Date: July 6, 2025
Version: 2.0 DIVINE EDITION

🎯 FEATURES:
✅ Advanced GPU Hardware Detection
✅ 80% Resource Utilization Optimization  
✅ CUDA Capability Assessment
✅ Dynamic Configuration Management
✅ Cross-Platform Compatibility
✅ Enterprise-Grade Performance Monitoring
"""

import os
import sys
import platform
import subprocess
import psutil
import warnings
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EnterpriseGPUManager:
    """
    🎯 ULTIMATE ENTERPRISE GPU RESOURCE MANAGER
    
    Advanced Features:
    ✅ Multi-GPU Detection & Management
    ✅ 80% Resource Utilization Strategy
    ✅ CUDA Capability Assessment  
    ✅ Dynamic Performance Optimization
    ✅ Real-time Resource Monitoring
    ✅ Cross-Platform Support
    """
    
    def __init__(self):
        self.system_info = self._detect_system_specs()
        self.gpu_hardware = self._detect_gpu_hardware()
        self.optimal_config = self._calculate_optimal_config()
        self.performance_history = []
        
        # Configure environment
        self._configure_environment()
        
    def _detect_system_specs(self) -> Dict[str, Any]:
        """🖥️ Comprehensive System Specification Detection"""
        specs = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'swap_total': psutil.swap_memory().total,
            'disk_usage': psutil.disk_usage('/').total if os.path.exists('/') else 0,
            'boot_time': psutil.boot_time(),
            'python_version': platform.python_version()
        }
        return specs
    
    def _detect_gpu_hardware(self) -> Dict[str, Any]:
        """🎮 Advanced Multi-GPU Hardware Detection"""
        gpu_info = {
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_devices': [],
            'total_gpu_memory': 0,
            'cuda_available': False,
            'cuda_version': None,
            'driver_version': None,
            'compute_capabilities': [],
            'detection_method': 'nvidia-smi'
        }
        
        # Method 1: NVIDIA-SMI Detection
        try:
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,power.draw,utilization.gpu',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 8:
                            device = {
                                'index': int(parts[0]),
                                'name': parts[1],
                                'memory_total': int(parts[2]),
                                'memory_free': int(parts[3]),
                                'memory_used': int(parts[4]),
                                'temperature': float(parts[5]) if parts[5] != 'N/A' else None,
                                'power_draw': float(parts[6]) if parts[6] != 'N/A' else None,
                                'utilization': float(parts[7]) if parts[7] != 'N/A' else None
                            }
                            gpu_info['gpu_devices'].append(device)
                
                gpu_info['gpu_count'] = len(gpu_info['gpu_devices'])
                gpu_info['gpu_available'] = gpu_info['gpu_count'] > 0
                gpu_info['total_gpu_memory'] = sum(d['memory_total'] for d in gpu_info['gpu_devices'])
                
                # Get driver version
                driver_result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                             capture_output=True, text=True, timeout=10)
                if driver_result.returncode == 0:
                    gpu_info['driver_version'] = driver_result.stdout.strip().split('\n')[0]
                    
        except Exception as e:
            print(f"⚠️ NVIDIA-SMI detection failed: {e}")
        
        # Method 2: TensorFlow GPU Detection
        try:
            import tensorflow as tf
            
            physical_gpus = tf.config.list_physical_devices('GPU')
            if physical_gpus:
                gpu_info['cuda_available'] = True
                gpu_info['detection_method'] = 'tensorflow'
                
                for i, gpu in enumerate(physical_gpus):
                    try:
                        details = tf.config.experimental.get_device_details(gpu)
                        if 'compute_capability' in details:
                            gpu_info['compute_capabilities'].append(details['compute_capability'])
                    except:
                        pass
                        
                # Get CUDA version
                if hasattr(tf, 'sysconfig'):
                    gpu_info['cuda_version'] = tf.sysconfig.get_build_info().get('cuda_version', 'Unknown')
                    
        except Exception as e:
            print(f"⚠️ TensorFlow GPU detection failed: {e}")
        
        # Method 3: PyTorch GPU Detection  
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                if not gpu_info['gpu_available']:  # Fallback if nvidia-smi failed
                    gpu_info['gpu_count'] = torch.cuda.device_count()
                    gpu_info['gpu_available'] = gpu_info['gpu_count'] > 0
                    
                    for i in range(gpu_info['gpu_count']):
                        props = torch.cuda.get_device_properties(i)
                        device = {
                            'index': i,
                            'name': props.name,
                            'memory_total': props.total_memory // 1024 // 1024,  # Convert to MB
                            'compute_capability': f"{props.major}.{props.minor}"
                        }
                        if not any(d['index'] == i for d in gpu_info['gpu_devices']):
                            gpu_info['gpu_devices'].append(device)
                            
        except Exception as e:
            print(f"⚠️ PyTorch GPU detection failed: {e}")
            
        return gpu_info
    
    def _calculate_optimal_config(self) -> Dict[str, Any]:
        """⚙️ Calculate 80% Optimal Resource Configuration"""
        config = {
            # GPU Configuration
            'use_gpu': False,
            'gpu_memory_fraction': 0.8,
            'selected_gpu': None,
            'gpu_memory_limit': None,
            
            # CPU Configuration  
            'cpu_threads': max(1, int(self.system_info['cpu_count_logical'] * 0.8)),
            'cpu_parallel_jobs': max(1, int(self.system_info['cpu_count_physical'] * 0.8)),
            
            # Memory Configuration
            'memory_limit_gb': int(self.system_info['memory_total'] / (1024**3) * 0.8),
            'swap_usage_limit': 0.5,
            
            # Processing Configuration
            'batch_size': 1024,
            'optuna_trials': 500,
            'shap_samples': 1000,
            'processing_mode': 'CPU_OPTIMIZED',
            
            # Performance Tuning
            'thread_affinity': True,
            'memory_preallocation': True,
            'garbage_collection': True
        }
        
        if self.gpu_hardware['gpu_available'] and self.gpu_hardware['gpu_count'] > 0:
            # Select best GPU
            best_gpu = max(self.gpu_hardware['gpu_devices'], 
                          key=lambda x: x.get('memory_total', 0))
            
            gpu_memory = best_gpu['memory_total']
            gpu_utilization = best_gpu.get('utilization', 0)
            
            # GPU Configuration based on memory and current utilization
            if gpu_memory >= 8000 and gpu_utilization < 50:  # 8GB+ and low utilization
                config.update({
                    'use_gpu': True,
                    'selected_gpu': best_gpu['index'],
                    'gpu_memory_fraction': 0.8,
                    'gpu_memory_limit': int(gpu_memory * 0.8),
                    'batch_size': min(8192, int(gpu_memory * 0.2)),
                    'optuna_trials': 1000,
                    'shap_samples': 3000,
                    'processing_mode': 'GPU_HIGH_PERFORMANCE'
                })
            elif gpu_memory >= 4000 and gpu_utilization < 70:  # 4GB+ and moderate utilization
                config.update({
                    'use_gpu': True,
                    'selected_gpu': best_gpu['index'],
                    'gpu_memory_fraction': 0.7,
                    'gpu_memory_limit': int(gpu_memory * 0.7),
                    'batch_size': min(4096, int(gpu_memory * 0.15)),
                    'optuna_trials': 750,
                    'shap_samples': 2000,
                    'processing_mode': 'GPU_BALANCED'
                })
            elif gpu_memory >= 2000 and gpu_utilization < 80:  # 2GB+ and higher utilization
                config.update({
                    'use_gpu': True,
                    'selected_gpu': best_gpu['index'],
                    'gpu_memory_fraction': 0.6,
                    'gpu_memory_limit': int(gpu_memory * 0.6),
                    'batch_size': 2048,
                    'optuna_trials': 600,
                    'shap_samples': 1500,
                    'processing_mode': 'GPU_CONSERVATIVE'
                })
        
        # CPU Optimization based on available memory
        memory_gb = self.system_info['memory_total'] / (1024**3)
        if memory_gb >= 32:
            config.update({
                'cpu_parallel_jobs': min(16, config['cpu_parallel_jobs']),
                'batch_size': max(config['batch_size'], 2048),
                'optuna_trials': max(config['optuna_trials'], 750)
            })
        elif memory_gb >= 16:
            config.update({
                'cpu_parallel_jobs': min(8, config['cpu_parallel_jobs']),
                'batch_size': max(config['batch_size'], 1024),
                'optuna_trials': max(config['optuna_trials'], 600)
            })
        
        return config
    
    def _configure_environment(self):
        """🔧 Configure System Environment for Optimal Performance"""
        try:
            # GPU Environment Configuration
            if self.optimal_config['use_gpu']:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.optimal_config['selected_gpu'])
                os.environ['GPU_FORCE_64BIT_PTR'] = '1'
                os.environ['CUDA_CACHE_DISABLE'] = '0'
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # CPU Optimization
            os.environ['OMP_NUM_THREADS'] = str(self.optimal_config['cpu_threads'])
            os.environ['MKL_NUM_THREADS'] = str(self.optimal_config['cpu_threads'])
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.optimal_config['cpu_threads'])
            
            # Memory Optimization
            os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
        except Exception as e:
            print(f"⚠️ Environment configuration warning: {e}")
    
    def configure_tensorflow(self) -> bool:
        """⚡ Configure TensorFlow for Optimal GPU/CPU Performance"""
        try:
            import tensorflow as tf
            
            if self.optimal_config['use_gpu']:
                # GPU Configuration
                physical_gpus = tf.config.list_physical_devices('GPU')
                if physical_gpus and self.optimal_config['selected_gpu'] < len(physical_gpus):
                    selected_gpu = physical_gpus[self.optimal_config['selected_gpu']]
                    
                    # Enable memory growth
                    tf.config.experimental.set_memory_growth(selected_gpu, True)
                    
                    # Set memory limit
                    memory_limit = self.optimal_config['gpu_memory_limit']
                    tf.config.experimental.set_virtual_device_configuration(
                        selected_gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
                    )
                    
                    print(f"✅ TensorFlow GPU Configured: GPU {self.optimal_config['selected_gpu']} ({memory_limit}MB)")
                    return True
            else:
                # CPU-only configuration
                tf.config.set_visible_devices([], 'GPU')
                tf.config.threading.set_intra_op_parallelism_threads(self.optimal_config['cpu_threads'])
                tf.config.threading.set_inter_op_parallelism_threads(self.optimal_config['cpu_parallel_jobs'])
                
                print(f"✅ TensorFlow CPU Configured: {self.optimal_config['cpu_threads']} threads")
                return False
                
        except Exception as e:
            print(f"⚠️ TensorFlow configuration failed: {e}")
            return False
    
    def configure_pytorch(self) -> bool:
        """🔥 Configure PyTorch for Optimal Performance"""
        try:
            import torch
            
            if self.optimal_config['use_gpu'] and torch.cuda.is_available():
                # Set default GPU
                torch.cuda.set_device(self.optimal_config['selected_gpu'])
                
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(
                    self.optimal_config['gpu_memory_fraction'],
                    device=self.optimal_config['selected_gpu']
                )
                
                print(f"✅ PyTorch GPU Configured: GPU {self.optimal_config['selected_gpu']}")
                return True
            else:
                # CPU configuration
                torch.set_num_threads(self.optimal_config['cpu_threads'])
                print(f"✅ PyTorch CPU Configured: {self.optimal_config['cpu_threads']} threads")
                return False
                
        except Exception as e:
            print(f"⚠️ PyTorch configuration failed: {e}")
            return False
    
    def get_current_usage(self) -> Dict[str, Any]:
        """📊 Get Real-time Resource Usage"""
        usage = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'swap_percent': psutil.swap_memory().percent,
            'gpu_usage': []
        }
        
        # GPU Usage
        if self.gpu_hardware['gpu_available']:
            try:
                for device in self.gpu_hardware['gpu_devices']:
                    gpu_usage = {
                        'index': device['index'],
                        'name': device['name'],
                        'utilization': device.get('utilization', 0),
                        'memory_used': device.get('memory_used', 0),
                        'memory_total': device.get('memory_total', 0),
                        'memory_percent': (device.get('memory_used', 0) / device.get('memory_total', 1)) * 100,
                        'temperature': device.get('temperature'),
                        'power_draw': device.get('power_draw')
                    }
                    usage['gpu_usage'].append(gpu_usage)
            except:
                pass
        
        return usage
    
    def get_optimization_report(self) -> str:
        """📋 Generate Comprehensive Optimization Report"""
        lines = []
        lines.append("🎯 NICEGOLD ENTERPRISE GPU RESOURCE MANAGER")
        lines.append("=" * 60)
        lines.append("")
        
        # System Information
        lines.append("🖥️  SYSTEM SPECIFICATIONS")
        lines.append("-" * 30)
        lines.append(f"Platform: {self.system_info['platform']} {self.system_info['platform_version']}")
        lines.append(f"Architecture: {self.system_info['architecture']}")
        lines.append(f"CPU: {self.system_info['processor']}")
        lines.append(f"Logical CPUs: {self.system_info['cpu_count_logical']}")
        lines.append(f"Physical CPUs: {self.system_info['cpu_count_physical']}")
        lines.append(f"Total Memory: {self.system_info['memory_total'] / (1024**3):.1f} GB")
        lines.append("")
        
        # GPU Information
        lines.append("🎮 GPU HARDWARE DETECTION")
        lines.append("-" * 30)
        if self.gpu_hardware['gpu_available']:
            lines.append(f"GPUs Available: {self.gpu_hardware['gpu_count']}")
            lines.append(f"Total GPU Memory: {self.gpu_hardware['total_gpu_memory']} MB")
            lines.append(f"CUDA Available: {self.gpu_hardware['cuda_available']}")
            lines.append(f"Driver Version: {self.gpu_hardware['driver_version']}")
            lines.append("")
            
            for i, device in enumerate(self.gpu_hardware['gpu_devices']):
                lines.append(f"  GPU {device['index']}: {device['name']}")
                lines.append(f"    Memory: {device['memory_total']} MB")
                lines.append(f"    Used: {device.get('memory_used', 0)} MB ({device.get('memory_used', 0)/device['memory_total']*100:.1f}%)")
                lines.append(f"    Utilization: {device.get('utilization', 0)}%")
                if device.get('temperature'):
                    lines.append(f"    Temperature: {device['temperature']}°C")
                lines.append("")
        else:
            lines.append("No GPU detected - CPU-only mode")
            lines.append("")
        
        # Optimization Configuration
        lines.append("⚡ OPTIMIZATION CONFIGURATION")
        lines.append("-" * 30)
        lines.append(f"Processing Mode: {self.optimal_config['processing_mode']}")
        lines.append(f"GPU Enabled: {'YES' if self.optimal_config['use_gpu'] else 'NO'}")
        
        if self.optimal_config['use_gpu']:
            lines.append(f"Selected GPU: {self.optimal_config['selected_gpu']}")
            lines.append(f"GPU Memory Limit: {self.optimal_config['gpu_memory_limit']} MB ({self.optimal_config['gpu_memory_fraction']*100}%)")
        
        lines.append(f"CPU Threads: {self.optimal_config['cpu_threads']}")
        lines.append(f"Parallel Jobs: {self.optimal_config['cpu_parallel_jobs']}")
        lines.append(f"Batch Size: {self.optimal_config['batch_size']}")
        lines.append(f"Optuna Trials: {self.optimal_config['optuna_trials']}")
        lines.append(f"SHAP Samples: {self.optimal_config['shap_samples']}")
        lines.append("")
        
        # Current Usage
        current_usage = self.get_current_usage()
        lines.append("📊 CURRENT RESOURCE USAGE")
        lines.append("-" * 30)
        lines.append(f"CPU Usage: {current_usage['cpu_percent']:.1f}%")
        lines.append(f"Memory Usage: {current_usage['memory_percent']:.1f}%")
        lines.append(f"Swap Usage: {current_usage['swap_percent']:.1f}%")
        
        if current_usage['gpu_usage']:
            for gpu in current_usage['gpu_usage']:
                lines.append(f"GPU {gpu['index']} Usage: {gpu['utilization']:.1f}%")
                lines.append(f"GPU {gpu['index']} Memory: {gpu['memory_percent']:.1f}%")
        
        return "\n".join(lines)
    
    def save_configuration(self, filepath: str = None):
        """💾 Save Configuration to File"""
        if not filepath:
            filepath = f"nicegold_gpu_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        config_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'gpu_hardware': self.gpu_hardware,
            'optimal_config': self.optimal_config,
            'current_usage': self.get_current_usage()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2, default=str)
            print(f"✅ Configuration saved to: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")

# 🧪 TESTING AND UTILITIES
def test_gpu_manager():
    """🧪 Comprehensive GPU Manager Testing"""
    print("🎯 TESTING NICEGOLD GPU RESOURCE MANAGER")
    print("=" * 50)
    
    try:
        manager = EnterpriseGPUManager()
        print("✅ GPU Manager initialized successfully")
        
        # Print comprehensive report
        print(manager.get_optimization_report())
        
        # Test TensorFlow configuration
        print("\n🔧 TESTING TENSORFLOW CONFIGURATION")
        tf_success = manager.configure_tensorflow()
        print(f"TensorFlow GPU: {'ENABLED' if tf_success else 'DISABLED'}")
        
        # Test PyTorch configuration  
        print("\n🔥 TESTING PYTORCH CONFIGURATION")
        torch_success = manager.configure_pytorch()
        print(f"PyTorch GPU: {'ENABLED' if torch_success else 'DISABLED'}")
        
        # Save configuration
        manager.save_configuration()
        
        return manager
        
    except Exception as e:
        print(f"❌ GPU Manager test failed: {e}")
        return None

if __name__ == "__main__":
    test_gpu_manager()
