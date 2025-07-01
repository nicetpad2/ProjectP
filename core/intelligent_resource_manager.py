#!/usr/bin/env python3
"""
🧠 INTELLIGENT RESOURCE MANAGEMENT SYSTEM
===========================================

ระบบจัดการทรัพยากรอัจฉริยะขั้นสูงสำหรับ NICEGOLD ProjectP Elliott Wave Pipeline
ตรวจจับและจัดสรร CPU, RAM, GPU อัตโนมัติด้วยกลยุทธ์ 80% allocation

🎯 Key Features:
- Smart Hardware Detection (CPU, RAM, GPU)
- 80% Optimal Resource Allocation Strategy
- Real-time Performance Monitoring
- Menu 1 Elliott Wave Pipeline Integration
- Enterprise-grade Resource Management

📊 Performance Targets:
- CPU Utilization: 80% of available cores
- Memory Allocation: 80% of available RAM
- GPU Selection: Best available device
- Batch Size Optimization: Resource-based dynamic sizing
"""

import os
import sys
import psutil
import platform
import threading
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentResourceManager:
    """
    🧠 Intelligent Resource Management System
    ระบบจัดการทรัพยากรอัจฉริยะขั้นสูง
    """
    
    def __init__(self, allocation_percentage: float = 0.8):
        """
        Initialize Intelligent Resource Manager
        
        Args:
            allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร (default: 80%)
        """
        self.allocation_percentage = allocation_percentage
        self.system_info = {}
        self.resource_config = {}
        self.monitoring_active = False
        self.performance_data = []
        self.start_time = datetime.now()
        
        # Initialize system detection
        self._detect_system_resources()
        self._calculate_optimal_allocation()
        
        logger.info("🧠 Intelligent Resource Manager initialized successfully")
    
    def _detect_system_resources(self) -> Dict[str, Any]:
        """
        🔍 ตรวจจับทรัพยากรระบบทั้งหมด
        """
        try:
            # CPU Detection
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                'cpu_percent': psutil.cpu_percent(interval=1),
                'architecture': platform.machine(),
                'processor': platform.processor()
            }
            
            # Memory Detection
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            memory_info = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': memory.percent,
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2)
            }
            
            # GPU Detection
            gpu_info = self._detect_gpu_resources()
            
            # Platform Information
            platform_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'python_version': platform.python_version(),
                'python_arch': platform.architecture()[0]
            }
            
            # Storage Information
            storage_info = self._detect_storage_resources()
            
            self.system_info = {
                'cpu': cpu_info,
                'memory': memory_info,
                'gpu': gpu_info,
                'platform': platform_info,
                'storage': storage_info,
                'detection_time': datetime.now().isoformat()
            }
            
            logger.info("🔍 System resource detection completed")
            return self.system_info
            
        except Exception as e:
            logger.error(f"❌ System resource detection failed: {e}")
            return {}
    
    def _detect_gpu_resources(self) -> Dict[str, Any]:
        """
        🚀 ตรวจจับ GPU และ CUDA capabilities
        """
        gpu_info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_devices': [],
            'recommended_device': None
        }
        
        try:
            # Try CUDA detection
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info['cuda_available'] = True
                    gpu_info['gpu_count'] = torch.cuda.device_count()
                    
                    devices = []
                    for i in range(gpu_info['gpu_count']):
                        device_props = torch.cuda.get_device_properties(i)
                        devices.append({
                            'device_id': i,
                            'name': device_props.name,
                            'memory_total_gb': round(device_props.total_memory / (1024**3), 2),
                            'capability': f"{device_props.major}.{device_props.minor}"
                        })
                    
                    gpu_info['gpu_devices'] = devices
                    # Recommend device with most memory
                    if devices:
                        gpu_info['recommended_device'] = max(devices, key=lambda x: x['memory_total_gb'])
                    
                    logger.info(f"🚀 CUDA detected: {gpu_info['gpu_count']} devices")
                else:
                    logger.info("🖥️ CUDA not available, using CPU-only mode")
            except ImportError:
                logger.info("📦 PyTorch not installed, checking alternatives...")
                
                # Try TensorFlow detection
                try:
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        gpu_info['cuda_available'] = True
                        gpu_info['gpu_count'] = len(gpus)
                        logger.info(f"🚀 TensorFlow GPU detected: {len(gpus)} devices")
                except ImportError:
                    logger.info("📦 TensorFlow not installed")
                    
        except Exception as e:
            logger.warning(f"⚠️ GPU detection error: {e}")
        
        return gpu_info
    
    def _detect_storage_resources(self) -> Dict[str, Any]:
        """
        💾 ตรวจจับทรัพยากรที่เก็บข้อมูล
        """
        storage_info = {
            'disks': [],
            'total_space_gb': 0,
            'available_space_gb': 0
        }
        
        try:
            # Get disk usage for all partitions
            partitions = psutil.disk_partitions()
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info = {
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'filesystem': partition.fstype,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'percent': round((usage.used / usage.total) * 100, 1)
                    }
                    storage_info['disks'].append(disk_info)
                    storage_info['total_space_gb'] += disk_info['total_gb']
                    storage_info['available_space_gb'] += disk_info['free_gb']
                except PermissionError:
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Storage detection error: {e}")
        
        return storage_info
    
    def _calculate_optimal_allocation(self) -> Dict[str, Any]:
        """
        ⚡ คำนวณการจัดสรรทรัพยากรที่เหมาะสมที่สุด (80% Strategy)
        """
        try:
            cpu_info = self.system_info.get('cpu', {})
            memory_info = self.system_info.get('memory', {})
            gpu_info = self.system_info.get('gpu', {})
            
            # CPU Allocation (80% of logical cores)
            logical_cores = cpu_info.get('logical_cores', 1)
            allocated_threads = max(1, int(logical_cores * self.allocation_percentage))
            
            # Memory Allocation (80% of available memory)
            available_memory_gb = memory_info.get('available_gb', 4.0)
            allocated_memory_gb = round(available_memory_gb * self.allocation_percentage, 2)
            
            # GPU Selection
            gpu_config = {
                'use_gpu': gpu_info.get('cuda_available', False),
                'device_id': 0,
                'device_info': gpu_info.get('recommended_device', {})
            }
            
            # Batch Size Optimization
            batch_size = self._calculate_optimal_batch_size(allocated_memory_gb, gpu_config)
            
            # ML Framework Configuration
            ml_config = self._generate_ml_framework_config(allocated_threads, allocated_memory_gb, gpu_config)
            
            self.resource_config = {
                'cpu': {
                    'total_cores': logical_cores,
                    'allocated_threads': allocated_threads,
                    'allocation_percentage': round((allocated_threads / logical_cores) * 100, 1)
                },
                'memory': {
                    'total_gb': memory_info.get('total_gb', 0),
                    'available_gb': available_memory_gb,
                    'allocated_gb': allocated_memory_gb,
                    'allocation_percentage': round((allocated_memory_gb / available_memory_gb) * 100, 1)
                },
                'gpu': gpu_config,
                'optimization': {
                    'batch_size': batch_size,
                    'recommended_workers': allocated_threads,
                    'memory_limit_gb': allocated_memory_gb
                },
                'ml_frameworks': ml_config,
                'calculation_time': datetime.now().isoformat()
            }
            
            logger.info(f"⚡ Optimal allocation calculated: {allocated_threads} threads, {allocated_memory_gb}GB RAM")
            return self.resource_config
            
        except Exception as e:
            logger.error(f"❌ Resource allocation calculation failed: {e}")
            return {}
    
    def _calculate_optimal_batch_size(self, allocated_memory_gb: float, gpu_config: Dict) -> int:
        """
        📦 คำนวณ batch size ที่เหมาะสมตามทรัพยากรที่มี
        """
        try:
            if gpu_config.get('use_gpu', False):
                # GPU-based batch size calculation
                gpu_memory_gb = gpu_config.get('device_info', {}).get('memory_total_gb', 4.0)
                # Reserve 20% for GPU overhead
                usable_gpu_memory = gpu_memory_gb * 0.8
                # Estimate batch size (rough estimation: 1GB can handle batch size ~64)
                batch_size = max(8, min(256, int(usable_gpu_memory * 16)))
            else:
                # CPU-based batch size calculation
                # More conservative for CPU processing
                batch_size = max(4, min(64, int(allocated_memory_gb * 4)))
            
            # Ensure batch size is power of 2 for optimal performance
            batch_size = 2 ** int(np.log2(batch_size)) if 'np' in globals() else batch_size
            
            return batch_size
            
        except Exception:
            return 32  # Safe default
    
    def _generate_ml_framework_config(self, threads: int, memory_gb: float, gpu_config: Dict) -> Dict:
        """
        🤖 สร้าง configuration สำหรับ ML frameworks
        """
        config = {
            'tensorflow': {
                'inter_op_parallelism_threads': threads,
                'intra_op_parallelism_threads': threads,
                'memory_limit_mb': int(memory_gb * 1024 * 0.7),  # 70% for TF
                'use_gpu': gpu_config.get('use_gpu', False)
            },
            'pytorch': {
                'num_threads': threads,
                'memory_limit_gb': round(memory_gb * 0.7, 2),
                'use_cuda': gpu_config.get('use_gpu', False),
                'device': f"cuda:{gpu_config.get('device_id', 0)}" if gpu_config.get('use_gpu') else "cpu"
            },
            'sklearn': {
                'n_jobs': threads,
                'memory_limit_mb': int(memory_gb * 1024 * 0.5)  # 50% for sklearn
            },
            'general': {
                'max_workers': threads,
                'chunk_size': max(100, int(10000 / threads)),
                'parallel_backend': 'threading' if memory_gb > 8 else 'loky'
            }
        }
        
        return config
    
    def apply_resource_optimization(self) -> bool:
        """
        🎯 ประยุกต์การปรับแต่งทรัพยากรกับระบบ
        """
        try:
            config = self.resource_config
            
            # Set environment variables for ML frameworks
            ml_config = config.get('ml_frameworks', {})
            
            # TensorFlow optimization
            tf_config = ml_config.get('tensorflow', {})
            os.environ['TF_NUM_INTEROP_THREADS'] = str(tf_config.get('inter_op_parallelism_threads', 4))
            os.environ['TF_NUM_INTRAOP_THREADS'] = str(tf_config.get('intra_op_parallelism_threads', 4))
            
            # PyTorch optimization
            pytorch_config = ml_config.get('pytorch', {})
            os.environ['OMP_NUM_THREADS'] = str(pytorch_config.get('num_threads', 4))
            os.environ['MKL_NUM_THREADS'] = str(pytorch_config.get('num_threads', 4))
            
            # General optimization
            general_config = ml_config.get('general', {})
            os.environ['NUMBA_NUM_THREADS'] = str(general_config.get('max_workers', 4))
            
            # GPU configuration
            gpu_config = config.get('gpu', {})
            if not gpu_config.get('use_gpu', False):
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            logger.info("🎯 Resource optimization applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Resource optimization failed: {e}")
            return False
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """
        📊 เริ่มการติดตามประสิทธิภาพแบบ real-time
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("📊 Real-time performance monitoring started")
    
    def stop_monitoring(self) -> None:
        """
        ⏹️ หยุดการติดตามประสิทธิภาพ
        """
        self.monitoring_active = False
        logger.info("⏹️ Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: float) -> None:
        """
        🔄 Loop การติดตามประสิทธิภาพ
        """
        while self.monitoring_active:
            try:
                # Collect current resource usage
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used_gb': round(psutil.virtual_memory().used / (1024**3), 2),
                    'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
                }
                
                # Add GPU monitoring if available
                if self.system_info.get('gpu', {}).get('cuda_available', False):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.memory_stats(0)
                            data_point['gpu_memory_used_gb'] = round(gpu_memory.get('reserved_bytes.all.current', 0) / (1024**3), 2)
                    except:
                        pass
                
                self.performance_data.append(data_point)
                
                # Keep only last 1000 data points to prevent memory issues
                if len(self.performance_data) > 1000:
                    self.performance_data = self.performance_data[-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(interval)
    
    def get_current_performance(self) -> Dict[str, Any]:
        """
        📈 รับข้อมูลประสิทธิภาพปัจจุบัน
        """
        try:
            current_stats = {
                'cpu_percent': psutil.cpu_percent(),
                'memory': psutil.virtual_memory()._asdict(),
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0],
                'uptime_minutes': (datetime.now() - self.start_time).total_seconds() / 60
            }
            
            # Add recent performance history
            if self.performance_data:
                current_stats['recent_history'] = self.performance_data[-10:]  # Last 10 data points
            
            return current_stats
            
        except Exception as e:
            logger.error(f"❌ Performance data collection failed: {e}")
            return {}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        📋 สร้างรายงานประสิทธิภาพที่ครอบคลุม
        """
        try:
            report = {
                'system_info': self.system_info,
                'resource_allocation': self.resource_config,
                'current_performance': self.get_current_performance(),
                'optimization_recommendations': self._generate_optimization_recommendations(),
                'report_timestamp': datetime.now().isoformat()
            }
            
            if self.performance_data:
                # Calculate performance statistics
                cpu_values = [d['cpu_percent'] for d in self.performance_data if 'cpu_percent' in d]
                memory_values = [d['memory_percent'] for d in self.performance_data if 'memory_percent' in d]
                
                if cpu_values and memory_values:
                    report['performance_statistics'] = {
                        'cpu': {
                            'avg': round(sum(cpu_values) / len(cpu_values), 2),
                            'max': max(cpu_values),
                            'min': min(cpu_values)
                        },
                        'memory': {
                            'avg': round(sum(memory_values) / len(memory_values), 2),
                            'max': max(memory_values),
                            'min': min(memory_values)
                        },
                        'data_points': len(self.performance_data)
                    }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Performance report generation failed: {e}")
            return {}
    
    def _generate_optimization_recommendations(self) -> list:
        """
        💡 สร้างคำแนะนำการปรับแต่งประสิทธิภาพ
        """
        recommendations = []
        
        try:
            cpu_info = self.system_info.get('cpu', {})
            memory_info = self.system_info.get('memory', {})
            gpu_info = self.system_info.get('gpu', {})
            
            # CPU recommendations
            if cpu_info.get('logical_cores', 0) < 4:
                recommendations.append({
                    'type': 'hardware',
                    'priority': 'high',
                    'message': '🔥 Consider upgrading to a CPU with more cores for better ML performance'
                })
            
            # Memory recommendations
            if memory_info.get('total_gb', 0) < 8:
                recommendations.append({
                    'type': 'hardware',
                    'priority': 'high',
                    'message': '🧠 Consider adding more RAM (minimum 16GB recommended for ML workloads)'
                })
            
            # GPU recommendations
            if not gpu_info.get('cuda_available', False):
                recommendations.append({
                    'type': 'hardware',
                    'priority': 'medium',
                    'message': '🚀 Consider adding a CUDA-compatible GPU for faster training'
                })
            
            # Performance recommendations
            if self.performance_data:
                recent_cpu = [d['cpu_percent'] for d in self.performance_data[-10:] if 'cpu_percent' in d]
                if recent_cpu and max(recent_cpu) > 90:
                    recommendations.append({
                        'type': 'performance',
                        'priority': 'medium',
                        'message': '⚠️ High CPU usage detected. Consider reducing batch size or parallel workers'
                    })
            
            return recommendations
            
        except Exception:
            return []
    
    def save_report(self, file_path: Optional[str] = None) -> str:
        """
        💾 บันทึกรายงานประสิทธิภาพ
        """
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"resource_report_{timestamp}.json"
            
            report = self.generate_performance_report()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Performance report saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Report saving failed: {e}")
            return ""
    
    def get_menu1_optimization_config(self) -> Dict[str, Any]:
        """
        🌊 รับ configuration ที่เหมาะสมสำหรับ Menu 1 Elliott Wave Pipeline
        """
        try:
            config = self.resource_config
            optimization = config.get('optimization', {})
            ml_frameworks = config.get('ml_frameworks', {})
            
            menu1_config = {
                'data_processing': {
                    'chunk_size': optimization.get('recommended_workers', 4) * 1000,
                    'parallel_workers': optimization.get('recommended_workers', 4),
                    'memory_limit_gb': optimization.get('memory_limit_gb', 4.0)
                },
                'elliott_wave': {
                    'batch_size': optimization.get('batch_size', 32),
                    'use_gpu': config.get('gpu', {}).get('use_gpu', False),
                    'workers': optimization.get('recommended_workers', 4)
                },
                'feature_selection': {
                    'n_trials': min(100, optimization.get('recommended_workers', 4) * 10),
                    'n_jobs': optimization.get('recommended_workers', 4),
                    'memory_limit': optimization.get('memory_limit_gb', 4.0)
                },
                'cnn_lstm': {
                    'batch_size': optimization.get('batch_size', 32),
                    'workers': min(4, optimization.get('recommended_workers', 4)),
                    'use_gpu': config.get('gpu', {}).get('use_gpu', False)
                },
                'dqn': {
                    'batch_size': min(128, optimization.get('batch_size', 32) * 2),
                    'memory_size': optimization.get('batch_size', 32) * 100,
                    'use_gpu': config.get('gpu', {}).get('use_gpu', False)
                },
                'monitoring': {
                    'enable_monitoring': True,
                    'monitoring_interval': 2.0
                }
            }
            
            return menu1_config
            
        except Exception as e:
            logger.error(f"❌ Menu 1 config generation failed: {e}")
            return {}
    
    def display_system_summary(self) -> None:
        """
        🖥️ แสดงสรุปข้อมูลระบบและการจัดสรร
        """
        try:
            print("\n" + "="*70)
            print("🧠 INTELLIGENT RESOURCE MANAGEMENT SYSTEM")
            print("="*70)
            
            # System Information
            cpu_info = self.system_info.get('cpu', {})
            memory_info = self.system_info.get('memory', {})
            gpu_info = self.system_info.get('gpu', {})
            platform_info = self.system_info.get('platform', {})
            
            print(f"\n🖥️  SYSTEM INFORMATION:")
            print(f"   Platform: {platform_info.get('system', 'Unknown')} {platform_info.get('release', '')}")
            print(f"   CPU: {cpu_info.get('physical_cores', 0)} cores ({cpu_info.get('logical_cores', 0)} threads)")
            print(f"   RAM: {memory_info.get('total_gb', 0):.1f} GB (Available: {memory_info.get('available_gb', 0):.1f} GB)")
            print(f"   GPU: {'✅ CUDA Available' if gpu_info.get('cuda_available', False) else '❌ CPU Only'}")
            
            # Resource Allocation
            cpu_config = self.resource_config.get('cpu', {})
            memory_config = self.resource_config.get('memory', {})
            optimization = self.resource_config.get('optimization', {})
            
            print(f"\n⚡ RESOURCE ALLOCATION (80% Strategy):")
            print(f"   CPU Threads: {cpu_config.get('allocated_threads', 0)} / {cpu_config.get('total_cores', 0)} ({cpu_config.get('allocation_percentage', 0):.1f}%)")
            print(f"   Memory: {memory_config.get('allocated_gb', 0):.1f} GB / {memory_config.get('available_gb', 0):.1f} GB ({memory_config.get('allocation_percentage', 0):.1f}%)")
            print(f"   Batch Size: {optimization.get('batch_size', 32)}")
            print(f"   Workers: {optimization.get('recommended_workers', 4)}")
            
            # Current Performance
            current = self.get_current_performance()
            print(f"\n📊 CURRENT PERFORMANCE:")
            print(f"   CPU Usage: {current.get('cpu_percent', 0):.1f}%")
            print(f"   Memory Usage: {current.get('memory', {}).get('percent', 0):.1f}%")
            print(f"   Uptime: {current.get('uptime_minutes', 0):.1f} minutes")
            
            print("="*70)
            
        except Exception as e:
            logger.error(f"❌ System summary display failed: {e}")


def initialize_intelligent_resources(allocation_percentage: float = 0.8, 
                                      enable_monitoring: bool = True) -> IntelligentResourceManager:
    """
    🚀 Initialize and configure Intelligent Resource Management System
    
    Args:
        allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร (default: 80%)
        enable_monitoring: เปิดใช้งานการติดตามประสิทธิภาพ
    
    Returns:
        IntelligentResourceManager: ตัวจัดการทรัพยากรที่พร้อมใช้งาน
    """
    try:
        # Initialize resource manager
        resource_manager = IntelligentResourceManager(allocation_percentage)
        
        # Apply optimization
        resource_manager.apply_resource_optimization()
        
        # Start monitoring if requested
        if enable_monitoring:
            resource_manager.start_monitoring()
        
        # Display system summary
        resource_manager.display_system_summary()
        
        logger.info("🚀 Intelligent Resource Management System ready")
        return resource_manager
        
    except Exception as e:
        logger.error(f"❌ Resource management initialization failed: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    try:
        print("🧠 Testing Intelligent Resource Management System...")
        
        # Initialize system
        resource_manager = initialize_intelligent_resources()
        
        # Test Menu 1 configuration
        menu1_config = resource_manager.get_menu1_optimization_config()
        print(f"\n🌊 Menu 1 Configuration: {json.dumps(menu1_config, indent=2)}")
        
        # Wait a bit for monitoring data
        import time
        time.sleep(5)
        
        # Generate and save report
        report_file = resource_manager.save_report()
        print(f"\n📋 Performance report saved: {report_file}")
        
        # Stop monitoring
        resource_manager.stop_monitoring()
        
        print("\n✅ Intelligent Resource Management System test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
