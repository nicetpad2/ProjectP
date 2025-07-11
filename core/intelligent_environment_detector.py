#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 NICEGOLD ENTERPRISE PROJECTP - INTELLIGENT ENVIRONMENT DETECTOR
ระบบตรวจจับสภาพแวดล้อมอัจฉริยะ (Intelligent Environment Detector)

🎯 Enterprise Features:
✅ ระบบตรวจจับสภาพแวดล้อมอัจฉริยะ (Cloud, Local, Colab, etc.)
✅ การตรวจจับฮาร์ดแวร์และทรัพยากรแบบอัตโนมัติ
✅ การปรับแต่งการตั้งค่าตามสภาพแวดล้อม
✅ ระบบจัดสรรทรัพยากร 80% ที่ชาญฉลาด
✅ การตรวจจับ GPU, CUDA, และ AI frameworks
✅ รองรับ Multi-platform (Windows, Linux, macOS)
✅ ระบบ fallback และ error handling

เวอร์ชัน: 1.0 Enterprise Edition
วันที่: 9 กรกฎาคม 2025
สถานะ: Production Ready
"""

import os
import sys
import platform
import logging
import warnings
import subprocess
import importlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import unified logger
try:
    from core.unified_enterprise_logger import get_unified_logger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

# System resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# GPU Detection
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# PyTorch Detection
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# TensorFlow Detection
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
    
    # Check TensorFlow GPU
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        TENSORFLOW_GPU_AVAILABLE = len(physical_devices) > 0
    except:
        TENSORFLOW_GPU_AVAILABLE = False
except ImportError:
    TENSORFLOW_AVAILABLE = False
    TENSORFLOW_GPU_AVAILABLE = False

# Google Colab Detection
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Jupyter Detection
try:
    import IPython
    IN_JUPYTER = True
except ImportError:
    IN_JUPYTER = False


# ====================================================
# ENUMERATIONS AND DATA STRUCTURES
# ====================================================

class EnvironmentType(Enum):
    """ประเภทของสภาพแวดล้อม"""
    GOOGLE_COLAB = "google_colab"
    JUPYTER_NOTEBOOK = "jupyter_notebook"
    CLOUD_VM = "cloud_vm"
    LOCAL_MACHINE = "local_machine"
    DOCKER_CONTAINER = "docker_container"
    VIRTUAL_MACHINE = "virtual_machine"
    UNKNOWN = "unknown"


class HardwareCapability(Enum):
    """ความสามารถของฮาร์ดแวร์"""
    HIGH_PERFORMANCE = "high_performance"  # >16GB RAM, >8 cores, GPU
    MEDIUM_PERFORMANCE = "medium_performance"  # 8-16GB RAM, 4-8 cores
    LOW_PERFORMANCE = "low_performance"  # 4-8GB RAM, 2-4 cores
    MINIMAL_PERFORMANCE = "minimal_performance"  # <4GB RAM, <2 cores
    UNKNOWN = "unknown"


class ResourceOptimizationLevel(Enum):
    """ระดับการปรับแต่งทรัพยากร"""
    AGGRESSIVE = "aggressive"  # 85% utilization
    STANDARD = "standard"  # 80% utilization
    CONSERVATIVE = "conservative"  # 70% utilization
    MINIMAL = "minimal"  # 50% utilization


@dataclass
class EnvironmentInfo:
    """ข้อมูลสภาพแวดล้อม"""
    environment_type: EnvironmentType
    hardware_capability: HardwareCapability
    optimization_level: ResourceOptimizationLevel
    cpu_cores: int
    memory_gb: float
    disk_gb: float
    gpu_count: int
    gpu_memory_gb: float
    operating_system: str
    python_version: str
    capabilities: Dict[str, bool] = field(default_factory=dict)
    restrictions: Dict[str, Any] = field(default_factory=dict)
    recommendations: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceAllocation:
    """การจัดสรรทรัพยากร"""
    cpu_percentage: float
    memory_percentage: float
    disk_percentage: float
    gpu_percentage: float
    target_utilization: float
    safety_margin: float
    emergency_reserve: float
    details: Dict[str, Any] = field(default_factory=dict)


# ====================================================
# INTELLIGENT ENVIRONMENT DETECTOR
# ====================================================

class IntelligentEnvironmentDetector:
    """
    🧠 ระบบตรวจจับสภาพแวดล้อมอัจฉริยะ
    ตรวจจับและวิเคราะห์สภาพแวดล้อมการทำงาน แล้วจัดสรรทรัพยากรให้เหมาะสม
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """เริ่มต้น Intelligent Environment Detector"""
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # Detection cache
        self._environment_info = None
        self._last_detection_time = None
        self._detection_cache_timeout = 300  # 5 minutes
        
        # Optimization settings
        self.optimization_profiles = {
            ResourceOptimizationLevel.AGGRESSIVE: {
                'target_utilization': 0.85,
                'safety_margin': 0.10,
                'emergency_reserve': 0.05
            },
            ResourceOptimizationLevel.STANDARD: {
                'target_utilization': 0.80,
                'safety_margin': 0.15,
                'emergency_reserve': 0.05
            },
            ResourceOptimizationLevel.CONSERVATIVE: {
                'target_utilization': 0.70,
                'safety_margin': 0.20,
                'emergency_reserve': 0.10
            },
            ResourceOptimizationLevel.MINIMAL: {
                'target_utilization': 0.50,
                'safety_margin': 0.30,
                'emergency_reserve': 0.20
            }
        }
        
        self.logger.info("🧠 Intelligent Environment Detector initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """ติดตั้ง Logger"""
        if LOGGER_AVAILABLE:
            logger = get_unified_logger()
        else:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - 🧠 [%(name)s] - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        
        return logger
    
    def detect_environment(self, force_refresh: bool = False) -> EnvironmentInfo:
        """ตรวจจับสภาพแวดล้อมและฮาร์ดแวร์"""
        # Check cache first
        if not force_refresh and self._environment_info and self._last_detection_time:
            time_diff = time.time() - self._last_detection_time
            if time_diff < self._detection_cache_timeout:
                return self._environment_info
        
        try:
            self.logger.info("🔍 Detecting environment and hardware...")
            
            # Detect environment type
            env_type = self._detect_environment_type()
            self.logger.info(f"🌍 Environment: {env_type.value}")
            
            # Detect hardware specifications
            hardware_info = self._detect_hardware_specifications()
            
            # Determine hardware capability
            hardware_capability = self._determine_hardware_capability(hardware_info)
            self.logger.info(f"💻 Hardware Capability: {hardware_capability.value}")
            
            # Determine optimization level
            optimization_level = self._determine_optimization_level(env_type, hardware_capability)
            self.logger.info(f"⚡ Optimization Level: {optimization_level.value}")
            
            # Detect capabilities and restrictions
            capabilities = self._detect_capabilities()
            restrictions = self._detect_restrictions(env_type)
            recommendations = self._generate_recommendations(env_type, hardware_capability)
            
            # Create environment info
            environment_info = EnvironmentInfo(
                environment_type=env_type,
                hardware_capability=hardware_capability,
                optimization_level=optimization_level,
                cpu_cores=hardware_info.get('cpu_cores', 0),
                memory_gb=hardware_info.get('memory_gb', 0),
                disk_gb=hardware_info.get('disk_gb', 0),
                gpu_count=hardware_info.get('gpu_count', 0),
                gpu_memory_gb=hardware_info.get('gpu_memory_gb', 0),
                operating_system=hardware_info.get('operating_system', 'Unknown'),
                python_version=hardware_info.get('python_version', 'Unknown'),
                capabilities=capabilities,
                restrictions=restrictions,
                recommendations=recommendations
            )
            
            # Cache the result
            self._environment_info = environment_info
            self._last_detection_time = time.time()
            
            self.logger.info("✅ Environment detection completed")
            return environment_info
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting environment: {e}")
            # Return minimal fallback environment
            return EnvironmentInfo(
                environment_type=EnvironmentType.UNKNOWN,
                hardware_capability=HardwareCapability.UNKNOWN,
                optimization_level=ResourceOptimizationLevel.CONSERVATIVE,
                cpu_cores=1,
                memory_gb=1.0,
                disk_gb=10.0,
                gpu_count=0,
                gpu_memory_gb=0.0,
                operating_system=platform.system(),
                python_version=platform.python_version(),
                capabilities={},
                restrictions={'error': str(e)},
                recommendations={}
            )
    
    def _detect_environment_type(self) -> EnvironmentType:
        """ตรวจจับประเภทสภาพแวดล้อม"""
        # Check Google Colab
        if IN_COLAB:
            return EnvironmentType.GOOGLE_COLAB
        
        # Check Jupyter Notebook
        if IN_JUPYTER:
            return EnvironmentType.JUPYTER_NOTEBOOK
        
        # Check Docker
        if self._is_docker_environment():
            return EnvironmentType.DOCKER_CONTAINER
        
        # Check Cloud VM
        if self._is_cloud_vm():
            return EnvironmentType.CLOUD_VM
        
        # Check Virtual Machine
        if self._is_virtual_machine():
            return EnvironmentType.VIRTUAL_MACHINE
        
        # Default to local machine
        return EnvironmentType.LOCAL_MACHINE
    
    def _is_docker_environment(self) -> bool:
        """ตรวจสอบว่าอยู่ใน Docker หรือไม่"""
        try:
            # Check for Docker-specific files
            if os.path.exists('/.dockerenv'):
                return True
                
            # Check cgroup for Docker
            if os.path.exists('/proc/1/cgroup'):
                with open('/proc/1/cgroup', 'r') as f:
                    content = f.read()
                    if 'docker' in content or 'containerd' in content:
                        return True
            
            return False
        except:
            return False
    
    def _is_cloud_vm(self) -> bool:
        """ตรวจสอบว่าอยู่ใน Cloud VM หรือไม่"""
        try:
            # Check for cloud provider metadata
            cloud_indicators = [
                '/sys/class/dmi/id/product_name',
                '/sys/class/dmi/id/sys_vendor',
                '/sys/class/dmi/id/bios_vendor'
            ]
            
            for indicator in cloud_indicators:
                if os.path.exists(indicator):
                    try:
                        with open(indicator, 'r') as f:
                            content = f.read().lower()
                            if any(provider in content for provider in 
                                  ['amazon', 'google', 'microsoft', 'azure', 'aws', 'gcp']):
                                return True
                    except:
                        continue
            
            # Check environment variables
            cloud_env_vars = ['AWS_EXECUTION_ENV', 'GOOGLE_CLOUD_PROJECT', 'AZURE_CLIENT_ID']
            for var in cloud_env_vars:
                if os.environ.get(var):
                    return True
            
            return False
        except:
            return False
    
    def _is_virtual_machine(self) -> bool:
        """ตรวจสอบว่าอยู่ใน Virtual Machine หรือไม่"""
        try:
            if PSUTIL_AVAILABLE:
                # Check for virtualization indicators
                if hasattr(psutil, 'virtual_memory'):
                    # Simple heuristic: very specific memory sizes often indicate VMs
                    memory_gb = psutil.virtual_memory().total / (1024**3)
                    if memory_gb in [1.0, 2.0, 4.0, 8.0, 16.0]:  # Common VM memory sizes
                        return True
            
            # Check for hypervisor indicators
            vm_indicators = [
                'vmware', 'virtualbox', 'qemu', 'kvm', 'xen', 'hyper-v'
            ]
            
            try:
                # Check DMI information
                result = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    dmesg_output = result.stdout.lower()
                    if any(indicator in dmesg_output for indicator in vm_indicators):
                        return True
            except:
                pass
            
            return False
        except:
            return False
    
    def _detect_hardware_specifications(self) -> Dict[str, Any]:
        """ตรวจจับข้อมูลฮาร์ดแวร์"""
        hardware_info = {
            'cpu_cores': 1,
            'memory_gb': 1.0,
            'disk_gb': 10.0,
            'gpu_count': 0,
            'gpu_memory_gb': 0.0,
            'operating_system': platform.system(),
            'python_version': platform.python_version()
        }
        
        try:
            # CPU Information
            if PSUTIL_AVAILABLE:
                hardware_info['cpu_cores'] = psutil.cpu_count(logical=True)
                
                # Memory Information
                memory = psutil.virtual_memory()
                hardware_info['memory_gb'] = memory.total / (1024**3)
                
                # Disk Information
                try:
                    disk = psutil.disk_usage('/')
                    hardware_info['disk_gb'] = disk.total / (1024**3)
                except:
                    # Fallback for different mount points
                    try:
                        disk = psutil.disk_usage('.')
                        hardware_info['disk_gb'] = disk.total / (1024**3)
                    except:
                        hardware_info['disk_gb'] = 100.0  # Default fallback
            else:
                # Fallback without psutil
                try:
                    hardware_info['cpu_cores'] = os.cpu_count() or 1
                except:
                    hardware_info['cpu_cores'] = 1
            
            # GPU Information
            gpu_info = self._detect_gpu_information()
            hardware_info['gpu_count'] = gpu_info['count']
            hardware_info['gpu_memory_gb'] = gpu_info['memory_gb']
            
            return hardware_info
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting hardware: {e}")
            return hardware_info
    
    def _detect_gpu_information(self) -> Dict[str, Any]:
        """ตรวจจับข้อมูล GPU"""
        gpu_info = {
            'count': 0,
            'memory_gb': 0.0,
            'names': [],
            'capabilities': {}
        }
        
        try:
            # PyTorch CUDA Detection
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                gpu_info['count'] = torch.cuda.device_count()
                gpu_info['capabilities']['cuda'] = True
                gpu_info['capabilities']['pytorch'] = True
                
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_info['names'].append(gpu_name)
                    
                    # Get memory information
                    try:
                        memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        gpu_info['memory_gb'] += memory_gb
                    except:
                        pass
            
            # GPUtil Detection
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_info['count'] = max(gpu_info['count'], len(gpus))
                        
                        for gpu in gpus:
                            if gpu.name not in gpu_info['names']:
                                gpu_info['names'].append(gpu.name)
                            
                            gpu_info['memory_gb'] += gpu.memoryTotal / 1024.0  # Convert MB to GB
                        
                        gpu_info['capabilities']['gputil'] = True
                except:
                    pass
            
            # TensorFlow GPU Detection
            if TENSORFLOW_AVAILABLE and TENSORFLOW_GPU_AVAILABLE:
                try:
                    physical_devices = tf.config.list_physical_devices('GPU')
                    gpu_info['count'] = max(gpu_info['count'], len(physical_devices))
                    gpu_info['capabilities']['tensorflow'] = True
                except:
                    pass
            
            return gpu_info
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting GPU: {e}")
            return gpu_info
    
    def _determine_hardware_capability(self, hardware_info: Dict[str, Any]) -> HardwareCapability:
        """กำหนดความสามารถของฮาร์ดแวร์"""
        cpu_cores = hardware_info.get('cpu_cores', 1)
        memory_gb = hardware_info.get('memory_gb', 1.0)
        gpu_count = hardware_info.get('gpu_count', 0)
        
        # High Performance: >16GB RAM, >8 cores, GPU
        if memory_gb > 16 and cpu_cores > 8 and gpu_count > 0:
            return HardwareCapability.HIGH_PERFORMANCE
        
        # Medium Performance: 8-16GB RAM, 4-8 cores
        elif memory_gb >= 8 and cpu_cores >= 4:
            return HardwareCapability.MEDIUM_PERFORMANCE
        
        # Low Performance: 4-8GB RAM, 2-4 cores
        elif memory_gb >= 4 and cpu_cores >= 2:
            return HardwareCapability.LOW_PERFORMANCE
        
        # Minimal Performance: <4GB RAM, <2 cores
        else:
            return HardwareCapability.MINIMAL_PERFORMANCE
    
    def _determine_optimization_level(self, env_type: EnvironmentType, 
                                    hardware_capability: HardwareCapability) -> ResourceOptimizationLevel:
        """กำหนดระดับการปรับแต่งทรัพยากร"""
        # Google Colab - Conservative (free resources)
        if env_type == EnvironmentType.GOOGLE_COLAB:
            return ResourceOptimizationLevel.CONSERVATIVE
        
        # Docker - Standard (controlled environment)
        elif env_type == EnvironmentType.DOCKER_CONTAINER:
            return ResourceOptimizationLevel.STANDARD
        
        # Cloud VM - Aggressive (paid resources)
        elif env_type == EnvironmentType.CLOUD_VM:
            return ResourceOptimizationLevel.AGGRESSIVE
        
        # Based on hardware capability
        elif hardware_capability == HardwareCapability.HIGH_PERFORMANCE:
            return ResourceOptimizationLevel.AGGRESSIVE
        
        elif hardware_capability == HardwareCapability.MEDIUM_PERFORMANCE:
            return ResourceOptimizationLevel.STANDARD
        
        elif hardware_capability == HardwareCapability.LOW_PERFORMANCE:
            return ResourceOptimizationLevel.CONSERVATIVE
        
        else:
            return ResourceOptimizationLevel.MINIMAL
    
    def _detect_capabilities(self) -> Dict[str, bool]:
        """ตรวจจับความสามารถของระบบ"""
        capabilities = {
            'psutil': PSUTIL_AVAILABLE,
            'torch': TORCH_AVAILABLE,
            'cuda': CUDA_AVAILABLE,
            'tensorflow': TENSORFLOW_AVAILABLE,
            'tensorflow_gpu': TENSORFLOW_GPU_AVAILABLE,
            'gputil': GPUTIL_AVAILABLE,
            'jupyter': IN_JUPYTER,
            'colab': IN_COLAB,
            'multiprocessing': True,
            'threading': True
        }
        
        # Check for specific Python packages
        packages_to_check = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
            'scipy', 'requests', 'aiohttp', 'asyncio', 'multiprocessing',
            'concurrent.futures', 'joblib', 'dask', 'ray'
        ]
        
        for package in packages_to_check:
            try:
                importlib.import_module(package)
                capabilities[package] = True
            except ImportError:
                capabilities[package] = False
        
        return capabilities
    
    def _detect_restrictions(self, env_type: EnvironmentType) -> Dict[str, Any]:
        """ตรวจจับข้อจำกัดของสภาพแวดล้อม"""
        restrictions = {}
        
        # Google Colab restrictions
        if env_type == EnvironmentType.GOOGLE_COLAB:
            restrictions.update({
                'runtime_limit': '12 hours',
                'memory_limit': 'Variable (typically 12-16GB)',
                'gpu_limit': 'Limited hours per day',
                'disk_limit': 'Temporary storage',
                'network_restrictions': 'Some ports blocked'
            })
        
        # Docker restrictions
        elif env_type == EnvironmentType.DOCKER_CONTAINER:
            restrictions.update({
                'resource_limits': 'Defined by container configuration',
                'filesystem_access': 'Limited to container',
                'network_access': 'Potentially restricted'
            })
        
        # Cloud VM restrictions
        elif env_type == EnvironmentType.CLOUD_VM:
            restrictions.update({
                'billing_based': 'Usage-based billing',
                'network_costs': 'Potential egress charges',
                'resource_scaling': 'May require instance resizing'
            })
        
        return restrictions
    
    def _generate_recommendations(self, env_type: EnvironmentType, 
                                hardware_capability: HardwareCapability) -> Dict[str, Any]:
        """สร้างคำแนะนำสำหรับการปรับแต่ง"""
        recommendations = {}
        
        # Environment-specific recommendations
        if env_type == EnvironmentType.GOOGLE_COLAB:
            recommendations.update({
                'memory_management': 'Use memory-efficient operations',
                'session_management': 'Save progress frequently',
                'gpu_usage': 'Optimize GPU usage for limited runtime',
                'data_handling': 'Use efficient data loading strategies'
            })
        
        elif env_type == EnvironmentType.DOCKER_CONTAINER:
            recommendations.update({
                'resource_monitoring': 'Monitor container resource limits',
                'process_management': 'Optimize process count',
                'memory_allocation': 'Respect container memory limits'
            })
        
        elif env_type == EnvironmentType.CLOUD_VM:
            recommendations.update({
                'cost_optimization': 'Monitor usage for cost efficiency',
                'scaling_strategy': 'Consider auto-scaling options',
                'resource_utilization': 'Maximize resource utilization'
            })
        
        # Hardware-specific recommendations
        if hardware_capability == HardwareCapability.HIGH_PERFORMANCE:
            recommendations.update({
                'parallelization': 'Utilize multi-processing and GPU acceleration',
                'batch_processing': 'Use large batch sizes',
                'memory_utilization': 'Leverage high memory capacity'
            })
        
        elif hardware_capability == HardwareCapability.MEDIUM_PERFORMANCE:
            recommendations.update({
                'balanced_approach': 'Balance CPU and memory usage',
                'moderate_batching': 'Use moderate batch sizes',
                'efficient_algorithms': 'Choose memory-efficient algorithms'
            })
        
        elif hardware_capability == HardwareCapability.LOW_PERFORMANCE:
            recommendations.update({
                'conservative_usage': 'Use conservative resource allocation',
                'small_batches': 'Use small batch sizes',
                'memory_optimization': 'Prioritize memory efficiency'
            })
        
        elif hardware_capability == HardwareCapability.MINIMAL_PERFORMANCE:
            recommendations.update({
                'minimal_usage': 'Use minimal resource allocation',
                'sequential_processing': 'Prefer sequential over parallel processing',
                'memory_conservation': 'Implement aggressive memory conservation'
            })
        
        return recommendations
    
    def get_optimal_resource_allocation(self, env_info: EnvironmentInfo = None) -> ResourceAllocation:
        """ดึงการจัดสรรทรัพยากรที่เหมาะสม"""
        if env_info is None:
            env_info = self.detect_environment()
        
        # Get optimization profile
        profile = self.optimization_profiles[env_info.optimization_level]
        
        # Calculate allocations based on hardware capability
        base_cpu = profile['target_utilization']
        base_memory = profile['target_utilization']
        base_disk = 0.5  # Conservative disk usage
        base_gpu = profile['target_utilization']
        
        # Adjust based on hardware capability
        if env_info.hardware_capability == HardwareCapability.HIGH_PERFORMANCE:
            cpu_percentage = min(base_cpu * 1.0, 0.85)  # Up to 85%
            memory_percentage = min(base_memory * 1.0, 0.85)
            gpu_percentage = min(base_gpu * 1.0, 0.85)
        
        elif env_info.hardware_capability == HardwareCapability.MEDIUM_PERFORMANCE:
            cpu_percentage = min(base_cpu * 0.9, 0.80)  # Up to 80%
            memory_percentage = min(base_memory * 0.9, 0.80)
            gpu_percentage = min(base_gpu * 0.9, 0.80)
        
        elif env_info.hardware_capability == HardwareCapability.LOW_PERFORMANCE:
            cpu_percentage = min(base_cpu * 0.8, 0.70)  # Up to 70%
            memory_percentage = min(base_memory * 0.8, 0.70)
            gpu_percentage = min(base_gpu * 0.8, 0.70)
        
        else:  # MINIMAL_PERFORMANCE
            cpu_percentage = min(base_cpu * 0.6, 0.50)  # Up to 50%
            memory_percentage = min(base_memory * 0.6, 0.50)
            gpu_percentage = min(base_gpu * 0.6, 0.50)
        
        # Adjust for specific environments
        if env_info.environment_type == EnvironmentType.GOOGLE_COLAB:
            # Be more conservative with Colab
            cpu_percentage *= 0.8
            memory_percentage *= 0.8
            gpu_percentage *= 0.8
        
        elif env_info.environment_type == EnvironmentType.DOCKER_CONTAINER:
            # Respect container limits
            cpu_percentage *= 0.9
            memory_percentage *= 0.9
        
        return ResourceAllocation(
            cpu_percentage=cpu_percentage,
            memory_percentage=memory_percentage,
            disk_percentage=base_disk,
            gpu_percentage=gpu_percentage,
            target_utilization=profile['target_utilization'],
            safety_margin=profile['safety_margin'],
            emergency_reserve=profile['emergency_reserve'],
            details={
                'environment': env_info.environment_type.value,
                'hardware_capability': env_info.hardware_capability.value,
                'optimization_level': env_info.optimization_level.value,
                'cpu_cores': env_info.cpu_cores,
                'memory_gb': env_info.memory_gb,
                'gpu_count': env_info.gpu_count
            }
        )
    
    def get_environment_report(self) -> Dict[str, Any]:
        """สร้างรายงานสภาพแวดล้อมโดยละเอียด"""
        try:
            env_info = self.detect_environment()
            allocation = self.get_optimal_resource_allocation(env_info)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'environment_info': {
                    'type': env_info.environment_type.value,
                    'hardware_capability': env_info.hardware_capability.value,
                    'optimization_level': env_info.optimization_level.value,
                    'operating_system': env_info.operating_system,
                    'python_version': env_info.python_version
                },
                'hardware_specifications': {
                    'cpu_cores': env_info.cpu_cores,
                    'memory_gb': env_info.memory_gb,
                    'disk_gb': env_info.disk_gb,
                    'gpu_count': env_info.gpu_count,
                    'gpu_memory_gb': env_info.gpu_memory_gb
                },
                'capabilities': env_info.capabilities,
                'restrictions': env_info.restrictions,
                'recommendations': env_info.recommendations,
                'optimal_allocation': {
                    'cpu_percentage': allocation.cpu_percentage,
                    'memory_percentage': allocation.memory_percentage,
                    'disk_percentage': allocation.disk_percentage,
                    'gpu_percentage': allocation.gpu_percentage,
                    'target_utilization': allocation.target_utilization,
                    'safety_margin': allocation.safety_margin,
                    'emergency_reserve': allocation.emergency_reserve
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating environment report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_environment_summary_text(self) -> str:
        """สร้างสรุปสภาพแวดล้อมแบบข้อความ"""
        try:
            env_info = self.detect_environment()
            allocation = self.get_optimal_resource_allocation(env_info)
            
            lines = [
                "🧠 INTELLIGENT ENVIRONMENT ANALYSIS",
                "=" * 50,
                f"Environment: {env_info.environment_type.value}",
                f"Hardware: {env_info.hardware_capability.value}",
                f"Optimization: {env_info.optimization_level.value}",
                f"OS: {env_info.operating_system}",
                f"Python: {env_info.python_version}",
                "",
                "💻 HARDWARE SPECIFICATIONS:",
                f"  CPU Cores: {env_info.cpu_cores}",
                f"  Memory: {env_info.memory_gb:.1f} GB",
                f"  Disk: {env_info.disk_gb:.1f} GB",
                f"  GPU Count: {env_info.gpu_count}",
                f"  GPU Memory: {env_info.gpu_memory_gb:.1f} GB",
                "",
                "⚡ OPTIMAL RESOURCE ALLOCATION:",
                f"  CPU: {allocation.cpu_percentage*100:.1f}%",
                f"  Memory: {allocation.memory_percentage*100:.1f}%",
                f"  Disk: {allocation.disk_percentage*100:.1f}%",
                f"  GPU: {allocation.gpu_percentage*100:.1f}%",
                f"  Target Utilization: {allocation.target_utilization*100:.1f}%",
                f"  Safety Margin: {allocation.safety_margin*100:.1f}%",
                "",
                "🔧 KEY CAPABILITIES:",
            ]
            
            # Add key capabilities
            key_capabilities = ['psutil', 'torch', 'cuda', 'tensorflow', 'gputil']
            for cap in key_capabilities:
                if cap in env_info.capabilities:
                    status = "✅" if env_info.capabilities[cap] else "❌"
                    lines.append(f"  {cap}: {status}")
            
            lines.append("=" * 50)
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating environment summary: {e}"


# ====================================================
# GLOBAL INSTANCE MANAGEMENT
# ====================================================

# Global instance for Intelligent Environment Detector
_intelligent_environment_detector = None

def get_intelligent_environment_detector(config: Dict[str, Any] = None) -> IntelligentEnvironmentDetector:
    """ดึง instance ของ Intelligent Environment Detector"""
    global _intelligent_environment_detector
    if _intelligent_environment_detector is None:
        _intelligent_environment_detector = IntelligentEnvironmentDetector(config)
    return _intelligent_environment_detector


# ====================================================
# MAIN FOR TESTING
# ====================================================

def main():
    """ฟังก์ชั่นหลักสำหรับทดสอบ"""
    print("🧠 NICEGOLD ENTERPRISE - INTELLIGENT ENVIRONMENT DETECTOR")
    print("=" * 70)
    
    # Initialize detector
    detector = get_intelligent_environment_detector()
    
    # Detect environment
    env_info = detector.detect_environment()
    
    # Get optimal allocation
    allocation = detector.get_optimal_resource_allocation(env_info)
    
    # Display summary
    print(detector.get_environment_summary_text())
    
    # Generate detailed report
    report = detector.get_environment_report()
    print("\n📊 DETAILED REPORT:")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
