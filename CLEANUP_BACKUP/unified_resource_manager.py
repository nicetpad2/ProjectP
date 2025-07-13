#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - UNIFIED RESOURCE MANAGER
ระบบจัดการทรัพยากรรวม (Unified Resource Manager) สำหรับ NICEGOLD Enterprise ProjectP

🎯 Enterprise Features:
✅ ระบบจัดการทรัพยากรเดียวที่ควบคุมทั้งระบบ
✅ การตรวจจับทรัพยากรระบบอัตโนมัติ (CPU, RAM, GPU, Disk)
✅ การจัดสรรทรัพยากรอัตโนมัติที่ 80% (Enterprise Standard)
✅ การควบคุมการใช้ทรัพยากรแบบไดนามิก
✅ การจัดการ GPU และ CUDA อัตโนมัติ
✅ ระบบป้องกันและกู้คืนทรัพยากรฉุกเฉิน
✅ การติดตามและบันทึกการใช้ทรัพยากร
✅ รองรับหลายแพลตฟอร์ม (Windows, Linux, macOS)

เวอร์ชัน: 1.0 Enterprise Edition
วันที่: 9 กรกฎาคม 2025
สถานะ: Production Ready
"""

import os
import sys
import threading
import time
import platform
import logging
import warnings
import json
import gc
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# System resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import unified logger
from core.unified_enterprise_logger import get_unified_logger

# GPU Detection
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

# CUDA Detection - PyTorch
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


# ====================================================
# ENUMERATIONS AND DATA STRUCTURES
# ====================================================

class ResourceType(Enum):
    """ประเภทของทรัพยากรระบบ"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


class ResourceStatus(Enum):
    """สถานะของทรัพยากร"""
    HEALTHY = "healthy"          # ทรัพยากรปกติ (<50%)
    MODERATE = "moderate"        # ทรัพยากรปานกลาง (50-70%)
    WARNING = "warning"          # การเตือน (70-85%)
    CRITICAL = "critical"        # วิกฤต (85-95%)
    EMERGENCY = "emergency"      # ฉุกเฉิน (>95%)


@dataclass
class ResourceInfo:
    """ข้อมูลทรัพยากรระบบ"""
    resource_type: ResourceType
    total: float               # จำนวนทรัพยากรทั้งหมด
    used: float                # จำนวนที่ใช้ไป
    available: float           # จำนวนที่ยังใช้ได้
    percentage: float          # เปอร์เซ็นต์การใช้งาน (0-100)
    status: ResourceStatus     # สถานะทรัพยากร
    details: Dict[str, Any] = field(default_factory=dict)  # รายละเอียดเพิ่มเติม
    timestamp: datetime = field(default_factory=datetime.now)  # เวลาที่ตรวจสอบ

    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dict สำหรับ serialization"""
        result = asdict(self)
        result['resource_type'] = self.resource_type.value
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class AllocationResult:
    """ผลลัพธ์ของการจัดสรรทรัพยากร"""
    success: bool              # สำเร็จหรือไม่
    allocated_percentage: float  # เปอร์เซ็นต์ที่จัดสรร (0-1)
    safety_margin: float       # Safety margin (0-1)
    emergency_reserve: float   # สำรองฉุกเฉิน (0-1)
    details: Dict[str, Any] = field(default_factory=dict)  # รายละเอียดเพิ่มเติม
    timestamp: datetime = field(default_factory=datetime.now)  # เวลาที่จัดสรร


@dataclass
class OptimizationResult:
    """ผลลัพธ์ของการปรับแต่งทรัพยากร"""
    success: bool              # สำเร็จหรือไม่
    optimizations: int         # จำนวนการปรับแต่งที่ทำ
    improvements: List[Dict[str, Any]] = field(default_factory=list)  # การปรับปรุง
    details: Dict[str, Any] = field(default_factory=dict)  # รายละเอียดเพิ่มเติม
    timestamp: datetime = field(default_factory=datetime.now)  # เวลาที่ปรับแต่ง


@dataclass
class SystemEnvironment:
    """ข้อมูลสภาพแวดล้อมระบบ"""
    os_info: Dict[str, str]     # ข้อมูล OS
    python_info: Dict[str, str]  # ข้อมูล Python
    hardware_info: Dict[str, Any]  # ข้อมูลฮาร์ดแวร์
    capabilities: Dict[str, bool]  # ความสามารถของระบบ
    timestamp: datetime = field(default_factory=datetime.now)


# ====================================================
# UNIFIED RESOURCE MANAGER
# ====================================================

class UnifiedResourceManager:
    """
    🏢 ระบบจัดการทรัพยากรรวม (Unified Resource Manager)
    ระบบจัดการทรัพยากรเดียวที่ควบคุมทั้งระบบ NICEGOLD Enterprise ProjectP
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """เริ่มต้น Unified Resource Manager"""
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # Enterprise Resource Settings
        self.target_utilization = self.config.get('target_utilization', 0.80)  # 80% เป้าหมาย
        self.safety_margin = self.config.get('safety_margin', 0.15)  # 15% safety margin
        self.emergency_reserve = self.config.get('emergency_reserve', 0.05)  # 5% สำรองฉุกเฉิน
        
        # Resource Thresholds
        self.thresholds = {
            'healthy_threshold': 0.50,
            'moderate_threshold': 0.70,
            'warning_threshold': 0.85,
            'critical_threshold': 0.95
        }
        
        # Monitoring Settings
        self.monitoring_interval = self.config.get('monitoring_interval', 5.0)  # 5 วินาที
        self.history_limit = self.config.get('history_limit', 1000)  # จำกัดประวัติ
        
        # State
        self.is_monitoring = False
        self.monitoring_thread = None
        self.resource_history = []
        self.allocation_history = []
        self.optimization_history = []
        self.environment = self._detect_environment()
        
        # GPU Management
        self.gpu_info = self._detect_gpu_capabilities()
        
        self.logger.info(f"🏢 Unified Resource Manager initialized (Target: {self.target_utilization*100:.0f}% utilization)")
    
    def _setup_logger(self) -> logging.Logger:
        """ติดตั้ง Logger สำหรับ Resource Manager"""
        logger = get_unified_logger()
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - 🏢 [%(name)s] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _detect_environment(self) -> SystemEnvironment:
        """ตรวจสอบสภาพแวดล้อมระบบ"""
        try:
            # OS Information
            os_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'platform': platform.platform(),
                'machine': platform.machine(),
                'processor': platform.processor()
            }
            
            # Python Information
            python_info = {
                'version': platform.python_version(),
                'implementation': platform.python_implementation(),
                'compiler': platform.python_compiler(),
                'build': platform.python_build()
            }
            
            # Hardware Information
            hardware_info = {}
            
            if PSUTIL_AVAILABLE:
                hardware_info['cpu_count_physical'] = psutil.cpu_count(logical=False)
                hardware_info['cpu_count_logical'] = psutil.cpu_count(logical=True)
                hardware_info['memory_total'] = psutil.virtual_memory().total
                
                # Safe disk info extraction
                disk_info = {}
                try:
                    for disk in psutil.disk_partitions():
                        if disk.fstype:
                            try:
                                usage = psutil.disk_usage(disk.mountpoint)
                                disk_info[disk.mountpoint] = usage.total
                            except (PermissionError, OSError):
                                # Skip disks that can't be accessed
                                continue
                except Exception as e:
                    self.logger.debug(f"Disk enumeration error: {e}")
                
                hardware_info['disk_total'] = disk_info
            
            # System Capabilities
            capabilities = {
                'psutil_available': PSUTIL_AVAILABLE,
                'torch_available': TORCH_AVAILABLE,
                'cuda_available': CUDA_AVAILABLE,
                'tensorflow_available': TENSORFLOW_AVAILABLE,
                'tensorflow_gpu_available': TENSORFLOW_GPU_AVAILABLE,
                'gputil_available': GPUTIL_AVAILABLE
            }
            
            return SystemEnvironment(
                os_info=os_info,
                python_info=python_info,
                hardware_info=hardware_info,
                capabilities=capabilities
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting environment: {e}")
            return SystemEnvironment(
                os_info={},
                python_info={},
                hardware_info={},
                capabilities={}
            )
    
    def _detect_gpu_capabilities(self) -> Dict[str, Any]:
        """ตรวจสอบความสามารถของ GPU"""
        gpu_info = {
            'gpu_count': 0,
            'gpus': [],
            'cuda_available': CUDA_AVAILABLE,
            'tensorflow_gpu_available': TENSORFLOW_GPU_AVAILABLE
        }
        
        try:
            # Check for CUDA capability
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['gpu_count'] = torch.cuda.device_count()
                
                for i in range(torch.cuda.device_count()):
                    gpu_data = {
                        'index': i,
                        'name': torch.cuda.get_device_name(i),
                        'capability': '.'.join(map(str, torch.cuda.get_device_capability(i)))
                    }
                    gpu_info['gpus'].append(gpu_data)
                
                self.logger.info(f"✅ CUDA available: version {torch.version.cuda}, {gpu_info['gpu_count']} device(s)")
            
            # Check GPUtil for additional information
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus and len(gpus) > 0:
                        gpu_info['gpu_count'] = len(gpus)
                        gpu_info['gpus'] = []
                        
                        for i, gpu in enumerate(gpus):
                            gpu_data = {
                                'index': i,
                                'name': gpu.name,
                                'total_memory_mb': gpu.memoryTotal,
                                'free_memory_mb': gpu.memoryFree,
                                'used_memory_mb': gpu.memoryUsed,
                                'temperature': gpu.temperature,
                                'load': gpu.load * 100.0
                            }
                            gpu_info['gpus'].append(gpu_data)
                        
                        self.logger.info(f"✅ GPUtil detected {len(gpus)} GPU(s)")
                except Exception as e:
                    self.logger.warning(f"⚠️ GPUtil error: {e}")
            
            # Check TensorFlow GPU
            if TENSORFLOW_AVAILABLE and TENSORFLOW_GPU_AVAILABLE:
                physical_devices = tf.config.list_physical_devices('GPU')
                gpu_info['tensorflow_gpu_count'] = len(physical_devices)
                
                if physical_devices:
                    self.logger.info(f"✅ TensorFlow detected {len(physical_devices)} GPU(s)")
                    
                    # Configure memory growth for better resource management
                    for device in physical_devices:
                        try:
                            tf.config.experimental.set_memory_growth(device, True)
                        except:
                            pass
            
            return gpu_info
            
        except Exception as e:
            self.logger.error(f"❌ Error detecting GPU capabilities: {e}")
            return gpu_info
    
    def start_monitoring(self) -> bool:
        """เริ่มการตรวจสอบทรัพยากรต่อเนื่อง"""
        try:
            if not self.is_monitoring:
                self.is_monitoring = True
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name="ResourceMonitoring",
                    daemon=True
                )
                self.monitoring_thread.start()
                self.logger.info("🔍 Resource monitoring started")
                return True
            return True  # Already monitoring
        except Exception as e:
            self.logger.error(f"❌ Error starting monitoring: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """หยุดการตรวจสอบทรัพยากร"""
        try:
            if self.is_monitoring:
                self.is_monitoring = False
                if self.monitoring_thread:
                    self.monitoring_thread.join(timeout=2.0)
                self.logger.info("🔍 Resource monitoring stopped")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error stopping monitoring: {e}")
            return False
    
    def _monitoring_loop(self) -> None:
        """ลูปการตรวจสอบทรัพยากรต่อเนื่อง"""
        while self.is_monitoring:
            try:
                # Collect resource information
                resources = self.get_resource_status()
                
                # Check for critical conditions
                self._check_critical_conditions(resources)
                
                # Perform periodic optimization if needed
                if self._needs_optimization(resources):
                    self.optimize_resources()
                
                # Update history with limited size
                self.resource_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'resources': {k: v.to_dict() for k, v in resources.items()}
                })
                
                # Keep history within limits
                if len(self.resource_history) > self.history_limit:
                    self.resource_history = self.resource_history[-self.history_limit:]
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_critical_conditions(self, resources: Dict[str, ResourceInfo]) -> None:
        """ตรวจสอบและจัดการกับทรัพยากรที่อยู่ในสถานะวิกฤต"""
        critical_resources = [r for r in resources.values() 
                            if r.status in (ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY)]
        
        if critical_resources:
            for resource in critical_resources:
                if resource.status == ResourceStatus.EMERGENCY:
                    self._handle_emergency_resource(resource)
                elif resource.status == ResourceStatus.CRITICAL:
                    self._handle_critical_resource(resource)
    
    def _handle_emergency_resource(self, resource: ResourceInfo) -> None:
        """จัดการกับทรัพยากรที่อยู่ในสถานะฉุกเฉิน"""
        self.logger.warning(f"🚨 EMERGENCY: {resource.resource_type.value} at {resource.percentage:.1f}%")
        
        if resource.resource_type == ResourceType.MEMORY:
            # Force garbage collection
            self.logger.info("🧹 Emergency memory cleanup - forcing garbage collection")
            gc.collect()
            
        elif resource.resource_type == ResourceType.CPU:
            # Recommend throttling CPU-intensive operations
            self.logger.info("⚡ CPU emergency - recommending process throttling")
            
        elif resource.resource_type == ResourceType.GPU:
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                # Clear CUDA cache
                self.logger.info("🧹 GPU emergency - clearing CUDA cache")
                torch.cuda.empty_cache()
    
    def _handle_critical_resource(self, resource: ResourceInfo) -> None:
        """จัดการกับทรัพยากรที่อยู่ในสถานะวิกฤต"""
        self.logger.warning(f"⚠️ CRITICAL: {resource.resource_type.value} at {resource.percentage:.1f}%")
        
        if resource.resource_type == ResourceType.MEMORY:
            # Suggest garbage collection
            gc.collect()
    
    def _needs_optimization(self, resources: Dict[str, ResourceInfo]) -> bool:
        """ตรวจสอบว่าต้องทำการปรับแต่งทรัพยากรหรือไม่"""
        # Check if any resource is above warning threshold
        for resource in resources.values():
            if resource.status in (ResourceStatus.WARNING, ResourceStatus.CRITICAL):
                return True
                
        # Check if we haven't optimized in a while
        if self.optimization_history:
            last_opt_time = datetime.fromisoformat(self.optimization_history[-1]['timestamp']) \
                            if isinstance(self.optimization_history[-1]['timestamp'], str) \
                            else self.optimization_history[-1]['timestamp']
            time_since_last = (datetime.now() - last_opt_time).total_seconds()
            # Optimize every 5 minutes regardless of status
            if time_since_last > 300:
                return True
                
        return False
    
    def get_resource_status(self) -> Dict[str, ResourceInfo]:
        """ดึงข้อมูลสถานะทรัพยากรปัจจุบัน"""
        resources = {}
        
        if PSUTIL_AVAILABLE:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count(logical=True)
            cpu_status = self._determine_resource_status(cpu_percent)
            
            resources[ResourceType.CPU.value] = ResourceInfo(
                resource_type=ResourceType.CPU,
                total=100.0 * cpu_count,  # 100% per core
                used=cpu_percent * cpu_count / 100.0 * 100.0,  # Convert to overall percentage
                available=(100.0 - cpu_percent) * cpu_count / 100.0 * 100.0,
                percentage=cpu_percent,
                status=cpu_status,
                details={
                    'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': cpu_count,
                    'per_core': psutil.cpu_percent(interval=0.1, percpu=True)
                }
            )
            
            # Memory
            memory = psutil.virtual_memory()
            memory_status = self._determine_resource_status(memory.percent)
            
            resources[ResourceType.MEMORY.value] = ResourceInfo(
                resource_type=ResourceType.MEMORY,
                total=memory.total,
                used=memory.used,
                available=memory.available,
                percentage=memory.percent,
                status=memory_status,
                details={
                    'total_gb': memory.total / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3)
                }
            )
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_status = self._determine_resource_status(disk.percent)
            
            resources[ResourceType.DISK.value] = ResourceInfo(
                resource_type=ResourceType.DISK,
                total=disk.total,
                used=disk.used,
                available=disk.free,
                percentage=disk.percent,
                status=disk_status,
                details={
                    'total_gb': disk.total / (1024**3),
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'mountpoint': '/'
                }
            )
            
            # GPU (if available)
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_percent = gpu.memoryUsed / gpu.memoryTotal * 100.0 if gpu.memoryTotal > 0 else 0
                        gpu_status = self._determine_resource_status(gpu_percent)
                        
                        resources[ResourceType.GPU.value] = ResourceInfo(
                            resource_type=ResourceType.GPU,
                            total=gpu.memoryTotal,
                            used=gpu.memoryUsed,
                            available=gpu.memoryFree,
                            percentage=gpu_percent,
                            status=gpu_status,
                            details={
                                'name': gpu.name,
                                'total_mb': gpu.memoryTotal,
                                'used_mb': gpu.memoryUsed,
                                'free_mb': gpu.memoryFree,
                                'temperature': gpu.temperature,
                                'load': gpu.load * 100.0
                            }
                        )
                except Exception as e:
                    # GPU info not critical, just log and continue
                    self.logger.debug(f"⚠️ GPU info collection error: {e}")
        
        return resources
    
    def _determine_resource_status(self, percentage: float) -> ResourceStatus:
        """กำหนดสถานะของทรัพยากรตามเปอร์เซ็นต์การใช้งาน"""
        if percentage >= self.thresholds['critical_threshold'] * 100:
            return ResourceStatus.EMERGENCY
        elif percentage >= self.thresholds['warning_threshold'] * 100:
            return ResourceStatus.CRITICAL
        elif percentage >= self.thresholds['moderate_threshold'] * 100:
            return ResourceStatus.WARNING
        elif percentage >= self.thresholds['healthy_threshold'] * 100:
            return ResourceStatus.MODERATE
        else:
            return ResourceStatus.HEALTHY
    
    def allocate_resources(self, requirements: Dict[str, float]) -> AllocationResult:
        """จัดสรรทรัพยากรตามเป้าหมาย 80%"""
        try:
            current_resources = self.get_resource_status()
            allocation_result = {
                'success': True,
                'allocated_percentage': 0.0,
                'allocations': {},
                'messages': []
            }
            
            for resource_type, amount in requirements.items():
                if resource_type not in current_resources:
                    allocation_result['success'] = False
                    allocation_result['messages'].append(
                        f"⚠️ Resource type '{resource_type}' not available"
                    )
                    continue
                
                resource = current_resources[resource_type]
                
                # Calculate available within target
                target_max = resource.total * self.target_utilization
                currently_used = resource.used
                available_within_target = target_max - currently_used
                
                if available_within_target <= 0:
                    # Already at or above target utilization
                    allocation_result['success'] = False
                    allocation_result['messages'].append(
                        f"⚠️ Resource '{resource_type}' already at target utilization"
                    )
                    continue
                
                # Check if requested amount fits within target
                if amount <= available_within_target:
                    # Can allocate the full amount
                    allocation_percentage = amount / available_within_target
                    allocation_result['allocations'][resource_type] = {
                        'requested': amount,
                        'allocated': amount,
                        'percentage_of_available': allocation_percentage * 100
                    }
                    allocation_result['messages'].append(
                        f"✅ Allocated {amount:.1f} units of '{resource_type}'"
                    )
                else:
                    # Can only allocate partial amount
                    allocation_result['success'] = False
                    allocation_result['allocations'][resource_type] = {
                        'requested': amount,
                        'allocated': available_within_target,
                        'percentage_of_available': 100.0
                    }
                    allocation_result['messages'].append(
                        f"⚠️ Requested {amount:.1f} units of '{resource_type}', "
                        f"but only {available_within_target:.1f} available within 80% target"
                    )
            
            # Calculate overall allocation percentage
            if allocation_result['allocations']:
                total_requested = sum(alloc['requested'] for alloc in allocation_result['allocations'].values())
                total_allocated = sum(alloc['allocated'] for alloc in allocation_result['allocations'].values())
                
                if total_requested > 0:
                    allocation_result['allocated_percentage'] = total_allocated / total_requested
            
            # Record allocation history
            self.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': allocation_result
            })
            
            # Keep history within limits
            if len(self.allocation_history) > self.history_limit:
                self.allocation_history = self.allocation_history[-self.history_limit:]
            
            return AllocationResult(
                success=allocation_result['success'],
                allocated_percentage=allocation_result['allocated_percentage'],
                safety_margin=self.safety_margin,
                emergency_reserve=self.emergency_reserve,
                details=allocation_result
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error allocating resources: {e}")
            return AllocationResult(
                success=False,
                allocated_percentage=0.0,
                safety_margin=self.safety_margin,
                emergency_reserve=self.emergency_reserve,
                details={'error': str(e)}
            )
    
    def optimize_resources(self) -> OptimizationResult:
        """ปรับแต่งทรัพยากรให้มีประสิทธิภาพ"""
        try:
            current_resources = self.get_resource_status()
            optimization_count = 0
            improvements = []
            
            # Check Memory Optimization
            if ResourceType.MEMORY.value in current_resources:
                memory = current_resources[ResourceType.MEMORY.value]
                if memory.status in (ResourceStatus.WARNING, ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY):
                    # Force garbage collection
                    gc.collect()
                    
                    # Check memory again
                    new_memory = self.get_resource_status()[ResourceType.MEMORY.value]
                    if new_memory.percentage < memory.percentage:
                        improvement = {
                            'resource': ResourceType.MEMORY.value,
                            'action': 'garbage_collection',
                            'before': memory.percentage,
                            'after': new_memory.percentage,
                            'improvement': memory.percentage - new_memory.percentage
                        }
                        improvements.append(improvement)
                        optimization_count += 1
            
            # GPU Memory Optimization
            if TORCH_AVAILABLE and CUDA_AVAILABLE and ResourceType.GPU.value in current_resources:
                gpu = current_resources[ResourceType.GPU.value]
                if gpu.status in (ResourceStatus.WARNING, ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY):
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                    
                    # Check GPU again
                    if GPUTIL_AVAILABLE:
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                new_gpu_percent = gpus[0].memoryUsed / gpus[0].memoryTotal * 100.0
                                if new_gpu_percent < gpu.percentage:
                                    improvement = {
                                        'resource': ResourceType.GPU.value,
                                        'action': 'cuda_cache_clear',
                                        'before': gpu.percentage,
                                        'after': new_gpu_percent,
                                        'improvement': gpu.percentage - new_gpu_percent
                                    }
                                    improvements.append(improvement)
                                    optimization_count += 1
                        except:
                            pass
            
            # Record optimization history
            result = {
                'timestamp': datetime.now().isoformat(),
                'optimization_count': optimization_count,
                'improvements': improvements
            }
            
            self.optimization_history.append(result)
            
            # Keep history within limits
            if len(self.optimization_history) > self.history_limit:
                self.optimization_history = self.optimization_history[-self.history_limit:]
            
            return OptimizationResult(
                success=optimization_count > 0,
                optimizations=optimization_count,
                improvements=improvements
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing resources: {e}")
            return OptimizationResult(
                success=False,
                optimizations=0,
                details={'error': str(e)}
            )
    
    def configure_gpu(self, memory_fraction: float = 0.8) -> bool:
        """ตั้งค่า GPU ให้ใช้ทรัพยากรตามที่กำหนด"""
        if not CUDA_AVAILABLE:
            self.logger.warning("⚠️ CUDA not available for GPU configuration")
            return False
        
        try:
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                # Clear cache first
                torch.cuda.empty_cache()
                
                # Configure for memory fraction
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.set_per_process_memory_fraction(memory_fraction, i)
                    
                    self.logger.info(f"✅ Set GPU memory fraction to {memory_fraction*100:.0f}%")
                    return True
            
            if TENSORFLOW_AVAILABLE and TENSORFLOW_GPU_AVAILABLE:
                try:
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                        
                        self.logger.info("✅ Configured TensorFlow GPU memory growth")
                        return True
                except:
                    pass
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error configuring GPU: {e}")
            return False
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """สรุปข้อมูลทรัพยากรระบบสำหรับแสดงผล"""
        try:
            current_resources = self.get_resource_status()
            
            summary = {
                'resource_status': {
                    resource_type: {
                        'percentage': resource.percentage,
                        'status': resource.status.value,
                        'usage': f"{resource.used:.1f} / {resource.total:.1f}"
                    }
                    for resource_type, resource in current_resources.items()
                },
                'target_utilization': self.target_utilization * 100,
                'safety_margin': self.safety_margin * 100,
                'emergency_reserve': self.emergency_reserve * 100,
                'monitoring_active': self.is_monitoring,
                'history_points': len(self.resource_history),
                'allocations': len(self.allocation_history),
                'optimizations': len(self.optimization_history),
                'environment': {
                    'os': self.environment.os_info.get('system', 'Unknown'),
                    'python': self.environment.python_info.get('version', 'Unknown'),
                    'cpu_cores': self.environment.hardware_info.get('cpu_count_logical', 0)
                },
                'capabilities': {
                    'cuda': CUDA_AVAILABLE,
                    'gpu': len(self.gpu_info.get('gpus', [])) > 0,
                    'tensorflow_gpu': TENSORFLOW_GPU_AVAILABLE
                }
            }
            
            # Add GPU information if available
            if self.gpu_info['gpu_count'] > 0:
                summary['gpu_info'] = {
                    'count': self.gpu_info['gpu_count'],
                    'names': [gpu.get('name', 'Unknown') for gpu in self.gpu_info['gpus']],
                    'cuda_version': self.gpu_info.get('cuda_version', 'Unknown')
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error getting resource summary: {e}")
            return {
                'error': str(e),
                'target_utilization': self.target_utilization * 100
            }
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """สร้างรายงานการใช้ทรัพยากรโดยละเอียด"""
        try:
            summary = self.get_resource_summary()
            current_resources = self.get_resource_status()
            
            # Get last 5 entries from histories
            recent_allocations = self.allocation_history[-5:] if self.allocation_history else []
            recent_optimizations = self.optimization_history[-5:] if self.optimization_history else []
            recent_resources = self.resource_history[-5:] if self.resource_history else []
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'current_resources': {k: v.to_dict() for k, v in current_resources.items()},
                'recent_allocations': recent_allocations,
                'recent_optimizations': recent_optimizations,
                'recent_resources': recent_resources,
                'environment': {
                    'os': self.environment.os_info,
                    'python': self.environment.python_info,
                    'hardware': self.environment.hardware_info,
                    'capabilities': self.environment.capabilities
                },
                'gpu_info': self.gpu_info
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Error generating detailed report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_resource_usage_percentages(self) -> Dict[str, float]:
        """ดึงค่าเปอร์เซ็นต์การใช้ทรัพยากรหลัก"""
        try:
            resources = self.get_resource_status()
            return {
                k: v.percentage for k, v in resources.items()
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting resource percentages: {e}")
            return {}

    def get_resource_utilization_text(self) -> str:
        """สร้างข้อความแสดงการใช้ทรัพยากร"""
        try:
            resources = self.get_resource_status()
            lines = [
                "🏢 RESOURCE UTILIZATION REPORT",
                "=" * 40
            ]
            
            # CPU
            if ResourceType.CPU.value in resources:
                cpu = resources[ResourceType.CPU.value]
                lines.append(f"CPU: {cpu.percentage:.1f}% ({cpu.status.value})")
                lines.append(f"  └─ Cores: {cpu.details.get('logical_cores', 'N/A')} logical, {cpu.details.get('physical_cores', 'N/A')} physical")
            
            # Memory
            if ResourceType.MEMORY.value in resources:
                memory = resources[ResourceType.MEMORY.value]
                lines.append(f"Memory: {memory.percentage:.1f}% ({memory.status.value})")
                lines.append(f"  └─ {memory.details.get('used_gb', 0):.1f} GB / {memory.details.get('total_gb', 0):.1f} GB")
            
            # Disk
            if ResourceType.DISK.value in resources:
                disk = resources[ResourceType.DISK.value]
                lines.append(f"Disk: {disk.percentage:.1f}% ({disk.status.value})")
                lines.append(f"  └─ {disk.details.get('used_gb', 0):.1f} GB / {disk.details.get('total_gb', 0):.1f} GB")
            
            # GPU
            if ResourceType.GPU.value in resources:
                gpu = resources[ResourceType.GPU.value]
                lines.append(f"GPU: {gpu.percentage:.1f}% ({gpu.status.value})")
                lines.append(f"  └─ {gpu.details.get('name', 'Unknown')}: {gpu.details.get('used_mb', 0):.0f} MB / {gpu.details.get('total_mb', 0):.0f} MB")
            
            lines.append("=" * 40)
            lines.append(f"Target Utilization: {self.target_utilization*100:.0f}%")
            lines.append(f"Safety Margin: {self.safety_margin*100:.0f}%")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating resource text: {e}"


# ====================================================
# GLOBAL INSTANCE MANAGEMENT
# ====================================================

# Global instance for Unified Resource Manager
_unified_resource_manager = None

def get_unified_resource_manager(config: Dict[str, Any] = None) -> UnifiedResourceManager:
    """ดึง instance ของ Unified Resource Manager"""
    global _unified_resource_manager
    if _unified_resource_manager is None:
        _unified_resource_manager = UnifiedResourceManager(config)
    return _unified_resource_manager


# ====================================================
# MAIN FOR TESTING
# ====================================================

def main():
    """ฟังก์ชั่นหลักสำหรับทดสอบ"""
    print("🏢 NICEGOLD ENTERPRISE - UNIFIED RESOURCE MANAGER")
    print("=" * 60)
    
    # Initialize manager
    manager = get_unified_resource_manager()
    
    # Start monitoring
    manager.start_monitoring()
    
    try:
        # Get initial resource status
        print("\n📊 RESOURCE STATUS:")
        resources = manager.get_resource_status()
        for r_type, r_info in resources.items():
            print(f"  {r_type}: {r_info.percentage:.1f}% ({r_info.status.value})")
        
        # Configure GPU
        if CUDA_AVAILABLE:
            print("\n🎮 CONFIGURING GPU:")
            result = manager.configure_gpu(memory_fraction=0.8)
            print(f"  GPU Configuration: {'✅ Success' if result else '❌ Failed'}")
        
        # Resource summary
        print("\n📋 RESOURCE SUMMARY:")
        summary = manager.get_resource_summary()
        for k, v in summary.items():
            if not isinstance(v, dict):
                print(f"  {k}: {v}")
        
        # Wait to collect some data
        print("\n⏳ Monitoring for 10 seconds...")
        time.sleep(10)
        
        # Show resource text
        print("\n" + manager.get_resource_utilization_text())
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        # Stop monitoring
        manager.stop_monitoring()
        print("\n✅ Unified Resource Manager stopped")


if __name__ == "__main__":
    main()
