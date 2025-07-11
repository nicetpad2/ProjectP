#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 NICEGOLD ENTERPRISE PROJECTP - SMART RESOURCE ORCHESTRATOR
ระบบจัดการทรัพยากรอัจฉริยะ (Smart Resource Orchestrator)

🎯 Enterprise Features:
✅ ระบบจัดการทรัพยากรอัจฉริยะที่ผสานรวม Environment Detection
✅ การจัดสรรทรัพยากร 80% แบบอัตโนมัติตามสภาพแวดล้อม
✅ ระบบปรับแต่งประสิทธิภาพแบบไดนามิกและเรียลไทม์
✅ การจัดการ GPU, CUDA, และ AI frameworks แบบอัจฉริยะ
✅ ระบบติดตามและปรับปรุงทรัพยากรต่อเนื่อง
✅ การจัดการฉุกเฉินและกู้คืนทรัพยากรอัตโนมัติ
✅ รองรับทุกสภาพแวดล้อม (Local, Cloud, Colab, Docker)

เวอร์ชัน: 1.0 Enterprise Edition
วันที่: 9 กรกฎาคม 2025
สถานะ: Production Ready
"""

import os
import sys
import threading
import time
import logging
import warnings
import json
import gc
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

# Import environment detector
try:
    from core.intelligent_environment_detector import (
        get_intelligent_environment_detector,
        EnvironmentInfo,
        ResourceAllocation,
        EnvironmentType,
        HardwareCapability,
        ResourceOptimizationLevel
    )
    ENVIRONMENT_DETECTOR_AVAILABLE = True
except ImportError:
    ENVIRONMENT_DETECTOR_AVAILABLE = False

# Import unified resource manager
try:
    from core.unified_resource_manager import (
        get_unified_resource_manager,
        ResourceInfo,
        ResourceType,
        ResourceStatus,
        AllocationResult,
        OptimizationResult
    )
    UNIFIED_RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_RESOURCE_MANAGER_AVAILABLE = False

# System resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# GPU Detection
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
except ImportError:
    TENSORFLOW_AVAILABLE = False


# ====================================================
# ENUMERATIONS AND DATA STRUCTURES
# ====================================================

class OrchestrationStatus(Enum):
    """สถานะของการจัดการทรัพยากร"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    OPTIMIZING = "optimizing"
    MONITORING = "monitoring"
    EMERGENCY = "emergency"
    STOPPED = "stopped"


class AdaptiveMode(Enum):
    """โหมดการปรับตัวของระบบ"""
    LEARNING = "learning"      # กำลังเรียนรู้สภาพแวดล้อม
    ADAPTING = "adapting"      # กำลังปรับตัว
    OPTIMIZED = "optimized"    # ปรับแต่งแล้ว
    MONITORING = "monitoring"  # ติดตามอย่างเดียว


@dataclass
class OrchestrationConfig:
    """การตั้งค่าการจัดการทรัพยากร"""
    target_utilization: float = 0.80
    safety_margin: float = 0.15
    emergency_reserve: float = 0.05
    monitoring_interval: float = 5.0
    optimization_interval: float = 30.0
    adaptation_threshold: float = 0.10
    learning_period: int = 100
    enable_gpu_management: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_predictive_scaling: bool = True


@dataclass
class OrchestrationResult:
    """ผลลัพธ์การจัดการทรัพยากร"""
    success: bool
    status: OrchestrationStatus
    mode: AdaptiveMode
    resource_allocation: Dict[str, float]
    optimizations_applied: int
    improvements: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


# ====================================================
# SMART RESOURCE ORCHESTRATOR
# ====================================================

class SmartResourceOrchestrator:
    """
    🤖 ระบบจัดการทรัพยากรอัจฉริยะ (Smart Resource Orchestrator)
    ผสานรวมการตรวจจับสภาพแวดล้อมและการจัดการทรัพยากรแบบอัตโนมัติ
    """
    
    def __init__(self, config: OrchestrationConfig = None):
        """เริ่มต้น Smart Resource Orchestrator"""
        self.config = config or OrchestrationConfig()
        self.logger = self._setup_logger()
        
        # Initialize components
        self.environment_detector = None
        self.resource_manager = None
        self.environment_info = None
        self.optimal_allocation = None
        
        # State management
        self.status = OrchestrationStatus.INITIALIZING
        self.mode = AdaptiveMode.LEARNING
        self.is_running = False
        self.orchestration_thread = None
        
        # Learning and adaptation
        self.learning_data = []
        self.adaptation_history = []
        self.performance_metrics = {}
        self.optimization_count = 0
        
        # Performance tracking
        self.resource_usage_history = []
        self.allocation_history = []
        self.optimization_history = []
        
        # Initialize system
        self._initialize_system()
        
        self.logger.info("🤖 Smart Resource Orchestrator initialized")
    
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
                    '%(asctime)s - 🤖 [%(name)s] - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)
        
        return logger
    
    def _initialize_system(self) -> bool:
        """เริ่มต้นระบบและส่วนประกอบ"""
        try:
            # Initialize environment detector
            if ENVIRONMENT_DETECTOR_AVAILABLE:
                self.environment_detector = get_intelligent_environment_detector()
                self.environment_info = self.environment_detector.detect_environment()
                self.optimal_allocation = self.environment_detector.get_optimal_resource_allocation(
                    self.environment_info
                )
                self.logger.info(f"✅ Environment detected: {self.environment_info.environment_type.value}")
                self.logger.info(f"✅ Hardware capability: {self.environment_info.hardware_capability.value}")
            else:
                self.logger.warning("⚠️ Environment detector not available")
                return False
            
            # Initialize resource manager
            if UNIFIED_RESOURCE_MANAGER_AVAILABLE:
                # Configure resource manager with detected environment
                resource_config = {
                    'target_utilization': self.optimal_allocation.target_utilization,
                    'safety_margin': self.optimal_allocation.safety_margin,
                    'emergency_reserve': self.optimal_allocation.emergency_reserve,
                    'monitoring_interval': self.config.monitoring_interval
                }
                
                self.resource_manager = get_unified_resource_manager(resource_config)
                self.logger.info("✅ Resource manager initialized with environment-specific settings")
            else:
                self.logger.warning("⚠️ Unified resource manager not available")
                return False
            
            # Configure GPU if available
            if self.config.enable_gpu_management and CUDA_AVAILABLE:
                self._configure_gpu_resources()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing system: {e}")
            return False
    
    def _configure_gpu_resources(self) -> bool:
        """ตั้งค่าทรัพยากร GPU"""
        try:
            if self.resource_manager:
                # Configure GPU with optimal allocation
                gpu_fraction = self.optimal_allocation.gpu_percentage
                success = self.resource_manager.configure_gpu(gpu_fraction)
                
                if success:
                    self.logger.info(f"✅ GPU configured with {gpu_fraction*100:.1f}% allocation")
                    return True
                else:
                    self.logger.warning("⚠️ GPU configuration failed")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error configuring GPU: {e}")
            return False
    
    def start_orchestration(self) -> bool:
        """เริ่มการจัดการทรัพยากรอัจฉริยะ"""
        try:
            if self.is_running:
                self.logger.warning("⚠️ Orchestration already running")
                return True
            
            # Start resource monitoring
            if self.resource_manager:
                self.resource_manager.start_monitoring()
            
            # Start orchestration thread
            self.is_running = True
            self.status = OrchestrationStatus.ACTIVE
            self.orchestration_thread = threading.Thread(
                target=self._orchestration_loop,
                name="SmartResourceOrchestration",
                daemon=True
            )
            self.orchestration_thread.start()
            
            self.logger.info("🤖 Smart Resource Orchestration started")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error starting orchestration: {e}")
            return False
    
    def stop_orchestration(self) -> bool:
        """หยุดการจัดการทรัพยากร"""
        try:
            if not self.is_running:
                return True
            
            self.is_running = False
            self.status = OrchestrationStatus.STOPPED
            
            # Stop resource monitoring
            if self.resource_manager:
                self.resource_manager.stop_monitoring()
            
            # Wait for orchestration thread to finish
            if self.orchestration_thread:
                self.orchestration_thread.join(timeout=5.0)
            
            self.logger.info("🤖 Smart Resource Orchestration stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stopping orchestration: {e}")
            return False
    
    def _orchestration_loop(self) -> None:
        """ลูปการจัดการทรัพยากรอัจฉริยะ"""
        last_optimization_time = time.time()
        
        while self.is_running:
            try:
                # Get current resource status
                if self.resource_manager:
                    current_resources = self.resource_manager.get_resource_status()
                    
                    # Check for critical situations
                    self._handle_critical_situations(current_resources)
                    
                    # Learn from current resource usage
                    self._learn_from_usage(current_resources)
                    
                    # Adapt resource allocation if needed
                    if self._should_adapt():
                        self._adapt_resource_allocation()
                    
                    # Perform optimization if needed
                    current_time = time.time()
                    if current_time - last_optimization_time >= self.config.optimization_interval:
                        self._perform_intelligent_optimization()
                        last_optimization_time = current_time
                    
                    # Update performance metrics
                    self._update_performance_metrics(current_resources)
                
                # Sleep for monitoring interval
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in orchestration loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _handle_critical_situations(self, resources: Dict[str, ResourceInfo]) -> None:
        """จัดการกับสถานการณ์วิกฤต"""
        critical_resources = [r for r in resources.values() 
                            if r.status in (ResourceStatus.CRITICAL, ResourceStatus.EMERGENCY)]
        
        if critical_resources:
            self.status = OrchestrationStatus.EMERGENCY
            self.logger.warning(f"🚨 Critical situation detected: {len(critical_resources)} resource(s)")
            
            # Immediate optimization
            if self.resource_manager:
                result = self.resource_manager.optimize_resources()
                if result.success:
                    self.logger.info(f"✅ Emergency optimization completed: {result.optimizations} optimizations")
                    self.optimization_count += result.optimizations
            
            # Force garbage collection
            if self.config.enable_memory_optimization:
                gc.collect()
                self.logger.info("🧹 Emergency garbage collection performed")
            
            # GPU memory cleanup
            if self.config.enable_gpu_management and CUDA_AVAILABLE:
                try:
                    if TORCH_AVAILABLE:
                        torch.cuda.empty_cache()
                        self.logger.info("🧹 GPU cache cleared")
                except Exception as e:
                    self.logger.warning(f"⚠️ GPU cache clear failed: {e}")
            
            # Reset status after handling
            self.status = OrchestrationStatus.ACTIVE
        else:
            # Normal monitoring
            self.status = OrchestrationStatus.MONITORING
    
    def _learn_from_usage(self, resources: Dict[str, ResourceInfo]) -> None:
        """เรียนรู้จากการใช้ทรัพยากร"""
        # Record usage patterns
        usage_data = {
            'timestamp': datetime.now(),
            'cpu_usage': resources.get(ResourceType.CPU.value, ResourceInfo(
                ResourceType.CPU, 0, 0, 0, 0, ResourceStatus.HEALTHY
            )).percentage,
            'memory_usage': resources.get(ResourceType.MEMORY.value, ResourceInfo(
                ResourceType.MEMORY, 0, 0, 0, 0, ResourceStatus.HEALTHY
            )).percentage,
            'gpu_usage': resources.get(ResourceType.GPU.value, ResourceInfo(
                ResourceType.GPU, 0, 0, 0, 0, ResourceStatus.HEALTHY
            )).percentage if ResourceType.GPU.value in resources else 0,
            'optimization_count': self.optimization_count
        }
        
        self.learning_data.append(usage_data)
        
        # Keep learning data within limits
        if len(self.learning_data) > self.config.learning_period:
            self.learning_data = self.learning_data[-self.config.learning_period:]
        
        # Update adaptive mode
        if len(self.learning_data) >= 10:
            if self.mode == AdaptiveMode.LEARNING:
                self.mode = AdaptiveMode.ADAPTING
                self.logger.info("🧠 Switching to adaptation mode")
            elif len(self.learning_data) >= self.config.learning_period:
                if self.mode == AdaptiveMode.ADAPTING:
                    self.mode = AdaptiveMode.OPTIMIZED
                    self.logger.info("🎯 Switching to optimized mode")
    
    def _should_adapt(self) -> bool:
        """ตรวจสอบว่าควรปรับการจัดสรรทรัพยากรหรือไม่"""
        if self.mode == AdaptiveMode.LEARNING:
            return False
        
        if len(self.learning_data) < 5:
            return False
        
        # Check if resource usage patterns have changed significantly
        recent_data = self.learning_data[-5:]
        older_data = self.learning_data[-10:-5] if len(self.learning_data) >= 10 else []
        
        if not older_data:
            return False
        
        # Calculate average usage differences
        recent_avg = {
            'cpu': sum(d['cpu_usage'] for d in recent_data) / len(recent_data),
            'memory': sum(d['memory_usage'] for d in recent_data) / len(recent_data),
            'gpu': sum(d['gpu_usage'] for d in recent_data) / len(recent_data)
        }
        
        older_avg = {
            'cpu': sum(d['cpu_usage'] for d in older_data) / len(older_data),
            'memory': sum(d['memory_usage'] for d in older_data) / len(older_data),
            'gpu': sum(d['gpu_usage'] for d in older_data) / len(older_data)
        }
        
        # Check for significant changes
        threshold = self.config.adaptation_threshold * 100  # Convert to percentage
        
        for resource in ['cpu', 'memory', 'gpu']:
            if abs(recent_avg[resource] - older_avg[resource]) > threshold:
                return True
        
        return False
    
    def _adapt_resource_allocation(self) -> None:
        """ปรับการจัดสรรทรัพยากรตามการเรียนรู้"""
        try:
            self.logger.info("🔄 Adapting resource allocation...")
            
            # Analyze usage patterns
            if len(self.learning_data) >= 5:
                recent_data = self.learning_data[-5:]
                
                # Calculate average usage
                avg_usage = {
                    'cpu': sum(d['cpu_usage'] for d in recent_data) / len(recent_data),
                    'memory': sum(d['memory_usage'] for d in recent_data) / len(recent_data),
                    'gpu': sum(d['gpu_usage'] for d in recent_data) / len(recent_data)
                }
                
                # Adapt allocation based on usage patterns
                adaptations = []
                
                # CPU adaptation
                if avg_usage['cpu'] > self.optimal_allocation.target_utilization * 100 * 0.9:
                    # High CPU usage - consider increasing allocation
                    new_cpu = min(self.optimal_allocation.cpu_percentage * 1.1, 0.90)
                    if new_cpu != self.optimal_allocation.cpu_percentage:
                        adaptations.append(f"CPU allocation: {self.optimal_allocation.cpu_percentage*100:.1f}% → {new_cpu*100:.1f}%")
                        self.optimal_allocation.cpu_percentage = new_cpu
                
                elif avg_usage['cpu'] < self.optimal_allocation.target_utilization * 100 * 0.5:
                    # Low CPU usage - consider decreasing allocation
                    new_cpu = max(self.optimal_allocation.cpu_percentage * 0.9, 0.30)
                    if new_cpu != self.optimal_allocation.cpu_percentage:
                        adaptations.append(f"CPU allocation: {self.optimal_allocation.cpu_percentage*100:.1f}% → {new_cpu*100:.1f}%")
                        self.optimal_allocation.cpu_percentage = new_cpu
                
                # Memory adaptation
                if avg_usage['memory'] > self.optimal_allocation.target_utilization * 100 * 0.9:
                    new_memory = min(self.optimal_allocation.memory_percentage * 1.1, 0.90)
                    if new_memory != self.optimal_allocation.memory_percentage:
                        adaptations.append(f"Memory allocation: {self.optimal_allocation.memory_percentage*100:.1f}% → {new_memory*100:.1f}%")
                        self.optimal_allocation.memory_percentage = new_memory
                
                elif avg_usage['memory'] < self.optimal_allocation.target_utilization * 100 * 0.5:
                    new_memory = max(self.optimal_allocation.memory_percentage * 0.9, 0.30)
                    if new_memory != self.optimal_allocation.memory_percentage:
                        adaptations.append(f"Memory allocation: {self.optimal_allocation.memory_percentage*100:.1f}% → {new_memory*100:.1f}%")
                        self.optimal_allocation.memory_percentage = new_memory
                
                # GPU adaptation
                if avg_usage['gpu'] > 0 and self.config.enable_gpu_management:
                    if avg_usage['gpu'] > self.optimal_allocation.target_utilization * 100 * 0.9:
                        new_gpu = min(self.optimal_allocation.gpu_percentage * 1.1, 0.90)
                        if new_gpu != self.optimal_allocation.gpu_percentage:
                            adaptations.append(f"GPU allocation: {self.optimal_allocation.gpu_percentage*100:.1f}% → {new_gpu*100:.1f}%")
                            self.optimal_allocation.gpu_percentage = new_gpu
                            
                            # Apply GPU changes
                            if self.resource_manager:
                                self.resource_manager.configure_gpu(new_gpu)
                
                # Record adaptations
                if adaptations:
                    adaptation_record = {
                        'timestamp': datetime.now(),
                        'adaptations': adaptations,
                        'avg_usage': avg_usage
                    }
                    self.adaptation_history.append(adaptation_record)
                    
                    # Keep history within limits
                    if len(self.adaptation_history) > 100:
                        self.adaptation_history = self.adaptation_history[-100:]
                    
                    for adaptation in adaptations:
                        self.logger.info(f"🔄 {adaptation}")
                
                self.logger.info("✅ Resource allocation adaptation completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error adapting resource allocation: {e}")
    
    def _perform_intelligent_optimization(self) -> None:
        """ทำการปรับแต่งประสิทธิภาพแบบอัจฉริยะ"""
        try:
            self.status = OrchestrationStatus.OPTIMIZING
            self.logger.info("⚡ Performing intelligent optimization...")
            
            optimizations_applied = 0
            improvements = []
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                before_memory = self._get_memory_usage()
                gc.collect()
                after_memory = self._get_memory_usage()
                
                if before_memory > after_memory:
                    improvement = {
                        'type': 'memory_cleanup',
                        'before': before_memory,
                        'after': after_memory,
                        'improvement': before_memory - after_memory
                    }
                    improvements.append(improvement)
                    optimizations_applied += 1
                    self.logger.info(f"🧹 Memory optimization: {before_memory:.1f}% → {after_memory:.1f}%")
            
            # GPU optimization
            if self.config.enable_gpu_management and CUDA_AVAILABLE:
                try:
                    if TORCH_AVAILABLE:
                        torch.cuda.empty_cache()
                        optimizations_applied += 1
                        improvements.append({
                            'type': 'gpu_cache_clear',
                            'description': 'GPU cache cleared'
                        })
                        self.logger.info("🧹 GPU cache optimization completed")
                except Exception as e:
                    self.logger.warning(f"⚠️ GPU optimization failed: {e}")
            
            # Resource manager optimization
            if self.resource_manager:
                result = self.resource_manager.optimize_resources()
                if result.success:
                    optimizations_applied += result.optimizations
                    improvements.extend(result.improvements)
                    self.logger.info(f"⚡ Resource manager optimization: {result.optimizations} optimizations")
            
            # Record optimization
            self.optimization_count += optimizations_applied
            optimization_record = {
                'timestamp': datetime.now(),
                'optimizations_applied': optimizations_applied,
                'improvements': improvements
            }
            self.optimization_history.append(optimization_record)
            
            # Keep history within limits
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]
            
            self.logger.info(f"✅ Intelligent optimization completed: {optimizations_applied} optimizations applied")
            
        except Exception as e:
            self.logger.error(f"❌ Error in intelligent optimization: {e}")
        finally:
            self.status = OrchestrationStatus.ACTIVE
    
    def _get_memory_usage(self) -> float:
        """ดึงข้อมูลการใช้หน่วยความจำ"""
        if PSUTIL_AVAILABLE:
            return psutil.virtual_memory().percent
        return 0.0
    
    def _update_performance_metrics(self, resources: Dict[str, ResourceInfo]) -> None:
        """อัพเดตเมตริกประสิทธิภาพ"""
        try:
            current_metrics = {
                'timestamp': datetime.now(),
                'cpu_usage': resources.get(ResourceType.CPU.value, ResourceInfo(
                    ResourceType.CPU, 0, 0, 0, 0, ResourceStatus.HEALTHY
                )).percentage,
                'memory_usage': resources.get(ResourceType.MEMORY.value, ResourceInfo(
                    ResourceType.MEMORY, 0, 0, 0, 0, ResourceStatus.HEALTHY
                )).percentage,
                'gpu_usage': resources.get(ResourceType.GPU.value, ResourceInfo(
                    ResourceType.GPU, 0, 0, 0, 0, ResourceStatus.HEALTHY
                )).percentage if ResourceType.GPU.value in resources else 0,
                'optimization_count': self.optimization_count,
                'status': self.status.value,
                'mode': self.mode.value
            }
            
            self.resource_usage_history.append(current_metrics)
            
            # Keep history within limits
            if len(self.resource_usage_history) > 1000:
                self.resource_usage_history = self.resource_usage_history[-1000:]
            
            # Update aggregated metrics
            if len(self.resource_usage_history) >= 10:
                recent_metrics = self.resource_usage_history[-10:]
                self.performance_metrics = {
                    'avg_cpu_usage': sum(m['cpu_usage'] for m in recent_metrics) / len(recent_metrics),
                    'avg_memory_usage': sum(m['memory_usage'] for m in recent_metrics) / len(recent_metrics),
                    'avg_gpu_usage': sum(m['gpu_usage'] for m in recent_metrics) / len(recent_metrics),
                    'total_optimizations': self.optimization_count,
                    'uptime_minutes': (datetime.now() - recent_metrics[0]['timestamp']).total_seconds() / 60
                }
            
        except Exception as e:
            self.logger.error(f"❌ Error updating performance metrics: {e}")
    
    def get_orchestration_status(self) -> OrchestrationResult:
        """ดึงสถานะการจัดการทรัพยากร"""
        try:
            # Get current resource allocation
            resource_allocation = {
                'cpu_percentage': self.optimal_allocation.cpu_percentage if self.optimal_allocation else 0.80,
                'memory_percentage': self.optimal_allocation.memory_percentage if self.optimal_allocation else 0.80,
                'disk_percentage': self.optimal_allocation.disk_percentage if self.optimal_allocation else 0.50,
                'gpu_percentage': self.optimal_allocation.gpu_percentage if self.optimal_allocation else 0.80,
                'target_utilization': self.optimal_allocation.target_utilization if self.optimal_allocation else 0.80
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations()
            
            # Get recent improvements
            recent_improvements = []
            if self.optimization_history:
                recent_opt = self.optimization_history[-1]
                recent_improvements = recent_opt.get('improvements', [])
            
            return OrchestrationResult(
                success=True,
                status=self.status,
                mode=self.mode,
                resource_allocation=resource_allocation,
                optimizations_applied=self.optimization_count,
                improvements=recent_improvements,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error getting orchestration status: {e}")
            return OrchestrationResult(
                success=False,
                status=OrchestrationStatus.STOPPED,
                mode=AdaptiveMode.LEARNING,
                resource_allocation={},
                optimizations_applied=0,
                improvements=[],
                recommendations=[f"Error: {e}"]
            )
    
    def _generate_recommendations(self) -> List[str]:
        """สร้างคำแนะนำสำหรับการปรับปรุง"""
        recommendations = []
        
        try:
            # Performance-based recommendations
            if self.performance_metrics:
                avg_cpu = self.performance_metrics.get('avg_cpu_usage', 0)
                avg_memory = self.performance_metrics.get('avg_memory_usage', 0)
                avg_gpu = self.performance_metrics.get('avg_gpu_usage', 0)
                
                if avg_cpu > 85:
                    recommendations.append("Consider reducing CPU-intensive operations or adding more CPU cores")
                elif avg_cpu < 30:
                    recommendations.append("CPU utilization is low - consider increasing parallel processing")
                
                if avg_memory > 85:
                    recommendations.append("High memory usage detected - consider memory optimization")
                elif avg_memory < 30:
                    recommendations.append("Memory utilization is low - consider increasing batch sizes")
                
                if avg_gpu > 85:
                    recommendations.append("GPU utilization is high - consider optimizing GPU operations")
                elif avg_gpu > 0 and avg_gpu < 30:
                    recommendations.append("GPU utilization is low - consider increasing GPU workload")
            
            # Environment-based recommendations
            if self.environment_info:
                if self.environment_info.environment_type == EnvironmentType.GOOGLE_COLAB:
                    recommendations.append("Running in Google Colab - save progress frequently")
                elif self.environment_info.environment_type == EnvironmentType.DOCKER_CONTAINER:
                    recommendations.append("Running in Docker - monitor container resource limits")
                elif self.environment_info.hardware_capability == HardwareCapability.MINIMAL_PERFORMANCE:
                    recommendations.append("Limited hardware detected - use conservative resource allocation")
            
            # Mode-based recommendations
            if self.mode == AdaptiveMode.LEARNING:
                recommendations.append("System is learning your usage patterns - performance will improve")
            elif self.mode == AdaptiveMode.ADAPTING:
                recommendations.append("System is adapting to your workload - monitoring resource usage")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """สร้างรายงานการจัดการทรัพยากรโดยละเอียด"""
        try:
            status = self.get_orchestration_status()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'orchestration_status': {
                    'status': status.status.value,
                    'mode': status.mode.value,
                    'running': self.is_running,
                    'optimizations_applied': status.optimizations_applied
                },
                'resource_allocation': status.resource_allocation,
                'performance_metrics': self.performance_metrics,
                'environment_info': {
                    'type': self.environment_info.environment_type.value if self.environment_info else 'Unknown',
                    'hardware_capability': self.environment_info.hardware_capability.value if self.environment_info else 'Unknown',
                    'cpu_cores': self.environment_info.cpu_cores if self.environment_info else 0,
                    'memory_gb': self.environment_info.memory_gb if self.environment_info else 0,
                    'gpu_count': self.environment_info.gpu_count if self.environment_info else 0
                },
                'learning_progress': {
                    'data_points': len(self.learning_data),
                    'adaptations_made': len(self.adaptation_history),
                    'optimizations_performed': len(self.optimization_history)
                },
                'recommendations': status.recommendations,
                'recent_improvements': status.improvements[-5:] if status.improvements else []
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_status_summary_text(self) -> str:
        """สร้างสรุปสถานะแบบข้อความ"""
        try:
            status = self.get_orchestration_status()
            
            lines = [
                "🤖 SMART RESOURCE ORCHESTRATOR STATUS",
                "=" * 50,
                f"Status: {status.status.value}",
                f"Mode: {status.mode.value}",
                f"Running: {'✅ Yes' if self.is_running else '❌ No'}",
                f"Optimizations Applied: {status.optimizations_applied}",
                "",
                "⚡ CURRENT RESOURCE ALLOCATION:",
                f"  CPU: {status.resource_allocation.get('cpu_percentage', 0)*100:.1f}%",
                f"  Memory: {status.resource_allocation.get('memory_percentage', 0)*100:.1f}%",
                f"  GPU: {status.resource_allocation.get('gpu_percentage', 0)*100:.1f}%",
                f"  Target: {status.resource_allocation.get('target_utilization', 0)*100:.1f}%",
                "",
                "📊 PERFORMANCE METRICS:",
            ]
            
            if self.performance_metrics:
                lines.extend([
                    f"  Avg CPU Usage: {self.performance_metrics.get('avg_cpu_usage', 0):.1f}%",
                    f"  Avg Memory Usage: {self.performance_metrics.get('avg_memory_usage', 0):.1f}%",
                    f"  Avg GPU Usage: {self.performance_metrics.get('avg_gpu_usage', 0):.1f}%",
                    f"  Uptime: {self.performance_metrics.get('uptime_minutes', 0):.1f} minutes"
                ])
            
            if status.recommendations:
                lines.append("")
                lines.append("💡 RECOMMENDATIONS:")
                for rec in status.recommendations[:3]:  # Show top 3 recommendations
                    lines.append(f"  • {rec}")
            
            lines.append("=" * 50)
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating status summary: {e}"


# ====================================================
# GLOBAL INSTANCE MANAGEMENT
# ====================================================

# Global instance for Smart Resource Orchestrator
_smart_resource_orchestrator = None

def get_smart_resource_orchestrator(config: OrchestrationConfig = None) -> SmartResourceOrchestrator:
    """ดึง instance ของ Smart Resource Orchestrator"""
    global _smart_resource_orchestrator
    if _smart_resource_orchestrator is None:
        _smart_resource_orchestrator = SmartResourceOrchestrator(config)
    return _smart_resource_orchestrator


# ====================================================
# MAIN FOR TESTING
# ====================================================

def main():
    """ฟังก์ชั่นหลักสำหรับทดสอบ"""
    print("🤖 NICEGOLD ENTERPRISE - SMART RESOURCE ORCHESTRATOR")
    print("=" * 70)
    
    # Initialize orchestrator
    config = OrchestrationConfig(
        target_utilization=0.80,
        monitoring_interval=2.0,
        optimization_interval=10.0
    )
    
    orchestrator = get_smart_resource_orchestrator(config)
    
    # Start orchestration
    success = orchestrator.start_orchestration()
    if success:
        print("✅ Smart Resource Orchestration started successfully")
        
        try:
            # Display status
            print("\n" + orchestrator.get_status_summary_text())
            
            # Monitor for 30 seconds
            print("\n⏳ Monitoring for 30 seconds...")
            time.sleep(30)
            
            # Display updated status
            print("\n" + orchestrator.get_status_summary_text())
            
            # Generate detailed report
            report = orchestrator.get_detailed_report()
            print("\n📊 DETAILED REPORT:")
            print(json.dumps(report, indent=2, default=str))
            
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            # Stop orchestration
            orchestrator.stop_orchestration()
            print("\n✅ Smart Resource Orchestration stopped")
    
    else:
        print("❌ Failed to start Smart Resource Orchestration")


if __name__ == "__main__":
    main()
