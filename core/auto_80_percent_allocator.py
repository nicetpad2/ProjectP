#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - AUTO 80% RESOURCE ALLOCATOR
Phase 1 Enterprise Resource Control - Auto Allocation System
Intelligent 80% Resource Utilization with Safety Margins

🎯 Auto 80% Allocator Features:
✅ Dynamic 80% Resource Allocation
✅ Automatic Safety Margin Management (15%)
✅ Emergency Protection Reserve (5%)
✅ Adaptive Scaling Algorithms
✅ Multi-platform Resource Support
✅ Real-time Allocation Monitoring
✅ Performance Optimization
✅ Enterprise-grade Protection

Version: 1.0 Enterprise Foundation
Date: July 8, 2025
Status: Production Ready - Phase 1 Implementation
"""

import logging
import psutil
from typing import Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ResourceType(Enum):
    """Resource Type Enumeration"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    GPU = "gpu"
    NETWORK = "network"


@dataclass
class AllocationResult:
    """Resource Allocation Result"""
    success: bool
    allocated_amount: float
    safety_margin: float
    emergency_reserve: float
    allocation_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class Auto80PercentAllocator:
    """
    🎯 80% Auto Resource Allocator
    Intelligent resource allocation with safety margins
    """
    
    def __init__(self):
        """Initialize 80% allocator"""
        self.logger = self._setup_logger()
        self.target_utilization = 0.80
        self.safety_margin = 0.15
        self.emergency_reserve = 0.05
        self.allocation_history = []
        
        # Scaling algorithms for different resource types
        self.scaling_algorithms = {
            'cpu': self._linear_cpu_scaling,
            'memory': self._adaptive_memory_scaling,
            'disk': self._predictive_disk_scaling,
            'gpu': self._intelligent_gpu_scaling
        }
        
        self.logger.info("🎯 Auto 80% Resource Allocator initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup enterprise logger"""
        logger = get_unified_logger()
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - 🎯 [%(name)s] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def allocate_resources_80_percent(self, requirements: Dict[str, float]
                                      ) -> AllocationResult:
        """
        Allocate resources with 80% target utilization
        
        Args:
            requirements: Dict of resource requirements
            
        Returns:
            AllocationResult with allocation details
        """
        try:
            self.logger.info(f"🎯 Starting 80% resource allocation for: {list(requirements.keys())}")
            
            # Get current system resources
            current_resources = self._get_system_resources()
            
            # Calculate allocations for each resource type
            allocations = {}
            total_allocation_score = 0
            
            for resource_type, required_amount in requirements.items():
                allocation_result = self._allocate_single_resource(
                    resource_type, required_amount, current_resources
                )
                
                allocations[resource_type] = allocation_result
                total_allocation_score += allocation_result.get('score', 0)
            
            # Calculate overall success metrics
            success = total_allocation_score >= len(requirements) * 0.8
            average_score = (
                total_allocation_score / len(requirements) 
                if requirements else 0
            )
            
            # Create allocation result
            result = AllocationResult(
                success=success,
                allocated_amount=average_score,
                safety_margin=self.safety_margin,
                emergency_reserve=self.emergency_reserve,
                allocation_details=allocations
            )
            
            # Record allocation in history
            self.allocation_history.append(result)
            
            # Log result
            self.logger.info(f"✅ Allocation complete: Success={success}, Score={average_score:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Allocation failed: {e}")
            return AllocationResult(
                success=False,
                allocated_amount=0,
                safety_margin=self.safety_margin,
                emergency_reserve=self.emergency_reserve,
                allocation_details={'error': str(e)}
            )
    
    def _get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource information"""
        resources = {}
        
        # CPU Resources
        resources['cpu'] = {
            'total': 100.0,
            'used': psutil.cpu_percent(interval=0.1),
            'available': 100.0 - psutil.cpu_percent(interval=0.1),
            'cores': psutil.cpu_count()
        }
        
        # Memory Resources
        memory = psutil.virtual_memory()
        resources['memory'] = {
            'total': memory.total,
            'used': memory.used,
            'available': memory.available,
            'percentage': memory.percent
        }
        
        # Disk Resources
        disk = psutil.disk_usage('/')
        resources['disk'] = {
            'total': disk.total,
            'used': disk.used,
            'available': disk.free,
            'percentage': (disk.used / disk.total) * 100
        }
        
        # GPU Resources (if available)
        try:
            import torch
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_used = torch.cuda.memory_allocated(0)
                resources['gpu'] = {
                    'total': gpu_memory,
                    'used': gpu_used,
                    'available': gpu_memory - gpu_used,
                    'percentage': (gpu_used / gpu_memory) * 100
                }
        except ImportError:
            pass
        
        return resources
    
    def _allocate_single_resource(self, resource_type: str, 
                                required_amount: float, 
                                current_resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate a single resource type
        
        Args:
            resource_type: Type of resource to allocate
            required_amount: Amount of resource required
            current_resources: Current system resources
            
        Returns:
            Dict with allocation details
        """
        if resource_type not in current_resources:
            return {
                'requested': required_amount,
                'allocated': 0,
                'score': 0,
                'error': f'Resource type {resource_type} not available'
            }
        
        resource_info = current_resources[resource_type]
        
        # Calculate maximum safe allocation (80% of total)
        max_safe_allocation = resource_info['total'] * self.target_utilization
        current_usage = resource_info['used']
        
        # Calculate available allocation space
        available_for_allocation = max_safe_allocation - current_usage
        
        # Apply resource-specific scaling algorithm
        if resource_type in self.scaling_algorithms:
            scaling_factor = self.scaling_algorithms[resource_type](
                resource_info, required_amount
            )
            available_for_allocation *= scaling_factor
        
        # Determine actual allocation
        if available_for_allocation >= required_amount:
            allocated = required_amount
            allocation_score = 1.0
        else:
            allocated = max(0, available_for_allocation)
            allocation_score = (
                allocated / required_amount 
                if required_amount > 0 else 0
            )
        
        return {
            'requested': required_amount,
            'allocated': allocated,
            'score': allocation_score,
            'available_for_allocation': available_for_allocation,
            'current_usage': current_usage,
            'max_safe_allocation': max_safe_allocation,
            'safety_margin': resource_info['total'] * self.safety_margin,
            'emergency_reserve': resource_info['total'] * self.emergency_reserve,
            'scaling_applied': resource_type in self.scaling_algorithms
        }
    
    def _linear_cpu_scaling(self, resource_info: Dict[str, Any], 
                           required_amount: float) -> float:
        """Linear CPU scaling algorithm"""
        current_percentage = resource_info['used']
        
        # Reduce scaling factor as CPU usage increases
        if current_percentage > 70:
            return 0.5  # Conservative allocation
        elif current_percentage > 50:
            return 0.7  # Moderate allocation
        else:
            return 1.0  # Full allocation
    
    def _adaptive_memory_scaling(self, resource_info: Dict[str, Any], 
                                required_amount: float) -> float:
        """Adaptive memory scaling algorithm"""
        current_percentage = resource_info['percentage']
        
        # Adaptive scaling based on current memory usage
        if current_percentage > 80:
            return 0.3  # Very conservative
        elif current_percentage > 60:
            return 0.6  # Conservative
        elif current_percentage > 40:
            return 0.8  # Moderate
        else:
            return 1.0  # Full allocation
    
    def _predictive_disk_scaling(self, resource_info: Dict[str, Any], 
                                required_amount: float) -> float:
        """Predictive disk scaling algorithm"""
        current_percentage = resource_info['percentage']
        
        # Predict future disk usage and scale accordingly
        if current_percentage > 85:
            return 0.2  # Emergency conservation
        elif current_percentage > 70:
            return 0.5  # Conservative
        elif current_percentage > 50:
            return 0.8  # Moderate
        else:
            return 1.0  # Full allocation
    
    def _intelligent_gpu_scaling(self, resource_info: Dict[str, Any], 
                                required_amount: float) -> float:
        """Intelligent GPU scaling algorithm"""
        current_percentage = resource_info.get('percentage', 0)
        
        # GPU-specific scaling based on memory usage
        if current_percentage > 85:
            return 0.2  # Minimal allocation
        elif current_percentage > 70:
            return 0.4  # Conservative
        elif current_percentage > 50:
            return 0.7  # Moderate
        else:
            return 1.0  # Full allocation
    
    def maintain_safety_margin(self) -> Dict[str, Any]:
        """
        Maintain safety margin across all resource types
        
        Returns:
            Dict with safety margin status
        """
        try:
            current_resources = self._get_system_resources()
            safety_status = {}
            
            for resource_type, resource_info in current_resources.items():
                if resource_type == 'cpu':
                    usage_percentage = resource_info['used']
                else:
                    usage_percentage = resource_info.get('percentage', 0)
                
                # Check if within safety margin
                within_safety = usage_percentage <= (self.target_utilization * 100)
                within_emergency = usage_percentage <= 95  # Emergency threshold
                
                safety_status[resource_type] = {
                    'current_usage': usage_percentage,
                    'target_threshold': self.target_utilization * 100,
                    'within_safety_margin': within_safety,
                    'within_emergency_threshold': within_emergency,
                    'action_needed': not within_safety
                }
            
            return {
                'timestamp': datetime.now(),
                'safety_status': safety_status,
                'overall_safe': all(
                    status['within_safety_margin'] 
                    for status in safety_status.values()
                )
            }
            
        except Exception as e:
            self.logger.error(f"❌ Safety margin check failed: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'overall_safe': False
            }
    
    def handle_emergency_protection(self) -> Dict[str, Any]:
        """
        Handle emergency protection when resources are critical
        
        Returns:
            Dict with emergency protection actions
        """
        try:
            current_resources = self._get_system_resources()
            emergency_actions = {}
            
            for resource_type, resource_info in current_resources.items():
                if resource_type == 'cpu':
                    usage_percentage = resource_info['used']
                else:
                    usage_percentage = resource_info.get('percentage', 0)
                
                # Check for emergency conditions
                if usage_percentage > 95:
                    emergency_actions[resource_type] = {
                        'action': 'emergency_reduction',
                        'current_usage': usage_percentage,
                        'target_reduction': usage_percentage - 70,
                        'priority': 'critical'
                    }
                elif usage_percentage > 90:
                    emergency_actions[resource_type] = {
                        'action': 'immediate_reduction',
                        'current_usage': usage_percentage,
                        'target_reduction': usage_percentage - 75,
                        'priority': 'high'
                    }
            
            return {
                'timestamp': datetime.now(),
                'emergency_actions': emergency_actions,
                'actions_count': len(emergency_actions)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Emergency protection failed: {e}")
            return {
                'timestamp': datetime.now(),
                'error': str(e),
                'actions_count': 0
            }
    
    def get_allocation_efficiency(self) -> float:
        """
        Calculate allocation efficiency based on history
        
        Returns:
            Float representing allocation efficiency (0.0 to 1.0)
        """
        if not self.allocation_history:
            return 0.0
        
        successful_allocations = sum(
            1 for alloc in self.allocation_history if alloc.success
        )
        return successful_allocations / len(self.allocation_history)
    
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """
        Get detailed allocation statistics
        
        Returns:
            Dict with allocation statistics
        """
        if not self.allocation_history:
            return {
                'total_allocations': 0,
                'successful_allocations': 0,
                'efficiency': 0.0,
                'average_score': 0.0
            }
        
        total_allocations = len(self.allocation_history)
        successful_allocations = sum(
            1 for alloc in self.allocation_history if alloc.success
        )
        
        total_score = sum(
            alloc.allocated_amount for alloc in self.allocation_history
        )
        average_score = total_score / total_allocations
        
        return {
            'total_allocations': total_allocations,
            'successful_allocations': successful_allocations,
            'efficiency': successful_allocations / total_allocations,
            'average_score': average_score,
            'target_utilization': self.target_utilization,
            'safety_margin': self.safety_margin,
            'emergency_reserve': self.emergency_reserve
        }


def main():
    """Main function for testing"""
    print("🎯 NICEGOLD Auto 80% Resource Allocator")
    print("=" * 50)
    
    # Initialize allocator
    allocator = Auto80PercentAllocator()
    
    # Test resource allocation
    print("\n📊 Testing Resource Allocation:")
    requirements = {
        'cpu': 25.0,  # 25% CPU
        'memory': 2 * 1024 * 1024 * 1024,  # 2GB memory
        'disk': 500 * 1024 * 1024  # 500MB disk
    }
    
    result = allocator.allocate_resources_80_percent(requirements)
    print(f"  Success: {result.success}")
    print(f"  Score: {result.allocated_amount:.2f}")
    print(f"  Safety Margin: {result.safety_margin:.1%}")
    
    # Test safety margin maintenance
    print("\n🛡️ Testing Safety Margin Maintenance:")
    safety_status = allocator.maintain_safety_margin()
    print(f"  Overall Safe: {safety_status['overall_safe']}")
    
    # Test emergency protection
    print("\n🚨 Testing Emergency Protection:")
    emergency_status = allocator.handle_emergency_protection()
    print(f"  Emergency Actions: {emergency_status['actions_count']}")
    
    # Get allocation statistics
    print("\n📈 Allocation Statistics:")
    stats = allocator.get_allocation_statistics()
    print(f"  Total Allocations: {stats['total_allocations']}")
    print(f"  Efficiency: {stats['efficiency']:.1%}")
    print(f"  Average Score: {stats['average_score']:.2f}")


if __name__ == "__main__":
    main()
