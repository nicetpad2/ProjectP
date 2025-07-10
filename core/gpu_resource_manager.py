#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎮 ENTERPRISE GPU RESOURCE MANAGER
Mock implementation for systems without GPU resources

This provides a fallback implementation for systems that don't have GPU capabilities
while maintaining the same interface as the full GPU manager.
"""

import logging
from typing import Dict, Any, Optional

class EnterpriseGPUManager:
    """Mock GPU Resource Manager for CPU-only systems"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize mock GPU manager"""
        self.config = config or {}
        self.is_available = False
        self.gpu_count = 0
        self.memory_total = 0
        self.memory_available = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report (CPU-only version)"""
        return {
            "status": "CPU_ONLY",
            "gpu_available": False,
            "gpu_count": 0,
            "memory_total": 0,
            "memory_available": 0,
            "optimization_mode": "CPU_FALLBACK",
            "message": "Running on CPU - GPU not available"
        }
    
    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration (alias for get_optimization_report)"""
        return self.get_optimization_report()
    
    def optimize_for_memory(self, target_percentage: float = 80.0) -> bool:
        """Mock memory optimization"""
        self.logger.info(f"GPU optimization requested ({target_percentage}%) - using CPU fallback")
        return True
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage (returns zeros for CPU-only)"""
        return {
            "total": 0.0,
            "used": 0.0,
            "available": 0.0,
            "percentage": 0.0
        }
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available"""
        return False
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information"""
        return {
            "device_type": "CPU",
            "device_name": "CPU_FALLBACK",
            "compute_capability": None,
            "memory_total": 0,
            "memory_available": 0
        }

# Convenience function for easy import
def get_gpu_manager(config: Optional[Dict] = None) -> EnterpriseGPUManager:
    """Get GPU manager instance"""
    return EnterpriseGPUManager(config)
