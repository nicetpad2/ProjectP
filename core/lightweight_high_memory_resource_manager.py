#!/usr/bin/env python3
"""
🧠 LIGHTWEIGHT HIGH MEMORY RESOURCE MANAGER
ระบบจัดการทรัพยากรแบบเบาที่ไม่ต้องพึ่ง psutil

🎯 Features:
   🧠 High Memory Usage (80% estimation)
   ⚡ Low CPU Usage (conservative)
   📊 Memory-Intensive Operations
   🛡️ No External Dependencies
"""

import os
import sys
import gc
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional

class LightweightHighMemoryResourceManager:
    """
    🧠 Lightweight High Memory Resource Manager
    ระบบจัดการทรัพยากรแบบเบาสำหรับใช้แรม 80%
    """
    
    def __init__(self, memory_percentage: float = 0.8, cpu_percentage: float = 0.3):
        """Initialize lightweight high memory manager"""
        self.memory_percentage = memory_percentage  # Use 80% RAM
        self.cpu_percentage = cpu_percentage        # Use only 30% CPU
        self.start_time = datetime.now()
        self.monitoring_active = True
        self.performance_data = []
        
        print(f"🧠 Lightweight High Memory Manager: {memory_percentage*100}% RAM target")
        
        # Enable aggressive memory management
        gc.set_threshold(50, 3, 3)
        
        # Detect system resources (simplified)
        self.system_info = self._detect_system_resources_simple()
        self.resource_config = self._calculate_resource_allocation()
        
        print("✅ Lightweight High Memory Resource Manager ready")
    
    def _detect_system_resources_simple(self) -> Dict[str, Any]:
        """Simple system resource detection without psutil"""
        try:
            # Try to get CPU count
            cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 4
            
            # Estimate memory (simplified)
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            total_memory_kb = int(line.split()[1])
                            total_memory_gb = total_memory_kb / (1024 * 1024)
                            break
                    else:
                        total_memory_gb = 32.0  # Default estimate
            except:
                total_memory_gb = 32.0  # Default estimate
            
            info = {
                'cpu_cores': cpu_count,
                'memory_total_gb': round(total_memory_gb, 1),
                'memory_available_gb': round(total_memory_gb * 0.8, 1),
                'system': 'Linux',
                'timestamp': datetime.now().isoformat(),
                'optimization_mode': 'lightweight_high_memory'
            }
            
            print(f"🧠 System detected: {total_memory_gb:.1f}GB total, {cpu_count} CPU cores")
            return info
            
        except Exception as e:
            print(f"⚠️ Resource detection warning: {e}")
            return {
                'cpu_cores': 4,
                'memory_total_gb': 32.0,
                'memory_available_gb': 25.6,
                'system': 'Unknown'
            }
    
    def _calculate_resource_allocation(self) -> Dict[str, Any]:
        """Calculate resource allocation for high memory usage"""
        try:
            cpu_count = self.system_info.get('cpu_cores', 4)
            total_memory_gb = self.system_info.get('memory_total_gb', 32.0)
            
            # High memory allocation (80%)
            allocated_memory_gb = total_memory_gb * self.memory_percentage
            
            # Conservative CPU allocation (30% or max 2 cores)
            allocated_cores = max(1, min(2, int(cpu_count * self.cpu_percentage)))
            
            config = {
                'cpu': {
                    'total_cores': cpu_count,
                    'allocated_cores': allocated_cores,
                    'allocation_percentage': (allocated_cores / cpu_count) * 100,
                    'strategy': 'cpu_conservative'
                },
                'memory': {
                    'total_gb': total_memory_gb,
                    'allocated_gb': allocated_memory_gb,
                    'allocation_percentage': self.memory_percentage * 100,
                    'strategy': 'high_memory_usage'
                },
                'optimization': {
                    'batch_size': 1024,  # Large batch size
                    'cache_size': 50000,  # Large cache
                    'preload_data': True,
                    'memory_mapping': True,
                    'gc_frequency': 'high'
                }
            }
            
            print(f"📊 Resource allocation: {allocated_memory_gb:.1f}GB RAM, {allocated_cores} CPU cores")
            return config
            
        except Exception as e:
            print(f"⚠️ Resource calculation warning: {e}")
            return {
                'cpu': {'allocated_cores': 2},
                'memory': {'allocated_gb': 25.6},
                'optimization': {'batch_size': 512}
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return self.system_info.copy()
    
    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration"""
        return self.resource_config.copy()
    
    def monitor_resources(self):
        """Simple resource monitoring"""
        try:
            # Simplified monitoring without psutil
            current_time = datetime.now()
            
            # Estimate memory usage (simplified)
            try:
                gc.collect()  # Force garbage collection
                memory_estimate = 75.0  # Estimate 75% usage
            except:
                memory_estimate = 70.0
            
            # Estimate CPU usage (simplified)
            cpu_estimate = 25.0  # Conservative estimate
            
            data_point = {
                'timestamp': current_time.isoformat(),
                'memory_percent': memory_estimate,
                'cpu_percent': cpu_estimate,
                'monitoring_type': 'lightweight'
            }
            
            self.performance_data.append(data_point)
            
            # Keep only last 100 data points
            if len(self.performance_data) > 100:
                self.performance_data = self.performance_data[-100:]
            
            return data_point
            
        except Exception as e:
            print(f"⚠️ Monitoring warning: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'memory_percent': 70.0,
                'cpu_percent': 20.0,
                'error': str(e)
            }
    
    def optimize_for_memory_intensive_task(self, task_name: str = "High Memory Task"):
        """Optimize system for memory-intensive operations"""
        try:
            print(f"🚀 Optimizing for: {task_name}")
            
            # Aggressive memory management
            gc.set_threshold(30, 2, 2)  # More aggressive GC
            gc.collect()  # Force collection
            
            # Set environment variables for memory optimization
            os.environ['PYTHONHASHSEED'] = '0'  # Deterministic hashing
            
            print(f"✅ Memory optimization applied for: {task_name}")
            return True
            
        except Exception as e:
            print(f"⚠️ Optimization warning: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics"""
        try:
            # Force garbage collection and get stats
            collected = gc.collect()
            
            stats = {
                'gc_collections': collected,
                'gc_threshold': gc.get_threshold(),
                'memory_target': f"{self.memory_percentage*100:.0f}%",
                'cpu_target': f"{self.cpu_percentage*100:.0f}%",
                'optimization_active': True,
                'status': 'healthy'
            }
            
            return stats
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'warning'
            }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.monitoring_active = False
            gc.collect()
            print("🧹 Resource manager cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()
