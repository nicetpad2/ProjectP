#!/usr/bin/env python3
"""
🧠 LIGHTWEIGHT RESOURCE MANAGER - No Dependencies
ระบบจัดการทรัพยากรแบบเบาที่ไม่ต้องพึ่ง external dependencies

🎯 Features:
   🧠 Built-in Memory Management
   ⚡ CPU Conservative Operations
   📊 Memory-Intensive Processing
   🛡️ Zero Dependencies
"""

import os
import gc
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

class LightweightResourceManager:
    """
    🧠 Lightweight Resource Manager - 80% RAM Optimized
    ระบบจัดการทรัพยากรที่ไม่ต้องพึ่ง external dependencies
    
    Features:
    - 🧠 High memory usage strategy (80% RAM)
    - ⚡ CPU-conservative operations
    - 📊 Built-in monitoring
    - 🛡️ Zero external dependencies
    """
    
    def __init__(self, memory_percentage: float = 0.8, cpu_percentage: float = 0.3):
        """Initialize lightweight resource manager"""
        self.memory_percentage = memory_percentage  # Use 80% RAM
        self.cpu_percentage = cpu_percentage        # Use only 30% CPU
        self.start_time = datetime.now()
        self.monitoring_active = True
        self.performance_data = []
        
        # Simple logging
        self.logger = None
        try:
            from core.advanced_terminal_logger import get_terminal_logger
            self.logger = get_terminal_logger()
            self.logger.system("🧠 Lightweight Resource Manager initializing...", "LightweightRM")
        except:
            print("🧠 Lightweight Resource Manager initializing...")
        
        # Lightweight thresholds
        self.memory_warning = 0.85   # Warning at 85%
        self.memory_critical = 0.95  # Critical at 95%
        self.cpu_warning = 0.70      # Warning at 70%
        self.cpu_critical = 0.85     # Critical at 85%
        
        # Enable aggressive memory management
        gc.set_threshold(50, 3, 3)  # Frequent garbage collection
        
        # Detect system resources (lightweight)
        self.system_info = self._detect_system_resources_lightweight()
        self.resource_config = self._calculate_resource_allocation()
        
        # Start lightweight monitoring
        self._start_lightweight_monitoring()
        
        if self.logger:
            self.logger.success("✅ Lightweight Resource Manager ready", "LightweightRM")
        else:
            print("✅ Lightweight Resource Manager ready")
    
    def _detect_system_resources_lightweight(self) -> Dict[str, Any]:
        """Detect system resources without external dependencies"""
        try:
            # Get CPU count from os
            cpu_count = os.cpu_count() or 4
            
            # Estimate memory (simplified approach)
            try:
                # Try reading from /proc/meminfo on Linux
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            total_memory_kb = int(line.split()[1])
                            total_memory_gb = total_memory_kb / (1024**2)
                            break
                    else:
                        total_memory_gb = 32.0  # Default assumption
            except:
                total_memory_gb = 32.0  # Default assumption
            
            info = {
                'cpu_cores': cpu_count,
                'memory_total_gb': round(total_memory_gb, 1),
                'memory_available_gb': round(total_memory_gb * 0.8, 1),  # Estimate 80% available
                'system': 'Linux',
                'timestamp': datetime.now().isoformat(),
                'optimization_mode': 'lightweight_high_memory'
            }
            
            if self.logger:
                self.logger.info(f"🧠 System detected: {total_memory_gb:.1f}GB total, {cpu_count} cores", "LightweightRM")
            else:
                print(f"🧠 System detected: {total_memory_gb:.1f}GB total, {cpu_count} cores")
                
            return info
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Resource detection error: {e}", "LightweightRM")
            else:
                print(f"⚠️ Resource detection error: {e}")
            return {'cpu_cores': 4, 'memory_total_gb': 32.0, 'system': 'Unknown'}
    
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
                    'cache_size': 10000,  # Large cache
                    'threading_limit': allocated_cores,
                    'gc_frequency': 'high'
                }
            }
            
            if self.logger:
                self.logger.info(f"📊 Resource allocation: {allocated_memory_gb:.1f}GB RAM, {allocated_cores} cores", "LightweightRM")
            else:
                print(f"📊 Resource allocation: {allocated_memory_gb:.1f}GB RAM, {allocated_cores} cores")
                
            return config
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Resource allocation error: {e}", "LightweightRM")
            else:
                print(f"⚠️ Resource allocation error: {e}")
            return {'cpu': {'allocated_cores': 1}, 'memory': {'allocated_gb': 25.6}}
    
    def _start_lightweight_monitoring(self):
        """Start lightweight resource monitoring"""
        try:
            def monitor_resources():
                while self.monitoring_active:
                    try:
                        # Lightweight monitoring (just track basic info)
                        timestamp = datetime.now()
                        
                        # Simple memory usage estimation
                        gc.collect()  # Force garbage collection
                        
                        # Record performance data
                        data_point = {
                            'timestamp': timestamp.isoformat(),
                            'memory_usage': 'high',  # Assumption for high memory mode
                            'cpu_usage': 'low',      # Conservative CPU usage
                            'gc_collections': len(gc.garbage)
                        }
                        
                        self.performance_data.append(data_point)
                        
                        # Keep only last 100 data points
                        if len(self.performance_data) > 100:
                            self.performance_data = self.performance_data[-100:]
                        
                        # Sleep for monitoring interval (low frequency)
                        time.sleep(30)  # Monitor every 30 seconds
                        
                    except Exception as e:
                        # Silent error handling for monitoring
                        pass
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            monitor_thread.start()
            
            if self.logger:
                self.logger.info("📊 Lightweight monitoring started", "LightweightRM")
            else:
                print("📊 Lightweight monitoring started")
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Monitoring startup error: {e}", "LightweightRM")
            else:
                print(f"⚠️ Monitoring startup error: {e}")
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration for high memory usage"""
        return {
            'memory_strategy': 'aggressive_allocation',
            'cpu_strategy': 'conservative',
            'batch_size': self.resource_config.get('optimization', {}).get('batch_size', 1024),
            'cache_size': self.resource_config.get('optimization', {}).get('cache_size', 10000),
            'threading_limit': self.resource_config.get('optimization', {}).get('threading_limit', 2),
            'memory_target': f"{self.memory_percentage*100:.0f}%",
            'cpu_target': f"{self.cpu_percentage*100:.0f}%"
        }
    
    def get_memory_settings(self) -> Dict[str, Any]:
        """Get memory settings optimized for 80% usage"""
        return {
            'target_usage_percent': self.memory_percentage * 100,
            'allocated_gb': self.resource_config.get('memory', {}).get('allocated_gb', 25.6),
            'strategy': 'high_memory_ml_processing',
            'cache_optimization': True,
            'batch_processing': True,
            'memory_mapping': True
        }
    
    def get_cpu_settings(self) -> Dict[str, Any]:
        """Get CPU settings optimized for conservative usage"""
        return {
            'target_usage_percent': self.cpu_percentage * 100,
            'allocated_cores': self.resource_config.get('cpu', {}).get('allocated_cores', 1),
            'strategy': 'cpu_conservative',
            'threading_limit': self.resource_config.get('cpu', {}).get('allocated_cores', 1),
            'process_priority': 'normal'
        }
    
    def optimize_for_ml_training(self) -> Dict[str, Any]:
        """Optimize settings for ML training with high memory"""
        return {
            'batch_size': 1024,  # Large batch size
            'memory_usage': 'aggressive',
            'cpu_usage': 'conservative',
            'caching': 'enabled',
            'parallel_jobs': min(2, self.resource_config.get('cpu', {}).get('allocated_cores', 1)),
            'memory_mapping': True,
            'data_preprocessing': 'memory_intensive'
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            if not self.performance_data:
                return {
                    'status': 'monitoring_starting',
                    'data_points': 0,
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
                }
            
            return {
                'status': 'active',
                'data_points': len(self.performance_data),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                'memory_strategy': 'high_memory_80_percent',
                'cpu_strategy': 'conservative_30_percent',
                'last_updated': self.performance_data[-1]['timestamp'] if self.performance_data else None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.monitoring_active = False
            gc.collect()
            
            if self.logger:
                self.logger.info("🧹 Lightweight Resource Manager cleanup completed", "LightweightRM")
            else:
                print("🧹 Lightweight Resource Manager cleanup completed")
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Cleanup error: {e}", "LightweightRM")
            else:
                print(f"⚠️ Cleanup error: {e}")
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass
