#!/usr/bin/env python3
"""
🧠 HIGH MEMORY RESOURCE MANAGER - 80% RAM Optimized
ระบบจัดการทรัพยากรแบบใช้ RAM 80% สำหรับสภาพแวดล้อมที่มี RAM เยอะแต่ CPU จำกัด

🎯 Optimized for:
   🧠 High Memory Usage (80% RAM)
   ⚡ Low CPU Usage (Minimize CPU load)
   📊 Memory-Intensive Operations
   🛡️ CPU-Conservative Processing
"""

import os
import sys
import psutil
import threading
import time
import logging
import warnings
import gc
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from core.unified_enterprise_logger import get_unified_logger

# Suppress warnings to reduce noise
warnings.filterwarnings('ignore')

class HighMemoryResourceManager:
    """
    🧠 High Memory Resource Manager - 80% RAM Optimized
    ระบบจัดการทรัพยากรที่เน้นการใช้ RAM สูงและ CPU ต่ำ
    
    Features:
    - 🧠 High memory usage (80% RAM allocation)
    - ⚡ CPU-conservative operations
    - 📊 Memory-intensive caching
    - 🛡️ CPU-efficient monitoring
    """
    
    def __init__(self, memory_percentage: float = 0.8, cpu_percentage: float = 0.3):
        """Initialize with high memory, low CPU strategy"""
        self.memory_percentage = memory_percentage  # Use 80% RAM
        self.cpu_percentage = cpu_percentage        # Use only 30% CPU
        self.start_time = datetime.now()
        self.monitoring_active = True
        self.monitor_thread = None
        self.performance_data = []
        
        # Initialize logging
        self.logger = get_unified_logger("HighMemoryResourceManager")
        self.logger.info("🧠 High Memory Resource Manager initializing...")
        
        # High memory, low CPU thresholds
        self.memory_warning = 0.85   # Warning at 85% (high)
        self.memory_critical = 0.95  # Critical at 95% (very high)
        self.cpu_warning = 0.70      # Warning at 70% (conservative)
        self.cpu_critical = 0.85     # Critical at 85% (conservative)
        
        # Enable aggressive memory management
        gc.set_threshold(50, 3, 3)  # More frequent garbage collection
        
        # Detect system resources
        self.system_info = self._detect_system_resources()
        self.resource_config = self._calculate_high_memory_allocation()
        
        # Start CPU-efficient monitoring
        self._start_cpu_efficient_monitoring()
        
        self.logger.info("✅ High Memory Resource Manager ready")

    def _detect_system_resources(self) -> Dict[str, Any]:
        """Detect system resources with focus on memory"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count(logical=True)
            
            # Calculate available resources
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            
            info = {
                'cpu_cores': cpu_count,
                'memory_total_gb': round(total_memory_gb, 1),
                'memory_available_gb': round(available_memory_gb, 1),
                'memory_usage_percent': memory.percent,
                'system': 'Linux',
                'timestamp': datetime.now().isoformat(),
                'optimization_mode': 'high_memory_low_cpu'
            }
            
            self.logger.info(f"🧠 High-RAM system detected: {total_memory_gb:.1f}GB total, {available_memory_gb:.1f}GB available", component="HighMemoryRM")
                
            return info
            
        except Exception as e:
            self.logger.warning(f"Resource detection error: {e}", component="HighMemoryRM")
            return {'cpu_cores': 4, 'memory_total_gb': 32.0, 'system': 'Unknown'}

    def _calculate_high_memory_allocation(self) -> Dict[str, Any]:
        """Calculate resource allocation optimized for high memory usage"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count(logical=True)
            
            # High memory allocation (80%)
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            allocated_memory_gb = available_memory_gb * self.memory_percentage
            
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
                    'available_gb': available_memory_gb,
                    'allocated_gb': allocated_memory_gb,
                    'allocation_percentage': self.memory_percentage * 100,
                    'strategy': 'high_memory_usage'
                },
                'optimization': {
                    'batch_size': 512,  # Large batch size for memory
                    'cache_size_mb': int(allocated_memory_gb * 1024 * 0.3),  # 30% of allocated memory for cache
                    'prefetch_data': True,  # Enable data prefetching
                    'memory_mapping': True,  # Use memory mapping for large files
                    'recommended_workers': 1,  # Single worker to save CPU
                    'use_gpu': False,  # CPU only
                    'cpu_intensive_ops': False  # Avoid CPU-heavy operations
                }
            }
            
            self.logger.info(f"📊 High-memory allocation: {allocated_memory_gb:.1f}GB RAM ({self.memory_percentage*100:.0f}%), {allocated_cores} cores ({(allocated_cores/cpu_count)*100:.0f}%)", component="HighMemoryRM")
                
            return config
            
        except Exception as e:
            self.logger.error(f"Allocation calculation error: {e}", component="HighMemoryRM")
            
            # Fallback high-memory allocation
            return {
                'cpu': {'allocated_cores': 1, 'allocation_percentage': 25, 'strategy': 'cpu_conservative'},
                'memory': {'allocated_gb': 16.0, 'allocation_percentage': 80, 'strategy': 'high_memory_usage'},
                'optimization': {'batch_size': 512, 'cache_size_mb': 8192, 'recommended_workers': 1}
            }

    def get_current_performance(self) -> Dict[str, Any]:
        """Get current system performance with focus on memory metrics"""
        try:
            current_time = datetime.now()
            uptime_seconds = (current_time - self.start_time).total_seconds()
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            performance = {
                'uptime': uptime_seconds,
                'uptime_str': str(current_time - self.start_time).split('.')[0],
                'cpu_percent': round(cpu_percent, 1),
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_current_mb': round(memory.used / (1024**2), 1),
                'memory_available_mb': round(memory.available / (1024**2), 1),
                'memory_target': f"{self.memory_percentage*100:.0f}%",
                'cpu_target': f"{self.cpu_percentage*100:.0f}%",
                'memory_percent': round(memory.percent, 1),
                'timestamp': current_time.isoformat(),
                'status': self._get_memory_optimized_status(cpu_percent, memory.percent),
                'total_logs': len(self.performance_data),
                'error_count': 0,
                'warning_count': 0,
                'optimization_mode': 'high_memory_low_cpu'
            }
            
            # Count warnings and errors based on our thresholds
            if performance['status'] == 'critical':
                performance['error_count'] = 1
            elif performance['status'] == 'warning':
                performance['warning_count'] = 1
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Performance monitoring error: {e}", component="HighMemoryRM")
            
            return {
                'uptime': 0,
                'uptime_str': '0:00:00',
                'cpu_percent': 0.0,
                'memory_used_gb': 0.0,
                'memory_percent': 0.0,
                'status': 'error',
                'error_count': 1,
                'warning_count': 0,
                'total_logs': 0
            }

    def get_current_allocation(self) -> Dict[str, Any]:
        """Get current resource allocation status"""
        return {
            'memory': {
                'allocated_gb': self.resource_config['memory']['allocated_gb'],
                'percentage': self.resource_config['memory']['allocation_percentage'],
                'strategy': 'high_memory_usage'
            },
            'cpu': {
                'allocated_cores': self.resource_config['cpu']['allocated_cores'],
                'percentage': self.resource_config['cpu']['allocation_percentage'],
                'strategy': 'cpu_conservative'
            },
            'status': 'active' if self.monitoring_active else 'inactive',
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

    def _get_memory_optimized_status(self, cpu_percent: float, memory_percent: float) -> str:
        """Determine system status with high memory tolerance"""
        # High memory usage is normal and expected
        if cpu_percent > self.cpu_critical * 100:  # CPU is the limiting factor
            return 'critical'
        elif memory_percent > self.memory_critical * 100:  # Very high memory
            return 'critical'
        elif cpu_percent > self.cpu_warning * 100:  # High CPU usage
            return 'warning'
        elif memory_percent > self.memory_warning * 100:  # High memory (but acceptable)
            return 'normal'  # This is expected for our strategy
        else:
            return 'optimal'

    def _start_cpu_efficient_monitoring(self):
        """Start CPU-efficient background monitoring"""
        def monitor():
            consecutive_errors = 0
            max_errors = 3
            
            while self.monitoring_active and consecutive_errors < max_errors:
                try:
                    perf_data = self.get_current_performance()
                    
                    # Keep only last 3 records to save memory (very minimal)
                    self.performance_data.append(perf_data)
                    if len(self.performance_data) > 3:
                        self.performance_data.pop(0)
                    
                    # Alert only on critical CPU usage (our main concern)
                    if perf_data['cpu_percent'] > self.cpu_critical * 100:
                        self.logger.warning(f"🚨 Critical CPU usage - Performance bottleneck", component="HighMemoryRM", 
                                              data={'cpu': perf_data['cpu_percent'], 'memory': perf_data['memory_percent']})
                    elif perf_data['cpu_percent'] > self.cpu_warning * 100:
                        self.logger.info(f"⚠️ High CPU usage detected", component="HighMemoryRM",
                                           data={'cpu': perf_data['cpu_percent']})
                    
                    # Memory usage info (high usage is expected and good)
                    if perf_data['memory_percent'] > 85:
                        self.logger.info(f"🧠 High memory utilization: {perf_data['memory_percent']:.1f}% (Expected)", component="HighMemoryRM")
                    
                    consecutive_errors = 0  # Reset error counter on success
                    time.sleep(60)  # Check every 60 seconds (less frequent to save CPU)
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.error(f"Monitoring error #{consecutive_errors}: {e}", component="HighMemoryRM")
                    
                    time.sleep(120)  # Wait longer on error
            
            if consecutive_errors >= max_errors:
                self.logger.error("🛑 CPU-efficient monitoring stopped due to repeated errors", component="HighMemoryRM")
                self.monitoring_active = False
                
        try:
            self.monitor_thread = threading.Thread(target=monitor, daemon=True)
            self.monitor_thread.start()
            
            self.logger.info("📊 CPU-efficient monitoring started", component="HighMemoryRM")
                
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}", component="HighMemoryRM")
            self.monitoring_active = False

    def start_monitoring(self):
        """Start monitoring system (alias for _start_cpu_efficient_monitoring)"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._start_cpu_efficient_monitoring()
            self.logger.info("🚀 High Memory Resource Manager monitoring started", component="HighMemoryRM")
        else:
            self.logger.info("ℹ️ Monitoring already active", component="HighMemoryRM")

    def stop_monitoring(self):
        """Stop monitoring system"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("🛑 High Memory Resource Manager monitoring stopped", component="HighMemoryRM")

    def restart_monitoring(self):
        """Restart monitoring system"""
        self.logger.info("🔄 Restarting High Memory Resource Manager monitoring", component="HighMemoryRM")
        
        self.stop_monitoring()
        time.sleep(1)  # Brief pause
        self.start_monitoring()

    def is_monitoring_active(self) -> bool:
        """Check if monitoring is active"""
        return self.monitoring_active and (self.monitor_thread is not None and self.monitor_thread.is_alive())

    def get_high_memory_config(self, task_type: str = 'general') -> Dict[str, Any]:
        """Get configuration optimized for high memory usage"""
        base_config = self.resource_config['optimization'].copy()
        
        # Task-specific high-memory optimizations
        if task_type == 'ml_training':
            base_config.update({
                'batch_size': 1024,  # Very large batch size
                'max_epochs': 20,    # Fewer epochs, more data per epoch
                'cache_training_data': True,  # Cache all training data in RAM
                'use_memory_mapping': True,   # Use memory mapping
                'prefetch_factor': 4,         # Prefetch multiple batches
                'num_workers': 1,             # Single worker to save CPU
                'pin_memory': True,           # Pin memory for faster GPU transfer
                'persistent_workers': True    # Keep workers alive
            })
        elif task_type == 'feature_selection':
            base_config.update({
                'n_trials': 30,      # More trials with cached data
                'timeout': 300,      # Longer timeout for thorough search
                'cache_cv_results': True,  # Cache cross-validation results
                'sample_size': 5000,       # Larger sample size
                'memory_efficient': False  # Prioritize speed over memory
            })
        elif task_type == 'data_processing':
            base_config.update({
                'chunk_size': 10000,     # Large chunks
                'cache_processed_data': True,  # Cache processed chunks
                'use_multiprocessing': False,  # Avoid CPU overhead
                'memory_map_files': True,      # Use memory mapping
                'buffer_size_mb': 1024         # Large buffer
            })
        
        return base_config

    def optimize_for_menu1(self) -> Dict[str, Any]:
        """Get optimized configuration specifically for Menu 1 with high memory"""
        base_config = self.get_high_memory_config('ml_training')
        
        # Menu 1 high-memory optimizations
        menu1_config = base_config.copy()
        menu1_config.update({
            'feature_selection': {
                'n_trials': 50,         # More trials with RAM
                'timeout': 600,         # 10 minutes timeout
                'cv_folds': 5,          # Standard CV folds
                'sample_size': 10000,   # Large sample size
                'cache_shap_values': True,  # Cache SHAP calculations
                'memory_intensive': True     # Enable memory-intensive operations
            },
            'cnn_lstm': {
                'epochs': 50,           # More epochs
                'batch_size': 256,      # Large batch size
                'patience': 10,         # More patience
                'validation_split': 0.2,
                'cache_validation': True,   # Cache validation data
                'use_memory_growth': True   # Gradual memory allocation
            },
            'dqn': {
                'episodes': 100,        # More episodes
                'memory_size': 50000,   # Large replay buffer
                'batch_size': 128,      # Large batch size
                'cache_experiences': True,  # Cache experiences in RAM
                'prioritized_replay': True  # Use prioritized experience replay
            },
            'data_processing': {
                'chunk_size': 50000,    # Very large chunks
                'max_features': 100,    # More features
                'cache_features': True, # Cache all engineered features
                'preload_data': True,   # Preload all data into RAM
                'memory_map_csv': True  # Memory map CSV files
            }
        })
        
        return menu1_config

    def get_system_resource_summary(self) -> str:
        """Get system resource summary optimized for high memory display"""
        try:
            config = self.resource_config
            perf = self.get_current_performance()
            
            summary = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║ 🧠 HIGH MEMORY RESOURCE MANAGEMENT SYSTEM (80% RAM Strategy)                            ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

🖥️  SYSTEM CONFIGURATION:
   Strategy: High Memory (80%) + Low CPU (30%)
   Total RAM: {config['memory']['total_gb']:.1f} GB
   Allocated RAM: {config['memory']['allocated_gb']:.1f} GB ({config['memory']['allocation_percentage']:.0f}%)
   CPU Cores: {config['cpu']['allocated_cores']} / {config['cpu']['total_cores']} cores ({config['cpu']['allocation_percentage']:.0f}%)

📊 CURRENT PERFORMANCE:
   Memory Usage: {perf['memory_used_gb']:.1f} GB ({perf['memory_percent']:.1f}%) 
   Memory Available: {perf['memory_available_gb']:.1f} GB
   CPU Usage: {perf['cpu_percent']:.1f}%
   Status: {perf['status'].upper()}
   
⚡ OPTIMIZATION SETTINGS:
   Batch Size: {config['optimization']['batch_size']}
   Cache Size: {config['optimization']['cache_size_mb']} MB
   Memory Mapping: ✅ Enabled
   CPU Conservation: ✅ Active

🎯 PERFORMANCE TARGETS:
   Memory: Use up to 85% (Current: {perf['memory_percent']:.1f}%)
   CPU: Keep under 70% (Current: {perf['cpu_percent']:.1f}%)
╚══════════════════════════════════════════════════════════════════════════════════════════╝
            """
            return summary.strip()
            
        except Exception as e:
            return f"📊 High Memory Resource Manager Active (Error in summary: {e})"

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status optimized for high memory environment"""
        performance = self.get_current_performance()
        
        # Calculate health score with high memory tolerance
        health_score = 100
        
        # CPU is our main concern
        if performance['cpu_percent'] > self.cpu_warning * 100:
            health_score -= 30
        if performance['cpu_percent'] > self.cpu_critical * 100:
            health_score -= 40
        
        # Memory warnings only at very high usage
        if performance['memory_percent'] > self.memory_warning * 100:
            health_score -= 10  # Minor penalty for very high memory
        if performance['memory_percent'] > self.memory_critical * 100:
            health_score -= 20
        
        return {
            'health_score': max(0, health_score),
            'cpu_usage': performance['cpu_percent'],
            'memory_usage': performance['memory_percent'],
            'memory_used_gb': performance['memory_used_gb'],
            'memory_available_gb': performance['memory_available_gb'],
            'status': performance['status'],
            'uptime': performance['uptime_str'],
            'optimization_mode': 'high_memory_low_cpu'
        }

    def check_resource_availability(self) -> Tuple[bool, str]:
        """Check if system has enough resources with high memory focus"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            available_memory_gb = memory.available / (1024**3)
            
            # High memory environment checks
            if available_memory_gb < 2.0:  # Minimum 2GB available
                return False, f"Insufficient memory: {available_memory_gb:.1f}GB available (need 2GB+)"
            
            if cpu_percent > 95:  # Very high CPU threshold
                return False, f"CPU overloaded: {cpu_percent:.1f}%"
            
            # Memory check is more lenient - high usage is expected
            if memory.percent > 98:  # Only critical at 98%+
                return False, f"Memory critically high: {memory.percent:.1f}%"
            
            return True, f"Resources available: {available_memory_gb:.1f}GB RAM, CPU {cpu_percent:.1f}%"
            
        except Exception as e:
            return False, f"Resource check error: {e}"

    def stop_monitoring(self):
        """Stop resource monitoring with proper cleanup"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        self.logger.info("🛑 High memory monitoring stopped", component="HighMemoryRM")

    def force_garbage_collection(self):
        """Force garbage collection to free up memory if needed"""
        try:
            collected = gc.collect()
            self.logger.info(f"🗑️ Garbage collection: freed {collected} objects", component="HighMemoryRM")
            return collected
        except Exception as e:
            self.logger.warning(f"Garbage collection error: {e}", component="HighMemoryRM")
            return 0

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except:
            pass

# Global instance for easy access
_high_memory_resource_manager = None

def initialize_high_memory_intelligent_resources(memory_percentage: float = 0.8, cpu_percentage: float = 0.3, **kwargs) -> HighMemoryResourceManager:
    """Initialize high memory intelligent resource manager"""
    global _high_memory_resource_manager
    
    try:
        _high_memory_resource_manager = HighMemoryResourceManager(
            memory_percentage=memory_percentage,
            cpu_percentage=cpu_percentage
        )
        return _high_memory_resource_manager
    except Exception as e:
        logger.error(f"Failed to initialize high memory resource manager: {e}")
        # Return a minimal fallback
        return HighMemoryResourceManager(memory_percentage=0.6, cpu_percentage=0.4)

def get_high_memory_resource_manager() -> HighMemoryResourceManager:
    """Get global high memory resource manager instance"""
    global _high_memory_resource_manager
    if _high_memory_resource_manager is None:
        _high_memory_resource_manager = initialize_high_memory_intelligent_resources()
    return _high_memory_resource_manager

def get_high_memory_config(task_type: str = 'general') -> Dict[str, Any]:
    """Get high memory optimized configuration for a task"""
    manager = get_high_memory_resource_manager()
    return manager.get_high_memory_config(task_type)

def check_high_memory_system_health() -> Dict[str, Any]:
    """Check current system health with high memory focus"""
    manager = get_high_memory_resource_manager()
    return manager.get_health_status()

if __name__ == "__main__":
    # Test the high memory resource manager
    print("🧠 Testing High Memory Resource Manager...")
    
    manager = initialize_high_memory_intelligent_resources()
    print(manager.get_system_resource_summary())
    
    # Test configurations
    ml_config = manager.get_high_memory_config('ml_training')
    menu1_config = manager.optimize_for_menu1()
    
    print(f"\n📊 ML Training Config: {ml_config}")
    print(f"\n🎯 Menu 1 Config: {menu1_config}")
    
    # Health check
    health = manager.get_health_status()
    print(f"\n🏥 Health Status: {health}")
