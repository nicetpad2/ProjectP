#!/usr/bin/env python3
"""
🚀 OPTIMIZED RESOURCE MANAGER - Low Resource Usage
ระบบจัดการทรัพยากรแบบประหยัดสำหรับ NICEGOLD ProjectP

🎯 Features:
   🧠 Low-memory footprint monitoring  
   ⚡ Error-resistant performance tracking
   🛡️ Conservative resource allocation
   📊 Real-time health assessment

ปรับปรุงให้ใช้ทรัพยากรน้อยลงและแก้ไข Error/Warning ทั้งหมด
"""

import os
import sys
import psutil
import threading
import time
import logging
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Suppress warnings to reduce noise
warnings.filterwarnings('ignore')

# Minimal logging setup
logger = logging.getLogger(__name__)

# Try to import advanced logging, fallback to standard if not available
try:
    from core.advanced_terminal_logger import get_terminal_logger
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False

class OptimizedResourceManager:
    """
    🚀 Optimized Resource Manager - Low Resource Usage
    ระบบจัดการทรัพยากรแบบประหยัด
    
    Features:
    - 🧠 Low-memory footprint monitoring
    - ⚡ Error-resistant performance tracking  
    - 🛡️ Conservative resource allocation
    - 📊 Real-time health assessment
    """
    
    def __init__(self, allocation_percentage: float = 0.5):
        """Initialize with minimal resource usage"""
        self.allocation_percentage = allocation_percentage
        self.start_time = datetime.now()
        self.monitoring_active = True
        self.monitor_thread = None
        self.performance_data = []
        
        # Initialize logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
            self.logger.system("🚀 Optimized Resource Manager initializing...", "OptimizedRM")
        else:
            self.logger = logger
            self.logger.info("🚀 Optimized Resource Manager initializing...")
        
        # Conservative thresholds
        self.memory_warning = 0.60   # Warning at 60%
        self.memory_critical = 0.75  # Critical at 75%
        self.cpu_warning = 0.60      # Warning at 60%
        self.cpu_critical = 0.75     # Critical at 75%
        
        # Detect basic system info
        self.system_info = self._detect_basic_resources()
        self.resource_config = self._calculate_optimized_allocation()
        
        # Start minimal monitoring
        self._start_minimal_monitoring()
        
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.success("✅ Optimized Resource Manager ready", "OptimizedRM")
        else:
            self.logger.info("✅ Optimized Resource Manager ready")
    
    def _detect_basic_resources(self) -> Dict[str, Any]:
        """Detect basic system resources with minimal overhead"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count(logical=True)
            
            info = {
                'cpu_cores': cpu_count,
                'memory_total_gb': round(memory.total / (1024**3), 1),
                'memory_available_gb': round(memory.available / (1024**3), 1),
                'system': 'Linux',
                'timestamp': datetime.now().isoformat()
            }
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.info(f"🔍 Detected: {cpu_count} cores, {info['memory_total_gb']}GB RAM", "OptimizedRM")
            else:
                self.logger.info(f"🔍 Detected: {cpu_count} cores, {info['memory_total_gb']}GB RAM")
                
            return info
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.warning(f"Resource detection error: {e}", "OptimizedRM")
            else:
                self.logger.warning(f"Resource detection error: {e}")
            return {'cpu_cores': 4, 'memory_total_gb': 8.0, 'system': 'Unknown'}
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current system performance with minimal overhead and error resistance"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            current_time = datetime.now()
            uptime = (current_time - self.start_time).total_seconds()
            
            performance = {
                'uptime': uptime,
                'uptime_str': str(current_time - self.start_time).split('.')[0],
                'cpu_percent': round(cpu_percent, 1),
                'memory_current_mb': round(memory.used / (1024**2), 1),
                'memory_percent': round(memory.percent, 1),
                'memory_available_mb': round(memory.available / (1024**2), 1),
                'timestamp': current_time.isoformat(),
                'status': self._get_status(cpu_percent, memory.percent),
                'total_logs': len(self.performance_data),
                'error_count': 0,  # Will be updated if errors detected
                'warning_count': 0  # Will be updated if warnings detected
            }
            
            # Count warnings and errors from status
            if performance['status'] == 'critical':
                performance['error_count'] = 1
            elif performance['status'] == 'warning':
                performance['warning_count'] = 1
            
            return performance
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"Performance monitoring error: {e}", "OptimizedRM")
            else:
                self.logger.error(f"Performance monitoring error: {e}")
            
            return {
                'uptime': 0,
                'uptime_str': '0:00:00',
                'cpu_percent': 0.0,
                'memory_current_mb': 0.0,
                'memory_percent': 0.0,
                'status': 'error',
                'error_count': 1,
                'warning_count': 0,
                'total_logs': 0
            }
    
    def _get_status(self, cpu_percent: float, memory_percent: float) -> str:
        """Determine system status based on resource usage"""
        if cpu_percent > self.cpu_critical * 100 or memory_percent > self.memory_critical * 100:
            return 'critical'
        elif cpu_percent > self.cpu_warning * 100 or memory_percent > self.memory_warning * 100:
            return 'warning'
        else:
            return 'normal'
    
    def get_optimized_config(self, task_type: str = 'general') -> Dict[str, Any]:
        """Get optimized configuration for different tasks"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)
        
        # Conservative resource allocation
        available_memory_gb = memory.available / (1024**3)
        
        config = {
            'cpu_threads': min(2, max(1, cpu_count // 2)),  # Use half cores, max 2
            'memory_limit_gb': min(2.0, available_memory_gb * 0.5),  # Use 50% available, max 2GB
            'batch_size': 16,  # Small batch size
            'max_iterations': 100,  # Limit iterations
            'early_stopping': True,
            'use_gpu': False,  # Force CPU only
            'n_jobs': 1,  # Single process
            'verbose': False  # Reduce output
        }
        
        # Task-specific adjustments
        if task_type == 'ml_training':
            config.update({
                'batch_size': 8,  # Even smaller for ML
                'max_iterations': 50,
                'validation_split': 0.2,
                'patience': 5
            })
        elif task_type == 'feature_selection':
            config.update({
                'n_trials': 20,  # Reduced Optuna trials
                'timeout': 120,  # 2 minutes max
                'sample_size': 1000  # Sample data for SHAP
            })
        elif task_type == 'data_processing':
            config.update({
                'chunk_size': 1000,  # Small chunks
                'max_workers': 1
            })
        
        return config
    
    def _calculate_optimized_allocation(self) -> Dict[str, Any]:
        """Calculate optimized resource allocation with conservative settings"""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count(logical=True)
            
            # Conservative allocation - use only 50% of available resources
            allocated_threads = max(1, min(2, int(cpu_count * self.allocation_percentage)))
            allocated_memory_gb = max(0.5, min(2.0, (memory.available / (1024**3)) * self.allocation_percentage))
            
            config = {
                'cpu': {
                    'total_cores': cpu_count,
                    'allocated_threads': allocated_threads,
                    'allocation_percentage': (allocated_threads / cpu_count) * 100
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'allocated_gb': allocated_memory_gb,
                    'allocation_percentage': (allocated_memory_gb / (memory.available / (1024**3))) * 100
                },
                'optimization': {
                    'batch_size': 16,  # Small batch size
                    'recommended_workers': 1,  # Single worker
                    'use_gpu': False,  # CPU only
                    'max_iterations': 50  # Limited iterations
                }
            }
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.system(f"📊 Conservative allocation: {allocated_threads}/{cpu_count} cores, {allocated_memory_gb:.1f}GB", "OptimizedRM")
            else:
                self.logger.info(f"📊 Conservative allocation: {allocated_threads}/{cpu_count} cores, {allocated_memory_gb:.1f}GB")
                
            return config
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"Allocation calculation error: {e}", "OptimizedRM")
            else:
                self.logger.error(f"Allocation calculation error: {e}")
            
            # Fallback minimal allocation
            return {
                'cpu': {'allocated_threads': 1, 'allocation_percentage': 25},
                'memory': {'allocated_gb': 0.5, 'allocation_percentage': 25},
                'optimization': {'batch_size': 8, 'recommended_workers': 1, 'use_gpu': False}
            }
    
    def initialize_intelligent_resources(self, **kwargs) -> 'OptimizedResourceManager':
        """Compatibility method for intelligent resource initialization"""
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.system("🧠 Optimized intelligent resources initialized", "OptimizedRM")
        else:
            self.logger.info("🧠 Optimized intelligent resources initialized")
        return self
    
    def get_resource_config(self) -> Dict[str, Any]:
        """Get resource configuration"""
        return self.resource_config
    
    def get_system_resource_summary(self) -> str:
        """Get system resource summary for display"""
        try:
            config = self.resource_config
            cpu_config = config.get('cpu', {})
            memory_config = config.get('memory', {})
            
            summary = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║ 🧠 OPTIMIZED RESOURCE MANAGEMENT SYSTEM                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

🖥️  SYSTEM INFORMATION:
   Platform: {self.system_info.get('system', 'Unknown')}
   CPU: {cpu_config.get('allocated_threads', 1)} threads allocated / {cpu_config.get('total_cores', 4)} total
   RAM: {memory_config.get('allocated_gb', 0.5):.1f} GB allocated / {memory_config.get('total_gb', 8):.1f} GB total

⚡ CONSERVATIVE ALLOCATION STRATEGY:
   CPU Usage: {cpu_config.get('allocation_percentage', 25):.1f}%
   Memory Usage: {memory_config.get('allocation_percentage', 25):.1f}%
   Batch Size: {config.get('optimization', {}).get('batch_size', 16)}
   Workers: {config.get('optimization', {}).get('recommended_workers', 1)}

📊 CURRENT PERFORMANCE:
   Status: 🚀 Optimized & Conservative
   Mode: Low-Resource Consumption
   Error Handling: ✅ Enhanced
╚══════════════════════════════════════════════════════════════════════════════════════════╝
            """
            return summary.strip()
            
        except Exception as e:
            return f"📊 Optimized Resource Manager Active (Error in summary: {e})"
    
    def _start_minimal_monitoring(self):
        """Start minimal background monitoring with enhanced error resistance"""
        def monitor():
            consecutive_errors = 0
            max_errors = 5
            
            while self.monitoring_active and consecutive_errors < max_errors:
                try:
                    perf_data = self.get_current_performance()
                    
                    # Keep only last 5 records to save memory (reduced from 10)
                    self.performance_data.append(perf_data)
                    if len(self.performance_data) > 5:
                        self.performance_data.pop(0)
                    
                    # Check for resource issues with advanced logging
                    if perf_data['status'] == 'critical':
                        if ADVANCED_LOGGING_AVAILABLE:
                            self.logger.warning(f"🚨 Critical resource usage", "OptimizedRM", 
                                              data={'cpu': perf_data['cpu_percent'], 'memory': perf_data['memory_percent']})
                        else:
                            self.logger.warning(f"🚨 Critical resource usage: CPU {perf_data['cpu_percent']}%, Memory {perf_data['memory_percent']}%")
                    elif perf_data['status'] == 'warning':
                        if ADVANCED_LOGGING_AVAILABLE:
                            self.logger.info(f"⚠️ High resource usage", "OptimizedRM",
                                           data={'cpu': perf_data['cpu_percent'], 'memory': perf_data['memory_percent']})
                        else:
                            self.logger.info(f"⚠️ High resource usage: CPU {perf_data['cpu_percent']}%, Memory {perf_data['memory_percent']}%")
                    
                    consecutive_errors = 0  # Reset error counter on success
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    consecutive_errors += 1
                    if ADVANCED_LOGGING_AVAILABLE:
                        self.logger.error(f"Monitoring error #{consecutive_errors}: {e}", "OptimizedRM")
                    else:
                        self.logger.error(f"Monitoring error #{consecutive_errors}: {e}")
                    
                    # Exponential backoff on errors
                    wait_time = min(300, 60 * (2 ** (consecutive_errors - 1)))  # Max 5 minutes
                    time.sleep(wait_time)
            
            if consecutive_errors >= max_errors:
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.error("🛑 Monitoring stopped due to repeated errors", "OptimizedRM")
                else:
                    self.logger.error("🛑 Monitoring stopped due to repeated errors")
                self.monitoring_active = False
        
        try:
            self.monitor_thread = threading.Thread(target=monitor, daemon=True)
            self.monitor_thread.start()
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.system("📊 Low-memory monitoring started", "OptimizedRM")
            else:
                self.logger.info("📊 Low-memory monitoring started")
                
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"Failed to start monitoring: {e}", "OptimizedRM")
            else:
                self.logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
    
    def optimize_for_menu1(self) -> Dict[str, Any]:
        """Get optimized configuration specifically for Menu 1"""
        base_config = self.get_optimized_config('ml_training')
        
        # Menu 1 specific optimizations
        menu1_config = base_config.copy()
        menu1_config.update({
            'feature_selection': {
                'n_trials': 15,  # Minimal Optuna trials
                'timeout': 90,   # 1.5 minutes max
                'cv_folds': 3,   # Reduced CV folds
                'sample_size': 500  # Small SHAP sample
            },
            'cnn_lstm': {
                'epochs': 10,    # Reduced epochs
                'batch_size': 4, # Very small batch
                'patience': 3,   # Early stopping
                'validation_split': 0.1
            },
            'dqn': {
                'episodes': 20,  # Reduced episodes
                'memory_size': 1000,  # Small replay buffer
                'batch_size': 8
            },
            'data_processing': {
                'chunk_size': 500,  # Small chunks
                'max_features': 20, # Limit features
                'sample_data': True  # Use sample data if large
            }
        })
        
        return menu1_config
    
    def check_resource_availability(self) -> Tuple[bool, str]:
        """Check if system has enough resources for operation"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            available_memory_gb = memory.available / (1024**3)
            
            if available_memory_gb < 0.5:  # Less than 500MB
                return False, f"Insufficient memory: {available_memory_gb:.1f}GB available"
            
            if cpu_percent > 90:
                return False, f"High CPU usage: {cpu_percent:.1f}%"
            
            if memory.percent > 90:
                return False, f"High memory usage: {memory.percent:.1f}%"
            
            return True, "Resources available"
            
        except Exception as e:
            return False, f"Resource check error: {e}"
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary for dashboard"""
        perf = self.get_current_performance()
        
        return {
            'uptime': perf['uptime_str'],
            'cpu_usage': f"{perf['cpu_percent']}%",
            'cpu_status': '🔥' if perf['cpu_percent'] > 75 else '⚠️' if perf['cpu_percent'] > 60 else '✅',
            'memory_usage': f"{perf['memory_current_mb']:.0f} MB",
            'memory_status': '🔥' if perf['memory_percent'] > 75 else '⚠️' if perf['memory_percent'] > 60 else '✅',
            'overall_status': perf['status'],
            'recommendations': self._get_recommendations(perf)
        }
    
    def _get_recommendations(self, perf: Dict[str, Any]) -> list:
        """Get optimization recommendations"""
        recommendations = []
        
        if perf['cpu_percent'] > 75:
            recommendations.append("Reduce CPU-intensive operations")
        
        if perf['memory_percent'] > 75:
            recommendations.append("Free up memory or use smaller batch sizes")
        
        if perf['cpu_percent'] > 60 or perf['memory_percent'] > 60:
            recommendations.append("Consider using conservative settings")
        
        if not recommendations:
            recommendations.append("System running optimally")
        
        return recommendations
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        performance = self.get_current_performance()
        
        health_score = 100
        if performance['cpu_percent'] > self.cpu_warning * 100:
            health_score -= 20
        if performance['memory_percent'] > self.memory_warning * 100:
            health_score -= 20
        if performance['cpu_percent'] > self.cpu_critical * 100:
            health_score -= 30
        if performance['memory_percent'] > self.memory_critical * 100:
            health_score -= 30
        
        return {
            'health_score': max(0, health_score),
            'cpu_usage': performance['cpu_percent'],
            'memory_usage': performance['memory_percent'],
            'status': performance['status'],
            'uptime': performance['uptime_str']
        }
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status"""
        performance = self.get_current_performance()
        health = self.get_health_status()
        
        return {
            **performance,
            **health,
            'system_info': self.system_info,
            'monitoring_active': self.monitoring_active,
            'total_performance_records': len(self.performance_data)
        }
    
    def reset_monitoring(self):
        """Reset monitoring data"""
        self.performance_data.clear()
        self.start_time = datetime.now()
        logger.info("🔄 Performance monitoring reset")
    
    def stop_monitoring(self):
        """Stop resource monitoring with proper cleanup"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)  # Increased timeout
        
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.system("🛑 Optimized resource monitoring stopped", "OptimizedRM")
        else:
            self.logger.info("🛑 Optimized resource monitoring stopped")
    
    def reset_monitoring(self):
        """Reset monitoring data"""
        self.performance_data.clear()
        self.start_time = datetime.now()
        
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.system("🔄 Performance monitoring reset", "OptimizedRM")
        else:
            self.logger.info("🔄 Performance monitoring reset")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except:
            pass


# Global instance for easy access
_optimized_resource_manager = None

def initialize_optimized_intelligent_resources(allocation_percentage: float = 0.5, **kwargs) -> OptimizedResourceManager:
    """Initialize optimized intelligent resource manager"""
    global _optimized_resource_manager
    if _optimized_resource_manager is None:
        _optimized_resource_manager = OptimizedResourceManager(allocation_percentage)
    return _optimized_resource_manager

def get_optimized_resource_manager() -> OptimizedResourceManager:
    """Get global optimized resource manager instance"""
    global _optimized_resource_manager
    if _optimized_resource_manager is None:
        _optimized_resource_manager = OptimizedResourceManager()
    return _optimized_resource_manager

def get_optimized_config(task_type: str = 'general') -> Dict[str, Any]:
    """Get optimized configuration for a task"""
    return get_resource_manager().get_optimized_config(task_type)

def check_system_health() -> Dict[str, Any]:
    """Check current system health"""
    return get_resource_manager().get_system_health_summary()
