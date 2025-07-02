#!/usr/bin/env python3
"""
🔧 Optimized Resource Manager Configuration
Reduces CPU and memory usage for better performance
"""

import os
import sys
import psutil
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

class OptimizedResourceManager:
    """Lightweight resource manager with reduced overhead"""
    
    def __init__(self):
        self.monitoring_active = False
        self.monitor_thread = None
        self.current_performance = {}
        self.last_update = datetime.now()
        self.update_interval = 5.0  # Update every 5 seconds instead of 0.5
        
    def get_current_performance(self) -> Dict[str, Any]:
        """Get current system performance with caching"""
        
        # Only update if enough time has passed
        now = datetime.now()
        if (now - self.last_update).total_seconds() < self.update_interval:
            return self.current_performance
        
        try:
            # Get basic system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Short interval
            memory = psutil.virtual_memory()
            
            self.current_performance = {
                'cpu_percent': round(cpu_percent, 1),
                'memory': {
                    'percent': round(memory.percent, 1),
                    'used_mb': round(memory.used / (1024**2), 1),
                    'available_mb': round(memory.available / (1024**2), 1)
                },
                'timestamp': now.isoformat(),
                'uptime_seconds': int(time.time() - psutil.boot_time())
            }
            
            self.last_update = now
            return self.current_performance
            
        except Exception as e:
            print(f"Resource monitoring error: {e}")
            return self.current_performance or {}
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimized configuration based on current resources"""
        
        perf = self.get_current_performance()
        cpu_percent = perf.get('cpu_percent', 0)
        memory_percent = perf.get('memory', {}).get('percent', 0)
        
        # Conservative configuration to reduce resource usage
        config = {
            'data_processing': {
                'chunk_size': 2000 if memory_percent > 70 else 4000,
                'batch_size': 16 if cpu_percent > 80 else 32,
                'n_jobs': max(1, psutil.cpu_count() // 2),  # Use half of available cores
                'low_memory': memory_percent > 75
            },
            'feature_selection': {
                'max_features': 20 if memory_percent > 70 else 30,
                'sample_size': 5000 if memory_percent > 70 else 10000,
                'n_trials': 20 if cpu_percent > 80 else 50
            },
            'cnn_lstm': {
                'batch_size': 16 if memory_percent > 70 else 32,
                'epochs': 50 if cpu_percent > 80 else 100,
                'patience': 5,
                'validation_split': 0.2
            },
            'dqn': {
                'batch_size': 16 if memory_percent > 70 else 32,
                'memory_size': 5000 if memory_percent > 70 else 10000,
                'episodes': 50 if cpu_percent > 80 else 100
            }
        }
        
        return config
    
    def start_lightweight_monitoring(self):
        """Start lightweight background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitoring_loop(self):
        """Lightweight monitoring loop"""
        while self.monitoring_active:
            try:
                # Just update the performance cache
                self.get_current_performance()
                time.sleep(self.update_interval)
            except Exception:
                pass

def initialize_optimized_resources():
    """Initialize optimized resource manager"""
    try:
        resource_manager = OptimizedResourceManager()
        resource_manager.start_lightweight_monitoring()
        return resource_manager
    except Exception as e:
        print(f"Warning: Could not initialize resource manager: {e}")
        return None

if __name__ == "__main__":
    # Test the optimized resource manager
    rm = initialize_optimized_resources()
    if rm:
        print("✅ Optimized Resource Manager initialized")
        perf = rm.get_current_performance()
        print(f"📊 CPU: {perf.get('cpu_percent', 0)}%")
        print(f"📊 Memory: {perf.get('memory', {}).get('percent', 0)}%")
        
        config = rm.get_optimization_config()
        print(f"🔧 Optimized batch size: {config['data_processing']['batch_size']}")
        print(f"🔧 Optimized chunk size: {config['data_processing']['chunk_size']}")
    else:
        print("❌ Resource manager initialization failed")
