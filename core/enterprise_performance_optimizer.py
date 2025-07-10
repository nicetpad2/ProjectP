"""
⚡ Advanced Performance Optimization System for NICEGOLD ProjectP
Enterprise-grade performance monitoring, optimization, and resource management
"""

import psutil
import time
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import threading
import queue
import gc
from datetime import datetime
import json
import warnings
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus


class PerformanceProfiler:
    """
    Advanced performance profiler for monitoring execution time, memory usage, and system resources
    """
    
    def __init__(self, config=None):
        """Initialize performance profiler with optional configuration"""
        self.config = config or {}
        self.logger = get_unified_logger()
        self.profiles = {}
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'gpu_usage': []
        }
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Apply configuration if provided
        if isinstance(config, dict):
            self.target_memory_usage = config.get('target_memory_usage', 80)
            self.monitoring_interval = config.get('monitoring_interval', 5)
            self.performance_threshold = config.get('performance_threshold', 0.70)
        else:
            # Default configuration
            self.target_memory_usage = 80
            self.monitoring_interval = 5
            self.performance_threshold = 0.70
        
    def profile_function(self, func_name: str = None):
        """
        Decorator for profiling function performance
        
        Args:
            func_name: Optional custom name for the function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or f"{func.__module__}.{func.__name__}"
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Store profile data
                    if name not in self.profiles:
                        self.profiles[name] = {
                            'call_count': 0,
                            'total_time': 0.0,
                            'avg_time': 0.0,
                            'min_time': float('inf'),
                            'max_time': 0.0,
                            'total_memory_delta': 0.0,
                            'avg_memory_delta': 0.0,
                            'last_execution': datetime.now().isoformat()
                        }
                    
                    profile = self.profiles[name]
                    profile['call_count'] += 1
                    profile['total_time'] += execution_time
                    profile['avg_time'] = profile['total_time'] / profile['call_count']
                    profile['min_time'] = min(profile['min_time'], execution_time)
                    profile['max_time'] = max(profile['max_time'], execution_time)
                    profile['total_memory_delta'] += memory_delta
                    profile['avg_memory_delta'] = profile['total_memory_delta'] / profile['call_count']
                    profile['last_execution'] = datetime.now().isoformat()
                    
                    # Log performance if execution time is significant
                    if execution_time > 1.0:  # Log if > 1 second
                        self.logger.info(f"Performance: {name} took {execution_time:.3f}s, memory delta: {memory_delta:.2f}MB")
                    
                    return result
                    
                except Exception as e:
                    # Log error but don't interfere with function execution
                    self.logger.error(f"Error profiling {name}: {str(e)}")
                    raise
                    
            return wrapper
        return decorator
    
    def start_system_monitoring(self, interval: float = 1.0):
        """
        Start continuous system monitoring
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("Started system performance monitoring")
    
    def stop_system_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Stopped system performance monitoring")
    
    def _monitor_system(self, interval: float):
        """Internal method for system monitoring"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.system_metrics['cpu_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': cpu_percent
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_metrics['memory_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'used_gb': memory.used / 1024**3,
                    'available_gb': memory.available / 1024**3,
                    'percent': memory.percent
                })
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.system_metrics['disk_io'].append({
                        'timestamp': datetime.now().isoformat(),
                        'read_mb': disk_io.read_bytes / 1024**2,
                        'write_mb': disk_io.write_bytes / 1024**2
                    })
                
                # Keep only last 1000 entries to prevent memory bloat
                for metric in self.system_metrics:
                    if len(self.system_metrics[metric]) > 1000:
                        self.system_metrics[metric] = self.system_metrics[metric][-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {str(e)}")
                time.sleep(interval)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Performance report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'function_profiles': self.profiles.copy(),
            'system_summary': {},
            'recommendations': [],
            'performance_score': 0.0
        }
        
        try:
            # System summary
            if self.system_metrics['cpu_usage']:
                recent_cpu = [m['value'] for m in self.system_metrics['cpu_usage'][-60:]]  # Last 60 measurements
                report['system_summary']['cpu'] = {
                    'current': recent_cpu[-1] if recent_cpu else 0,
                    'average': np.mean(recent_cpu) if recent_cpu else 0,
                    'max': np.max(recent_cpu) if recent_cpu else 0
                }
            
            if self.system_metrics['memory_usage']:
                recent_memory = [m['percent'] for m in self.system_metrics['memory_usage'][-60:]]
                report['system_summary']['memory'] = {
                    'current_percent': recent_memory[-1] if recent_memory else 0,
                    'average_percent': np.mean(recent_memory) if recent_memory else 0,
                    'max_percent': np.max(recent_memory) if recent_memory else 0
                }
            
            # Performance recommendations
            recommendations = []
            
            # CPU recommendations
            avg_cpu = report['system_summary'].get('cpu', {}).get('average', 0)
            if avg_cpu > 80:
                recommendations.append("High CPU usage detected - consider optimizing computationally intensive operations")
            
            # Memory recommendations
            avg_memory = report['system_summary'].get('memory', {}).get('average_percent', 0)
            if avg_memory > 80:
                recommendations.append("High memory usage detected - implement memory optimization strategies")
            
            # Function performance recommendations
            for func_name, profile in self.profiles.items():
                if profile['avg_time'] > 5.0:  # Functions taking > 5 seconds on average
                    recommendations.append(f"Optimize {func_name} - average execution time: {profile['avg_time']:.2f}s")
                
                if profile['avg_memory_delta'] > 100:  # Functions using > 100MB on average
                    recommendations.append(f"Memory optimization needed for {func_name} - average memory delta: {profile['avg_memory_delta']:.2f}MB")
            
            report['recommendations'] = recommendations
            
            # Calculate performance score (0-100)
            cpu_score = max(0, 100 - avg_cpu)
            memory_score = max(0, 100 - avg_memory)
            function_score = 100 if not self.profiles else min(100, max(0, 100 - len([p for p in self.profiles.values() if p['avg_time'] > 1.0]) * 10))
            
            report['performance_score'] = (cpu_score + memory_score + function_score) / 3
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            report['recommendations'].append(f"Performance reporting error: {str(e)}")
        
        return report

class MemoryOptimizer:
    """
    Advanced memory optimization utilities for large-scale data processing
    """
    
    def __init__(self):
        self.logger = get_unified_logger()
        
    def optimize_dataframe(self, df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage through dtype optimization
        
        Args:
            df: DataFrame to optimize
            aggressive: Whether to use aggressive optimization (may lose precision)
            
        Returns:
            Optimized DataFrame
        """
        try:
            original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                col_min = df[col].min()
                col_max = df[col].max()
                
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
            
            for col in df.select_dtypes(include=['float64']).columns:
                if aggressive or df[col].isna().sum() == 0:  # Only if no NaN values or aggressive mode
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object columns
            for col in df.select_dtypes(include=['object']).columns:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                
                if num_unique_values / num_total_values < 0.5:  # If less than 50% unique values
                    df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
            memory_saved = original_memory - optimized_memory
            
            self.logger.info(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB (saved {memory_saved:.2f}MB)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error optimizing DataFrame: {str(e)}")
            return df
    
    def chunked_processing(self, data: pd.DataFrame, chunk_size: int = 10000, 
                          process_func: Callable = None) -> List[Any]:
        """
        Process large DataFrame in chunks to manage memory usage
        
        Args:
            data: Large DataFrame to process
            chunk_size: Number of rows per chunk
            process_func: Function to apply to each chunk
            
        Returns:
            List of processed results
        """
        results = []
        
        try:
            num_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(data))
                
                chunk = data.iloc[start_idx:end_idx].copy()
                
                if process_func:
                    result = process_func(chunk)
                    results.append(result)
                
                # Force garbage collection after processing each chunk
                del chunk
                gc.collect()
                
                if i % 10 == 0:  # Log progress every 10 chunks
                    self.logger.info(f"Processed chunk {i+1}/{num_chunks}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in chunked processing: {str(e)}")
            return results

class CacheManager:
    """
    Intelligent caching system for frequently accessed data and computations
    """
    
    def __init__(self, max_cache_size_mb: int = 500):
        self.cache = {}
        self.access_times = {}
        self.cache_sizes = {}
        self.max_cache_size_mb = max_cache_size_mb
        self.current_cache_size_mb = 0
        self.logger = get_unified_logger()
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key for data"""
        if isinstance(data, pd.DataFrame):
            return f"df_{hash(str(data.shape) + str(data.columns.tolist()))}"
        elif isinstance(data, np.ndarray):
            return f"np_{hash(str(data.shape) + str(data.dtype))}"
        else:
            return f"obj_{hash(str(data))}"
    
    def put(self, key: str, value: Any, size_mb: float = None) -> bool:
        """
        Add item to cache
        
        Args:
            key: Cache key
            value: Value to cache
            size_mb: Size of the value in MB (estimated if not provided)
            
        Returns:
            Success status
        """
        try:
            if size_mb is None:
                if isinstance(value, pd.DataFrame):
                    size_mb = value.memory_usage(deep=True).sum() / 1024**2
                elif isinstance(value, np.ndarray):
                    size_mb = value.nbytes / 1024**2
                else:
                    size_mb = 1.0  # Default estimate
            
            # Check if we need to free space
            while self.current_cache_size_mb + size_mb > self.max_cache_size_mb and self.cache:
                self._evict_lru()
            
            # Add to cache
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.cache_sizes[key] = size_mb
            self.current_cache_size_mb += size_mb
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding to cache: {str(e)}")
            return False
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()  # Update access time
            return self.cache[key]
        return None
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        lru_key = min(self.access_times, key=self.access_times.get)
        size_freed = self.cache_sizes.get(lru_key, 0)
        
        del self.cache[lru_key]
        del self.access_times[lru_key]
        del self.cache_sizes[lru_key]
        self.current_cache_size_mb -= size_freed
        
        self.logger.debug(f"Evicted cache entry {lru_key}, freed {size_freed:.2f}MB")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
        self.cache_sizes.clear()
        self.current_cache_size_mb = 0
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'items_count': len(self.cache),
            'total_size_mb': self.current_cache_size_mb,
            'max_size_mb': self.max_cache_size_mb,
            'utilization_percent': (self.current_cache_size_mb / self.max_cache_size_mb) * 100,
            'items': list(self.cache.keys())
        }

# Global instances for easy access
profiler = PerformanceProfiler()
memory_optimizer = MemoryOptimizer()
cache_manager = CacheManager()

# Convenience decorator
def performance_monitor(func_name: str = None):
    """Convenience decorator for performance monitoring"""
    return profiler.profile_function(func_name)

# Example usage
if __name__ == "__main__":
    # Start monitoring
    profiler.start_system_monitoring()
    
    # Example function with performance monitoring
    @performance_monitor("test_function")
    def test_expensive_operation():
        """Test function for performance monitoring"""
        time.sleep(2)
        data = np.random.random((1000, 1000))
        return np.sum(data)
    
    # Run test
    result = test_expensive_operation()
    
    # Generate report
    report = profiler.get_performance_report()
    
    print("⚡ PERFORMANCE OPTIMIZATION REPORT")
    print("=" * 50)
    print(f"Performance Score: {report['performance_score']:.1f}/100")
    print(f"CPU Usage: {report['system_summary'].get('cpu', {}).get('current', 0):.1f}%")
    print(f"Memory Usage: {report['system_summary'].get('memory', {}).get('current_percent', 0):.1f}%")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    profiler.stop_system_monitoring()
