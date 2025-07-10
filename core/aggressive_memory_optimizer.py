"""
🚀 NICEGOLD ENTERPRISE PROJECTP - AGGRESSIVE MEMORY OPTIMIZER
Advanced Memory Management for Enterprise Trading Systems

This module provides aggressive memory optimization and management
for high-performance trading pipeline operations.

Author: NICEGOLD Enterprise Team
Date: July 8, 2025
Version: 2.0 DIVINE EDITION
"""

import gc
import os
import psutil
import threading
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import warnings
import logging
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_unified_logger()

class AggressiveMemoryOptimizer:
    """
    🚀 AGGRESSIVE MEMORY OPTIMIZER
    
    Enterprise-grade memory management system for optimal
    performance in trading pipeline operations.
    """
    
    def __init__(self, target_usage_percent: float = 80.0):
        """
        Initialize the Aggressive Memory Optimizer
        
        Args:
            target_usage_percent: Target memory usage percentage (default: 80%)
        """
        self.target_usage_percent = target_usage_percent
        self.process = psutil.Process()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.memory_stats = {
            'peak_usage': 0,
            'current_usage': 0,
            'optimizations_performed': 0,
            'gc_collections': 0
        }
        
        logger.info(f"🚀 Aggressive Memory Optimizer initialized (target: {target_usage_percent}%)")
        
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get comprehensive memory information
        
        Returns:
            Dictionary with memory statistics
        """
        virtual_memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        return {
            'system_total': virtual_memory.total,
            'system_available': virtual_memory.available,
            'system_used': virtual_memory.used,
            'system_percent': virtual_memory.percent,
            'process_rss': process_memory.rss,
            'process_vms': process_memory.vms,
            'process_percent': self.process.memory_percent(),
            'target_percent': self.target_usage_percent
        }
    
    def optimize_memory(self, aggressive: bool = True) -> Dict[str, Any]:
        """
        Perform aggressive memory optimization
        
        Args:
            aggressive: Whether to perform aggressive optimization
            
        Returns:
            Dictionary with optimization results
        """
        initial_memory = self.get_memory_info()
        
        # Perform garbage collection
        if aggressive:
            # Multiple GC passes for aggressive optimization
            for i in range(3):
                collected = gc.collect()
                self.memory_stats['gc_collections'] += collected
                
        else:
            # Single GC pass for standard optimization
            collected = gc.collect()
            self.memory_stats['gc_collections'] += collected
        
        # Clear unnecessary caches
        self._clear_caches()
        
        # Update statistics
        final_memory = self.get_memory_info()
        self.memory_stats['optimizations_performed'] += 1
        self.memory_stats['current_usage'] = final_memory['process_percent']
        
        if final_memory['process_percent'] > self.memory_stats['peak_usage']:
            self.memory_stats['peak_usage'] = final_memory['process_percent']
        
        optimization_results = {
            'success': True,
            'memory_freed_mb': (initial_memory['process_rss'] - final_memory['process_rss']) / 1024 / 1024,
            'memory_percent_before': initial_memory['process_percent'],
            'memory_percent_after': final_memory['process_percent'],
            'optimization_type': 'aggressive' if aggressive else 'standard',
            'gc_collected': collected if not aggressive else 'multiple_passes'
        }
        
        logger.info(f"🧹 Memory optimization completed: {optimization_results['memory_freed_mb']:.2f}MB freed")
        
        return optimization_results
    
    def _clear_caches(self):
        """Clear various Python caches"""
        try:
            # Clear warnings registry
            if hasattr(warnings, 'filters'):
                warnings.resetwarnings()
            
            # Clear import caches
            if hasattr(gc, 'get_stats'):
                gc.set_debug(0)
                
        except Exception as e:
            logger.warning(f"⚠️ Cache clearing partially failed: {e}")
    
    @contextmanager
    def memory_managed_context(self, optimize_on_exit: bool = True):
        """
        Context manager for memory-managed operations
        
        Args:
            optimize_on_exit: Whether to optimize memory on context exit
        """
        initial_memory = self.get_memory_info()
        logger.info(f"🚀 Entering memory-managed context (current: {initial_memory['process_percent']:.1f}%)")
        
        try:
            yield self
        finally:
            if optimize_on_exit:
                self.optimize_memory(aggressive=True)
                final_memory = self.get_memory_info()
                logger.info(f"🏁 Exiting memory-managed context (final: {final_memory['process_percent']:.1f}%)")
    
    def start_monitoring(self, interval: float = 30.0):
        """
        Start continuous memory monitoring
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            logger.warning("⚠️ Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"📊 Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("🛑 Memory monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Memory monitoring loop"""
        while self.monitoring_active:
            try:
                memory_info = self.get_memory_info()
                current_percent = memory_info['process_percent']
                
                # Check if optimization is needed
                if current_percent > self.target_usage_percent:
                    logger.warning(f"🚨 Memory usage high: {current_percent:.1f}% > {self.target_usage_percent}%")
                    self.optimize_memory(aggressive=True)
                
                # Update current usage
                self.memory_stats['current_usage'] = current_percent
                
                # Log periodic status
                if self.memory_stats['optimizations_performed'] % 10 == 0:
                    logger.info(f"📊 Memory status: {current_percent:.1f}% (target: {self.target_usage_percent}%)")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"❌ Memory monitoring error: {e}")
                time.sleep(interval)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics
        
        Returns:
            Dictionary with optimization statistics
        """
        return {
            **self.memory_stats,
            'monitoring_active': self.monitoring_active,
            'target_usage_percent': self.target_usage_percent,
            'current_memory_info': self.get_memory_info()
        }
    
    def force_cleanup(self):
        """
        Force aggressive cleanup of all resources
        """
        logger.info("🧹 Performing force cleanup...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Multiple aggressive optimizations
        for i in range(5):
            self.optimize_memory(aggressive=True)
            time.sleep(0.1)
        
        # Final memory report
        final_memory = self.get_memory_info()
        logger.info(f"🏁 Force cleanup completed (final usage: {final_memory['process_percent']:.1f}%)")
        
        return final_memory


# Global instance for easy access
_global_optimizer = None

def get_memory_optimizer(target_usage_percent: float = 80.0) -> AggressiveMemoryOptimizer:
    """
    Get the global memory optimizer instance
    
    Args:
        target_usage_percent: Target memory usage percentage
        
    Returns:
        AggressiveMemoryOptimizer instance
    """
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = AggressiveMemoryOptimizer(target_usage_percent)
    
    return _global_optimizer


def optimize_memory_now(aggressive: bool = True) -> Dict[str, Any]:
    """
    Convenient function to optimize memory immediately
    
    Args:
        aggressive: Whether to perform aggressive optimization
        
    Returns:
        Optimization results
    """
    optimizer = get_memory_optimizer()
    return optimizer.optimize_memory(aggressive=aggressive)


@contextmanager
def memory_managed_operation(optimize_on_exit: bool = True):
    """
    Context manager for memory-managed operations
    
    Args:
        optimize_on_exit: Whether to optimize memory on exit
    """
    optimizer = get_memory_optimizer()
    with optimizer.memory_managed_context(optimize_on_exit=optimize_on_exit):
        yield optimizer


def start_memory_monitoring(interval: float = 30.0):
    """
    Start global memory monitoring
    
    Args:
        interval: Monitoring interval in seconds
    """
    optimizer = get_memory_optimizer()
    optimizer.start_monitoring(interval=interval)


def stop_memory_monitoring():
    """Stop global memory monitoring"""
    optimizer = get_memory_optimizer()
    optimizer.stop_monitoring()


def get_memory_stats() -> Dict[str, Any]:
    """
    Get current memory statistics
    
    Returns:
        Dictionary with memory statistics
    """
    optimizer = get_memory_optimizer()
    return optimizer.get_optimization_stats()


# Example usage and testing
if __name__ == "__main__":
    print("🚀 NICEGOLD AGGRESSIVE MEMORY OPTIMIZER - TEST MODE")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = AggressiveMemoryOptimizer(target_usage_percent=75.0)
    
    # Show initial memory
    initial_memory = optimizer.get_memory_info()
    print(f"📊 Initial memory usage: {initial_memory['process_percent']:.1f}%")
    
    # Perform optimization
    results = optimizer.optimize_memory(aggressive=True)
    print(f"🧹 Optimization results: {results}")
    
    # Show final memory
    final_memory = optimizer.get_memory_info()
    print(f"📊 Final memory usage: {final_memory['process_percent']:.1f}%")
    
    # Test context manager
    print("\n🔄 Testing memory-managed context...")
    with optimizer.memory_managed_context():
        print("✅ Inside memory-managed context")
        # Simulate some work
        test_list = [i for i in range(10000)]
        print(f"📊 Created test data: {len(test_list)} items")
    
    print("🎉 Aggressive Memory Optimizer test completed!")
