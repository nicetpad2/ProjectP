#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - TENSORFLOW CONFIGURATION
Enterprise TensorFlow Configuration and Warning Suppression

ไฟล์นี้จัดการการตั้งค่า TensorFlow สำหรับ Enterprise Production
รวมถึงการลด warnings และการปรับ performance ให้เหมาะสม

เวอร์ชัน: 1.0 Enterprise Edition
วันที่: 11 กรกฎาคม 2025
สถานะ: Production Ready
"""

import os
import warnings
import logging
from typing import Dict, Any

def suppress_tensorflow_warnings():
    """Suppress TensorFlow warnings for production"""
    try:
        # ✅ PHASE 1: Environment Variables (Before TensorFlow Import)
        
        # Basic TensorFlow logging control
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only ERROR messages
        
        # Disable specific problematic optimizations
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN warnings
        os.environ['TF_DISABLE_MKL'] = '1'          # Disable MKL warnings
        
        # Force CPU mode (Enterprise Production Safe)
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
        
        # Thread optimization
        cpu_count = os.cpu_count() or 1
        os.environ['TF_NUM_INTEROP_THREADS'] = str(min(8, cpu_count))
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(min(8, cpu_count))
        
        # Additional warning suppression
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_DETERMINISTIC_OPS'] = '1'  # More predictable behavior
        
        print("✅ TensorFlow environment configured for Enterprise Production")
        
        # ✅ PHASE 2: Python Warnings (General)
        
        # Filter specific TensorFlow warnings
        warnings.filterwarnings('ignore', message='.*use_unbounded_threadpool.*')
        warnings.filterwarnings('ignore', message='.*NodeDef mentions attribute.*')
        warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
        
        print("✅ Python warnings filtered for TensorFlow")
        
        # ✅ PHASE 3: TensorFlow-Specific Configuration (After Import)
        
        try:
            import tensorflow as tf
            
            # Set TensorFlow logging level to ERROR only
            tf.get_logger().setLevel('ERROR')
            
            # Disable autograph verbosity
            tf.autograph.set_verbosity(0)
            
            # Additional TensorFlow configuration
            tf.config.set_soft_device_placement(True)
            
            # Disable XLA JIT compilation (reduces warnings)
            tf.config.optimizer.set_jit(False)
            
            print("✅ TensorFlow internal configuration optimized")
            
        except ImportError:
            print("ℹ️ TensorFlow not available for advanced configuration")
            
        return True
        
    except Exception as e:
        print(f"⚠️ Warning: Some TensorFlow optimization failed: {e}")
        return False

def tune_cpu_performance():
    """Tune CPU performance for optimal training"""
    try:
        import multiprocessing
        
        cpu_count = multiprocessing.cpu_count()
        
        # ✅ OPTIMAL THREAD CONFIGURATION
        
        # OpenMP threads (for BLAS operations)
        os.environ['OMP_NUM_THREADS'] = str(min(8, cpu_count))
        
        # MKL threads (Intel Math Kernel Library)
        os.environ['MKL_NUM_THREADS'] = str(min(8, cpu_count))
        
        # NumExpr threads (for pandas/numpy operations)
        os.environ['NUMEXPR_NUM_THREADS'] = str(min(8, cpu_count))
        
        # OpenBLAS threads
        os.environ['OPENBLAS_NUM_THREADS'] = str(min(8, cpu_count))
        
        # Additional performance tuning
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(min(8, cpu_count))
        
        print(f"✅ CPU performance tuned for {min(8, cpu_count)} threads (out of {cpu_count} available)")
        
        return {
            'cpu_cores_total': cpu_count,
            'cpu_cores_used': min(8, cpu_count),
            'performance_tuned': True
        }
        
    except Exception as e:
        print(f"⚠️ Warning: CPU performance tuning failed: {e}")
        return {'performance_tuned': False}

def optimize_memory_usage():
    """Optimize memory usage for large datasets"""
    try:
        import gc
        import pandas as pd
        
        # ✅ GARBAGE COLLECTION OPTIMIZATION
        
        # Force garbage collection
        gc.collect()
        
        # Adjust garbage collection thresholds for better memory management
        gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        # ✅ PANDAS OPTIMIZATION
        
        # Optimize pandas memory usage
        pd.options.mode.chained_assignment = None  # Suppress chained assignment warnings
        pd.set_option('mode.copy_on_write', False)  # Reduce memory copying
        
        # Set optimal chunk sizes for large datasets
        OPTIMAL_CHUNK_SIZE = 10000  # 10K rows per chunk
        
        print("✅ Memory usage optimized for large datasets")
        
        return {
            'chunk_size': OPTIMAL_CHUNK_SIZE,
            'memory_optimized': True,
            'gc_optimized': True
        }
        
    except Exception as e:
        print(f"⚠️ Warning: Memory optimization failed: {e}")
        return {'memory_optimized': False}

def configure_enterprise_tensorflow():
    """Complete Enterprise TensorFlow configuration"""
    try:
        print("🚀 Starting Enterprise TensorFlow Configuration...")
        
        # Step 1: Suppress warnings
        warnings_result = suppress_tensorflow_warnings()
        
        # Step 2: Tune CPU performance  
        cpu_result = tune_cpu_performance()
        
        # Step 3: Optimize memory usage
        memory_result = optimize_memory_usage()
        
        # Summary
        config_summary = {
            'warnings_suppressed': warnings_result,
            'cpu_optimized': cpu_result.get('performance_tuned', False),
            'memory_optimized': memory_result.get('memory_optimized', False),
            'cpu_cores_used': cpu_result.get('cpu_cores_used', 1),
            'chunk_size': memory_result.get('chunk_size', 1000),
            'enterprise_ready': True
        }
        
        print("🎉 Enterprise TensorFlow Configuration Complete!")
        print(f"   • Warnings: {'✅ Suppressed' if warnings_result else '❌ Failed'}")
        print(f"   • CPU: {'✅ Optimized' if cpu_result.get('performance_tuned') else '❌ Failed'}")
        print(f"   • Memory: {'✅ Optimized' if memory_result.get('memory_optimized') else '❌ Failed'}")
        
        return config_summary
        
    except Exception as e:
        print(f"❌ Enterprise TensorFlow Configuration failed: {e}")
        return {'enterprise_ready': False, 'error': str(e)}

# 🚀 AUTO-CONFIGURATION
# This will run automatically when the module is imported
if __name__ == "__main__":
    # Test the configuration
    result = configure_enterprise_tensorflow()
    print(f"Configuration result: {result}")
else:
    # Auto-configure when imported
    suppress_tensorflow_warnings()
    print("✅ TensorFlow auto-configuration applied") 