#!/usr/bin/env python3
"""
🚫 COMPREHENSIVE CUDA ELIMINATION SYSTEM
Completely eliminates all CUDA-related warnings and errors
"""

import os
import sys
import warnings
import logging
import contextlib

# PHASE 1: Environment-level CUDA suppression (before ANY imports)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=""'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Additional comprehensive CUDA suppression
os.environ['CUDA_CACHE_DISABLE'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Additional aggressive suppressions
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_DISABLE_CUDA_MALLOC'] = '1'
os.environ['NVIDIA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Suppress cuFFT, cuDNN, cuBLAS registration specifically
os.environ['TF_CPP_VMODULE'] = 'gpu_device=0,gpu_kernel=0,gpu_util=0'
os.environ['TF_DISABLE_GPU'] = '1'

# PHASE 2: Python warnings suppression
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*cuDNN.*')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')
warnings.filterwarnings('ignore', message='.*cuFFT.*')
warnings.filterwarnings('ignore', message='.*Unable to register.*')
warnings.filterwarnings('ignore', message='.*XLA.*')
warnings.filterwarnings('ignore', message='.*GPU.*')
warnings.filterwarnings('ignore', message='.*tensorflow.*')

# PHASE 3: Logging suppression
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorboard').setLevel(logging.ERROR)

# PHASE 4: Context manager for complete stderr suppression
@contextlib.contextmanager
def suppress_all_cuda_output():
    """Context manager to suppress all CUDA-related output"""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        try:
            sys.stderr = devnull
            sys.stdout = devnull
            yield
        finally:
            sys.stderr = old_stderr
            sys.stdout = old_stdout

def apply_cuda_suppression():
    """Apply all CUDA suppression measures"""
    # Additional runtime suppression
    try:
        import tensorflow as tf
        if hasattr(tf, 'config'):
            tf.config.set_visible_devices([], 'GPU')
    except:
        pass
    
    try:
        import torch
        if hasattr(torch, 'cuda'):
            torch.cuda.is_available = lambda: False
    except:
        pass

# Auto-apply on import
apply_cuda_suppression()
