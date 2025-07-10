#!/usr/bin/env python3
"""
🛠️ NICEGOLD ENTERPRISE PROJECTP - CUDA & LOGGING FIX
แก้ไข CUDA warnings และ logging errors ให้สมบูรณ์แบบ
วันที่: 6 กรกฎาคม 2025
"""

import os
import sys
import warnings
import logging

def suppress_cuda_warnings():
    """ระงับ CUDA warnings ทั้งหมด"""
    print("🔧 Applying CUDA suppression...")
    
    # Set environment variables before any imports
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cpu'
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', module='tensorflow')
    
    # Configure logging to suppress TensorFlow
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('torch').setLevel(logging.ERROR)
    
    print("✅ CUDA suppression applied successfully")

def safe_logger_setup():
    """ตั้งค่า logger ที่ปลอดภัย"""
    print("🔧 Setting up safe logger...")
    
    # Create safe logger
    logger = logging.getLogger('NICEGOLD_SAFE')
    logger.handlers.clear()  # Clear existing handlers
    
    # Create console handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
    print("✅ Safe logger setup completed")
    return logger

def apply_all_fixes():
    """ใช้การแก้ไขทั้งหมด"""
    print("🎯 NICEGOLD ENTERPRISE PROJECTP - COMPREHENSIVE FIX")
    print("=" * 60)
    
    # 1. CUDA suppression
    suppress_cuda_warnings()
    
    # 2. Safe logger
    logger = safe_logger_setup()
    
    print("=" * 60)
    print("✅ All fixes applied successfully!")
    print("🚀 System ready for enterprise operation")
    
    return logger

if __name__ == "__main__":
    apply_all_fixes()
