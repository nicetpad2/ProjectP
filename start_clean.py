#!/usr/bin/env python3
"""
🛡️ NICEGOLD ProjectP - CUDA-Free Start
เริ่มระบบโดยไม่แสดง CUDA warnings
"""
import os
import sys
import warnings

# Comprehensive CUDA suppression
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=""'

# Suppress all warnings
warnings.filterwarnings('ignore')

# เตรียม environment
os.chdir('/content/drive/MyDrive/ProjectP')
sys.path.insert(0, '/content/drive/MyDrive/ProjectP')

print("🛡️ NICEGOLD ProjectP - CUDA-Free Mode")
print("="*50)
print("📝 หมายเหตุ: CUDA warnings ถูกปิดการแสดงผล")
print("💻 ระบบใช้ CPU mode (ปกติสำหรับ Google Colab)")
print("="*50)

# Test basic functionality
try:
    import numpy as np
    import pandas as pd
    print(f"✅ NumPy {np.__version__}")
    print(f"✅ Pandas {pd.__version__}")
except Exception as e:
    print(f"❌ Basic packages error: {e}")
    exit(1)

print("\n🚀 เริ่มต้นระบบหลัก...")
print("-" * 50)

# Start main system silently
try:
    exec(open('ProjectP.py').read())
except KeyboardInterrupt:
    print("\n👋 ขอบคุณที่ใช้งาน NICEGOLD ProjectP!")
except Exception as e:
    print(f"\n❌ ข้อผิดพลาด: {e}")
    print("💡 ลองรัน: python ProjectP.py")
