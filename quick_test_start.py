#!/usr/bin/env python3
"""
🚀 NICEGOLD ProjectP - Quick Test & Start
ทดสอบและเริ่มต้นระบบอย่างรวดเร็ว
"""
import os
import sys

# เตรียม environment
os.chdir('/content/drive/MyDrive/ProjectP')
sys.path.insert(0, '/content/drive/MyDrive/ProjectP')

print("🎯 NICEGOLD ProjectP - Quick Test & Start")
print("="*50)

# Test 1: Basic Python
try:
    import numpy as np
    import pandas as pd
    print(f"✅ NumPy {np.__version__}")
    print(f"✅ Pandas {pd.__version__}")
except Exception as e:
    print(f"❌ Basic packages error: {e}")
    exit(1)

# Test 2: Core import
try:
    from core.config import load_enterprise_config
    print("✅ Core config imported")
except Exception as e:
    print(f"❌ Core config error: {e}")

# Test 3: Protection system
try:
    from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
    protection = EnterpriseMLProtectionSystem()
    print("✅ Enterprise ML Protection initialized")
except Exception as e:
    print(f"❌ Protection system error: {e}")

print("\n🎉 ระบบทดสอบเสร็จสิ้น!")
print("\n🚀 เริ่มต้นระบบหลัก...")
print("="*50)

# Start main system
try:
    exec(open('ProjectP.py').read())
except KeyboardInterrupt:
    print("\n👋 ขอบคุณที่ใช้งาน NICEGOLD ProjectP!")
except Exception as e:
    print(f"\n❌ ข้อผิดพลาด: {e}")
    print("💡 ลองรัน: python ProjectP.py")
