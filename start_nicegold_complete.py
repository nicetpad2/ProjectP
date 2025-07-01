#!/usr/bin/env python3
"""
🚀 NICEGOLD ProjectP - Quick Start
สคริปต์เริ่มต้นระบบอย่างรวดเร็ว
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# เพิ่ม path ของโปรเจค
project_path = "/content/drive/MyDrive/ProjectP"
sys.path.append(project_path)
os.chdir(project_path)

print("🚀 เริ่มต้น NICEGOLD ProjectP...")
print("="*60)

try:
    # เริ่มระบบหลัก
    exec(open("ProjectP.py").read())
except KeyboardInterrupt:
    print("\n👋 ขอบคุณที่ใช้งาน NICEGOLD ProjectP!")
except Exception as e:
    print(f"❌ ข้อผิดพลาด: {e}")
    print("💡 ลองรันคำสั่ง: python ProjectP.py")
