#!/usr/bin/env python3
"""
🚀 NICEGOLD ProjectP - Complete Installation Script
ติดตั้งและตั้งค่าระบบ NICEGOLD ProjectP ให้พร้อมใช้งาน
"""
import os
import sys
import subprocess
import time

def print_status(message, status="INFO"):
    """พิมพ์สถานะการติดตั้ง"""
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"{icons.get(status, 'ℹ️')} {message}")

def run_command(command, description=""):
    """รันคำสั่งและแสดงผล"""
    if description:
        print_status(f"กำลัง{description}...")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def install_dependencies():
    """ติดตั้ง dependencies หลัก"""
    print_status("🔧 เริ่มติดตั้งระบบ NICEGOLD ProjectP...")
    
    # อัปเกรด pip
    success, output = run_command("pip install --upgrade pip", "อัปเกรด pip")
    if success:
        print_status("pip อัปเกรดเสร็จเรียบร้อย", "SUCCESS")
    else:
        print_status("ไม่สามารถอัปเกรด pip ได้", "WARNING")
    
    # ติดตั้ง core packages
    core_packages = [
        "numpy==1.26.4",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "PyYAML>=6.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "colorama>=0.4.0",
        "optuna>=3.0.0"
    ]
    
    print_status("📦 ติดตั้ง core packages...")
    for package in core_packages:
        success, output = run_command(f"pip install {package}")
        if success:
            print_status(f"ติดตั้ง {package.split('==')[0].split('>=')[0]} สำเร็จ", "SUCCESS")
        else:
            print_status(f"ไม่สามารถติดตั้ง {package} ได้", "WARNING")
    
    # ML packages (optional)
    ml_packages = [
        "tensorflow>=2.15.0",
        "torch>=2.0.0",
        "shap>=0.40.0"
    ]
    
    print_status("🤖 ติดตั้ง ML packages (อาจใช้เวลานาน)...")
    for package in ml_packages:
        success, output = run_command(f"pip install {package}")
        if success:
            print_status(f"ติดตั้ง {package.split('>=')[0]} สำเร็จ", "SUCCESS")
        else:
            print_status(f"ไม่สามารถติดตั้ง {package} ได้ (ไม่จำเป็นสำหรับการทำงานพื้นฐาน)", "WARNING")

def verify_installation():
    """ตรวจสอบการติดตั้ง"""
    print_status("🔍 ตรวจสอบการติดตั้ง...")
    
    # ตรวจสอบ core imports
    test_code = '''
import sys
sys.path.append("/content/drive/MyDrive/ProjectP")

try:
    import numpy as np
    import pandas as pd
    import sklearn
    import scipy
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ Pandas: {pd.__version__}")
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    print(f"✅ SciPy: {scipy.__version__}")
    
    # Test ProjectP imports
    from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
    print("✅ Enterprise ML Protection System: OK")
    
    from core.config import load_enterprise_config
    print("✅ Core Config: OK")
    
    from core.logger import setup_logger
    print("✅ Core Logger: OK")
    
    print("\\n🎉 การติดตั้งสมบูรณ์! ระบบพร้อมใช้งาน")
    
except Exception as e:
    print(f"❌ ข้อผิดพลาด: {e}")
    exit(1)
'''
    
    success, output = run_command(f'python -c "{test_code}"')
    if success:
        print(output)
        return True
    else:
        print_status("การตรวจสอบล้มเหลว", "ERROR")
        print(output)
        return False

def create_startup_script():
    """สร้างสคริปต์เริ่มต้นระบบ"""
    startup_script = '''#!/usr/bin/env python3
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
    print("\\n👋 ขอบคุณที่ใช้งาน NICEGOLD ProjectP!")
except Exception as e:
    print(f"❌ ข้อผิดพลาด: {e}")
    print("💡 ลองรันคำสั่ง: python ProjectP.py")
'''
    
    with open("/content/drive/MyDrive/ProjectP/start_nicegold_complete.py", "w", encoding="utf-8") as f:
        f.write(startup_script)
    
    print_status("สร้าง startup script เสร็จเรียบร้อย", "SUCCESS")

def main():
    """ฟังก์ชันหลัก"""
    print("="*60)
    print("🏢 NICEGOLD ProjectP - Complete Installation")
    print("   ระบบติดตั้งและตั้งค่า NICEGOLD ProjectP")
    print("="*60)
    
    # เปลี่ยนไปยัง directory โปรเจค
    project_dir = "/content/drive/MyDrive/ProjectP"
    if os.path.exists(project_dir):
        os.chdir(project_dir)
        print_status(f"เปลี่ยนไปยัง: {project_dir}", "SUCCESS")
    else:
        print_status(f"ไม่พบ directory: {project_dir}", "ERROR")
        return
    
    # ติดตั้ง dependencies
    install_dependencies()
    
    print("\\n" + "="*60)
    
    # ตรวจสอบการติดตั้ง
    if verify_installation():
        print_status("การติดตั้งสำเร็จ! 🎉", "SUCCESS")
        
        # สร้าง startup script
        create_startup_script()
        
        print("\\n" + "="*60)
        print("🎯 วิธีใช้งาน:")
        print("   1. รันระบบหลัก: python ProjectP.py")
        print("   2. รันด้วย startup script: python start_nicegold_complete.py")
        print("   3. ทดสอบระบบ: python simple_protection_test.py")
        print("="*60)
        print("\\n✅ NICEGOLD ProjectP พร้อมใช้งาน!")
        
    else:
        print_status("การติดตั้งไม่สมบูรณ์", "ERROR")

if __name__ == "__main__":
    main()
