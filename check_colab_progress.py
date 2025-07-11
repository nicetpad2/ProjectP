#!/usr/bin/env python3
"""
🔍 Google Colab Progress Bars Fix
ตรวจสอบและแก้ไข progress bars ให้เหมาะกับ Colab environment
"""

import os
import sys
import time
from pathlib import Path

def detect_environment():
    """ตรวจสอบ environment ที่รันอยู่"""
    print("🔍 Environment Detection:")
    print("=" * 50)
    
    # Check for Google Colab
    try:
        import google.colab
        print("✅ Environment: Google Colab")
        print("📊 Colab-specific optimizations needed")
        return "colab"
    except ImportError:
        pass
    
    # Check for Jupyter
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            print("✅ Environment: Jupyter/IPython")
            return "jupyter"
    except ImportError:
        pass
    
    print("✅ Environment: Terminal/Command Line")
    return "terminal"

def test_rich_compatibility():
    """ทดสอบความเข้ากันได้ของ Rich library"""
    print("\n🎨 Testing Rich Library Compatibility:")
    print("=" * 50)
    
    try:
        from rich.console import Console
        from rich.progress import Progress, TaskID
        from rich.live import Live
        
        console = Console()
        print("✅ Rich Console: Available")
        
        # Test progress bar
        try:
            with Progress() as progress:
                task = progress.add_task("Testing...", total=100)
                for i in range(10):
                    progress.update(task, advance=10)
                    time.sleep(0.1)
            print("✅ Rich Progress: Working")
            return True
        except Exception as e:
            print(f"⚠️ Rich Progress: Limited ({str(e)[:50]}...)")
            return False
            
    except ImportError as e:
        print(f"❌ Rich Library: Not available ({e})")
        return False

def create_colab_optimized_progress():
    """สร้าง progress system ที่เหมาะกับ Colab"""
    print("\n🔧 Creating Colab-Optimized Progress System:")
    print("=" * 50)
    
    progress_code = '''
import time
import sys
from datetime import datetime

class ColabProgressBar:
    """Progress bar ที่เหมาะกับ Google Colab"""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, step_name="", increment=1):
        """อัปเดต progress"""
        self.current_step += increment
        percentage = (self.current_step / self.total_steps) * 100
        elapsed_time = time.time() - self.start_time
        
        # สร้าง progress bar แบบ text
        bar_length = 40
        filled_length = int(bar_length * self.current_step // self.total_steps)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # คำนวณเวลาที่เหลือ
        if self.current_step > 0:
            eta = (elapsed_time / self.current_step) * (self.total_steps - self.current_step)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --:--"
        
        # แสดงผล (overwrite line)
        output = f"\\r🚀 {self.description} [{bar}] {percentage:.1f}% ({self.current_step}/{self.total_steps}) - {step_name} - {eta_str}"
        sys.stdout.write(output)
        sys.stdout.flush()
        
        if self.current_step >= self.total_steps:
            print(f"\\n✅ {self.description} Complete! ({elapsed_time:.1f}s)")
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_step < self.total_steps:
            print(f"\\n⚠️ {self.description} interrupted at {self.current_step}/{self.total_steps}")

# Test the progress bar
def test_colab_progress():
    """ทดสอบ progress bar สำหรับ Colab"""
    print("\\n🧪 Testing Colab Progress Bar:")
    
    with ColabProgressBar(10, "Demo Process") as progress:
        for i in range(10):
            time.sleep(0.5)
            progress.update(f"Step {i+1}")
    
    return True

if __name__ == "__main__":
    test_colab_progress()
'''
    
    # เขียนไฟล์
    with open("colab_progress.py", "w", encoding="utf-8") as f:
        f.write(progress_code)
    
    print("✅ Created: colab_progress.py")
    
    # ทดสอบ
    try:
        exec(progress_code)
        print("✅ Colab Progress System: Working")
        return True
    except Exception as e:
        print(f"❌ Colab Progress System: Error - {e}")
        return False

def check_process_status():
    """ตรวจสอบสถานะ process ที่กำลังทำงาน"""
    print("\n📊 Process Status Check:")
    print("=" * 50)
    
    try:
        import psutil
        
        # หา Python processes
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"🐍 Found {len(python_processes)} Python processes:")
            for proc in python_processes[:5]:  # Show top 5
                print(f"   PID: {proc['pid']}, CPU: {proc['cpu_percent']:.1f}%, Memory: {proc['memory_percent']:.1f}%")
        else:
            print("⚠️ No Python processes found")
            
        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"\n💻 System Resources:")
        print(f"   CPU Usage: {cpu_percent}%")
        print(f"   Memory Usage: {memory.percent}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
        
        return True
        
    except ImportError:
        print("❌ psutil not available - cannot check process status")
        return False

def main():
    """Main function"""
    print("🔍 GOOGLE COLAB PROGRESS BARS DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Detect environment
    env = detect_environment()
    
    # 2. Test Rich compatibility
    rich_works = test_rich_compatibility()
    
    # 3. Create Colab-optimized progress
    if env == "colab":
        colab_progress_works = create_colab_optimized_progress()
    else:
        colab_progress_works = False
    
    # 4. Check process status
    process_status = check_process_status()
    
    # Summary
    print("\n📋 SUMMARY:")
    print("=" * 30)
    print(f"Environment: {env}")
    print(f"Rich Progress: {'✅ Working' if rich_works else '❌ Limited'}")
    print(f"Colab Progress: {'✅ Available' if colab_progress_works else '❌ Not needed'}")
    print(f"Process Monitoring: {'✅ Available' if process_status else '❌ Limited'}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 30)
    
    if env == "colab" and not rich_works:
        print("🔧 Use text-based progress bars for Colab")
        print("🔧 Implement timeout mechanisms for training")
        print("🔧 Add periodic status updates")
    elif rich_works:
        print("✅ Rich progress bars should work fine")
    
    if not process_status:
        print("⚠️ Install psutil for better monitoring: pip install psutil")

if __name__ == "__main__":
    main() 