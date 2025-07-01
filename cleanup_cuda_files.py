#!/usr/bin/env python3
"""
🧹 CLEANUP CUDA FIX FILES
ทำความสะอาดไฟล์ทดสอบและเครื่องมือแก้ไข CUDA หลังจากแก้ไขเสร็จแล้ว
"""

import os
from pathlib import Path

def cleanup_cuda_files():
    """ลบไฟล์ทดสอบและเครื่องมือที่ใช้แก้ไข CUDA"""
    
    files_to_remove = [
        "fix_cuda_issues.py",
        "fix_elliott_cuda.py", 
        "test_cuda_fix.py",
        "cleanup_cuda_files.py"  # This file itself
    ]
    
    print("🧹 CUDA FIX FILES CLEANUP")
    print("=" * 40)
    
    removed_count = 0
    
    for file_path in files_to_remove:
        if Path(file_path).exists():
            try:
                os.remove(file_path)
                print(f"✅ Removed: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Failed to remove {file_path}: {e}")
        else:
            print(f"📁 Not found: {file_path}")
    
    print("\n" + "=" * 40)
    print("🎉 CLEANUP COMPLETE")
    print("=" * 40)
    print(f"✅ Removed {removed_count} files")
    print("🚀 ProjectP.py is ready for production use")
    print("📋 Keep CUDA_FIX_COMPLETE_SOLUTION.md for reference")

if __name__ == "__main__":
    cleanup_cuda_files()
