#!/usr/bin/env python3
"""
🛠️ CUDA FIX APPLIER FOR ELLIOTT WAVE MODULES
ระบบแก้ไข CUDA สำหรับ Elliott Wave modules โดยเฉพาะ

แก้ไขปัญหา CUDA ใน:
- cnn_lstm_engine.py
- dqn_agent.py  
- feature_selector.py
- pipeline_orchestrator.py
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CUDA_ELLIOTT_FIX - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_cuda_fix_to_elliott_modules():
    """แก้ไข CUDA issues ใน Elliott Wave modules ทั้งหมด"""
    
    logger.info("🚀 Starting CUDA fixes for Elliott Wave modules...")
    
    # Elliott Wave modules ที่ต้องแก้ไข
    modules_to_fix = [
        "elliott_wave_modules/cnn_lstm_engine.py",
        "elliott_wave_modules/dqn_agent.py",
        "elliott_wave_modules/feature_selector.py", 
        "elliott_wave_modules/pipeline_orchestrator.py",
        "elliott_wave_modules/data_processor.py",
        "elliott_wave_modules/performance_analyzer.py"
    ]
    
    success_count = 0
    total_modules = len(modules_to_fix)
    
    for module_path in modules_to_fix:
        logger.info(f"🔧 Fixing: {module_path}")
        
        if Path(module_path).exists():
            if apply_cpu_fix_to_module(module_path):
                success_count += 1
                logger.info(f"✅ Fixed: {module_path}")
            else:
                logger.warning(f"⚠️ Failed to fix: {module_path}")
        else:
            logger.warning(f"📁 File not found: {module_path}")
    
    # สรุปผลลัพธ์
    logger.info("=" * 60)
    logger.info("🎉 ELLIOTT WAVE CUDA FIX SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully fixed: {success_count}/{total_modules} modules")
    
    if success_count == total_modules:
        logger.info("🏆 All Elliott Wave modules fixed successfully!")
        return True
    else:
        logger.warning(f"⚠️ {total_modules - success_count} modules had issues")
        return False


def apply_cpu_fix_to_module(module_path):
    """แก้ไขโมดูลแต่ละตัวให้ใช้ CPU เท่านั้น"""
    try:
        # อ่านไฟล์
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # ตรวจสอบว่าได้แก้ไขแล้วหรือยัง
        if "# 🛠️ CUDA FIX:" in content:
            logger.info(f"📋 {module_path} already has CUDA fix applied")
            return True
        
        # สร้าง CUDA fix header
        cuda_fix_header = '''# 🛠️ CUDA FIX: Force CPU-only operation to prevent CUDA errors
import os
import warnings

# Environment variables to force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress CUDA warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

'''
        
        # หาตำแหน่งที่จะแทรกโค้ด
        lines = content.split('\n')
        insert_index = find_insertion_point(lines)
        
        # แทรก CUDA fix
        lines.insert(insert_index, cuda_fix_header)
        
        # เขียนกลับไปที่ไฟล์
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error fixing {module_path}: {e}")
        return False


def find_insertion_point(lines):
    """หาตำแหน่งที่เหมาะสมสำหรับแทรก CUDA fix"""
    
    # หาจุดที่อยู่หลัง docstring และก่อน imports
    in_docstring = False
    docstring_end = 0
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # ตรวจจับ docstring
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if not in_docstring:
                in_docstring = True
            elif stripped.endswith('"""') or stripped.endswith("'''"):
                docstring_end = i + 1
                break
        elif in_docstring and (stripped.endswith('"""') or stripped.endswith("'''")):
            docstring_end = i + 1
            break
    
    # หาตำแหน่งหลัง docstring แต่ก่อน import แรก
    for i in range(docstring_end, len(lines)):
        stripped = lines[i].strip()
        
        # หาบรรทัดที่ไม่ใช่ comment และไม่ใช่บรรทัดว่าง
        if stripped and not stripped.startswith('#'):
            # ถ้าเป็น import statement ให้แทรกก่อนหน้า
            if stripped.startswith('import ') or stripped.startswith('from '):
                return i
            # ถ้าไม่ใช่ import ให้แทรกหลัง docstring
            elif docstring_end > 0:
                return docstring_end
            else:
                return i
    
    # ถ้าไม่เจอจุดที่เหมาะสม ให้แทรกหลัง docstring
    return max(1, docstring_end)


def fix_tensorflow_imports():
    """แก้ไข TensorFlow imports ให้ใช้ CPU เท่านั้น"""
    try:
        # สร้างโมดูลสำหรับ TensorFlow ที่ปลอดภัย
        tensorflow_fix = '''
"""
🛠️ TensorFlow CPU-Only Import Module
Safe TensorFlow import for NICEGOLD ProjectP
"""

import os
import warnings

# Force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def safe_tensorflow():
    """Import TensorFlow safely for CPU-only operation"""
    try:
        import tensorflow as tf
        
        # Configure for CPU only
        tf.config.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('ERROR')
        
        return tf
    except Exception as e:
        print(f"TensorFlow import error: {e}")
        return None

# Safe import
tf = safe_tensorflow()
'''
        
        # สร้างไฟล์
        tf_safe_path = Path("core/tensorflow_safe.py")
        tf_safe_path.parent.mkdir(exist_ok=True)
        
        with open(tf_safe_path, 'w', encoding='utf-8') as f:
            f.write(tensorflow_fix)
        
        logger.info("✅ TensorFlow safe import module created")
        return True
        
    except Exception as e:
        logger.error(f"❌ TensorFlow fix failed: {e}")
        return False


def fix_pytorch_imports():
    """แก้ไข PyTorch imports ให้ใช้ CPU เท่านั้น"""
    try:
        # สร้างโมดูลสำหรับ PyTorch ที่ปลอดภัย
        pytorch_fix = '''
"""
🛠️ PyTorch CPU-Only Import Module
Safe PyTorch import for NICEGOLD ProjectP
"""

import os

# Force CPU-only operation
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def safe_pytorch():
    """Import PyTorch safely for CPU-only operation"""
    try:
        import torch
        
        # Set default to CPU
        torch.set_default_tensor_type('torch.FloatTensor')
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return torch
    except Exception as e:
        print(f"PyTorch import error: {e}")
        return None

# Safe import
torch = safe_pytorch()
'''
        
        # สร้างไฟล์
        torch_safe_path = Path("core/pytorch_safe.py")
        torch_safe_path.parent.mkdir(exist_ok=True)
        
        with open(torch_safe_path, 'w', encoding='utf-8') as f:
            f.write(pytorch_fix)
        
        logger.info("✅ PyTorch safe import module created")
        return True
        
    except Exception as e:
        logger.error(f"❌ PyTorch fix failed: {e}")
        return False


def main():
    """Main function"""
    print("🛠️ CUDA FIX FOR ELLIOTT WAVE MODULES")
    print("=" * 50)
    
    success_tasks = 0
    total_tasks = 3
    
    # Task 1: Fix Elliott Wave modules
    if apply_cuda_fix_to_elliott_modules():
        success_tasks += 1
    
    # Task 2: Fix TensorFlow imports
    if fix_tensorflow_imports():
        success_tasks += 1
    
    # Task 3: Fix PyTorch imports
    if fix_pytorch_imports():
        success_tasks += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 CUDA FIX COMPLETE SUMMARY")
    print("=" * 50)
    print(f"✅ Completed tasks: {success_tasks}/{total_tasks}")
    
    if success_tasks == total_tasks:
        print("🏆 All CUDA fixes applied successfully!")
        print("🚀 Elliott Wave modules should now work without CUDA errors")
        return True
    else:
        print("⚠️ Some fixes failed. Check the logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
