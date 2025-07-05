#!/usr/bin/env python3
"""
🔧 ENTERPRISE COMPREHENSIVE FIX: Remove ALL sampling from the entire codebase
แก้ไขทุกไฟล์ที่ใช้ sampling หรือ row limits ให้ใช้ข้อมูลทั้งหมดจาก datacsv/ โฟลเดอร์
"""

import os
import re

def fix_all_sampling_issues():
    """แก้ไขปัญหา sampling ในทุกไฟล์"""
    
    print("🚀 Starting COMPREHENSIVE sampling fix...")
    
    # 1. แก้ไข advanced_feature_selector.py
    fix_advanced_feature_selector()
    
    # 2. แก้ไข fast_feature_selector.py
    fix_fast_feature_selector()
    
    # 3. แก้ไข nicegold_resource_optimization_engine.py
    fix_resource_optimization_engine()
    
    # 4. ตรวจสอบไฟล์อื่นๆ
    scan_and_fix_other_files()
    
    print("✅ COMPREHENSIVE sampling fix completed!")

def fix_advanced_feature_selector():
    """แก้ไข advanced_feature_selector.py"""
    print("🔧 Fixing advanced_feature_selector.py...")
    
    file_path = "/mnt/data/projects/ProjectP/advanced_feature_selector.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # แทนที่ sampling logic ทั้งหมด
        patterns = [
            # Pattern 1: Sampling in _standard_selection_with_sampling
            (r'if len\(X\) > 100000:\s*self\.logger\.info\(f"📊 Sampling.*?\n.*?sample_idx = np\.random\.choice.*?\n.*?X_sample = X\.iloc\[sample_idx\]\s*\n.*?y_sample = y\.iloc\[sample_idx\]\s*\n.*?else:\s*\n.*?X_sample = X\s*\n.*?y_sample = y',
             'X_sample = X.copy()\n        y_sample = y.copy()\n        self.logger.info(f"✅ ENTERPRISE: Using ALL {len(X_sample):,} rows from datacsv/ (NO SAMPLING)")'),
            
            # Pattern 2: Any remaining sampling references
            (r'Sampling.*?rows.*?for efficiency', 'ENTERPRISE: Using ALL rows from datacsv/ (NO SAMPLING)'),
            
            # Pattern 3: sample size references
            (r'sample_size.*?len\(X_sample\)', 'full_data_size = len(X_sample)'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed advanced_feature_selector.py")
        
    except Exception as e:
        print(f"❌ Error fixing advanced_feature_selector.py: {e}")

def fix_fast_feature_selector():
    """แก้ไข fast_feature_selector.py"""
    print("🔧 Fixing fast_feature_selector.py...")
    
    file_path = "/mnt/data/projects/ProjectP/fast_feature_selector.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # แทนที่ SHAP sampling
        content = re.sub(
            r'X_shap\.iloc\[:200\]',
            'X_shap',
            content
        )
        
        # แทนที่ความคิดเห็นเกี่ยวกับ sampling
        content = re.sub(
            r'# Very small sample',
            '# Full dataset for enterprise compliance',
            content
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed fast_feature_selector.py")
        
    except Exception as e:
        print(f"❌ Error fixing fast_feature_selector.py: {e}")

def fix_resource_optimization_engine():
    """แก้ไข nicegold_resource_optimization_engine.py"""
    print("🔧 Fixing nicegold_resource_optimization_engine.py...")
    
    file_path = "/mnt/data/projects/ProjectP/nicegold_resource_optimization_engine.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # แทนที่ SHAP sampling
        content = re.sub(
            r'X\.sample\(min\(800, len\(X\)\), random_state=42\)',
            'X.copy()  # Enterprise: Use full dataset',
            content
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Fixed nicegold_resource_optimization_engine.py")
        
    except Exception as e:
        print(f"❌ Error fixing nicegold_resource_optimization_engine.py: {e}")

def scan_and_fix_other_files():
    """สแกนและแก้ไขไฟล์อื่นๆ ที่อาจมี sampling"""
    print("🔍 Scanning for other sampling issues...")
    
    # ไฟล์ที่ต้องตรวจสอบ
    files_to_check = [
        "elliott_wave_modules/data_processor.py",
        "elliott_wave_modules/enterprise_ml_protection_backup.py",
        "elliott_wave_modules/enterprise_ml_protection_original.py",
    ]
    
    for file_path in files_to_check:
        full_path = f"/mnt/data/projects/ProjectP/{file_path}"
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # ตรวจสอบ sampling patterns
                if '.sample(' in content or 'nrows=' in content or '.iloc[:' in content:
                    print(f"⚠️ Found potential sampling in {file_path}")
                    
                    # แก้ไขปัญหาทั่วไป
                    content = re.sub(r'\.sample\([^)]+\)', '.copy()  # Enterprise: Use full dataset', content)
                    content = re.sub(r'nrows=\d+', 'nrows=None  # Enterprise: Use all rows', content)
                    
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"✅ Fixed sampling in {file_path}")
                
            except Exception as e:
                print(f"❌ Error checking {file_path}: {e}")

if __name__ == "__main__":
    fix_all_sampling_issues()
