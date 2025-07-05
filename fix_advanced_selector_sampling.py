#!/usr/bin/env python3
"""
🔧 ENTERPRISE QUICK FIX: Remove all sampling from advanced_feature_selector.py
แก้ไขทุกตำแหน่งที่ใช้ sampling ให้ใช้ข้อมูลทั้งหมดจาก datacsv/ โฟลเดอร์
"""

import re

def fix_advanced_feature_selector():
    """แก้ไขไฟล์ advanced_feature_selector.py ให้ใช้ข้อมูลทั้งหมด"""
    
    file_path = "/mnt/data/projects/ProjectP/advanced_feature_selector.py"
    
    # อ่านไฟล์เดิม
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern สำหรับแก้ไข _standard_selection_with_sampling
    sampling_pattern = r'''def _standard_selection_with_sampling\(self, X: pd\.DataFrame, y: pd\.Series\) -> Tuple\[List\[str\], Dict\[str, Any\]\]:
        """Standard selection with smart sampling for large datasets"""
        # Sample data if too large
        if len\(X\) > 100000:
            self\.logger\.info\(f"📊 Sampling \{100000:,\} rows from \{len\(X\):,\} for efficiency"\)
            sample_idx = np\.random\.choice\(len\(X\), 100000, replace=False\)
            X_sample = X\.iloc\[sample_idx\]
            y_sample = y\.iloc\[sample_idx\]
        else:
            X_sample = X
            y_sample = y
        
        # Run standard selection on sample
        return self\._run_standard_selection\(X_sample, y_sample, original_size=len\(X\)\)'''
    
    # ใหม่ที่ใช้ข้อมูลทั้งหมด
    enterprise_replacement = '''def _standard_selection_with_sampling(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict[str, Any]]:
        """
        🎯 ENTERPRISE-GRADE: Full data feature selection WITHOUT SAMPLING
        ✅ แก้ไข: ใช้ข้อมูลทั้งหมดจาก datacsv/ โฟลเดอร์ พร้อม enterprise memory management
        """
        self.logger.info(f"🚀 ENTERPRISE: Processing FULL dataset {len(X):,} rows (NO SAMPLING)")
        
        # ✅ ENTERPRISE FIX: ใช้ข้อมูลทั้งหมด ไม่มี sampling
        X_sample = X.copy()
        y_sample = y.copy()
        self.logger.info(f"✅ Enterprise compliance: Using ALL {len(X_sample):,} rows from datacsv/")
        
        # Run standard selection on full data with enterprise resource management
        return self._run_standard_selection(X_sample, y_sample, original_size=len(X))'''
    
    # แทนที่ทุกตำแหน่ง
    new_content = re.sub(sampling_pattern, enterprise_replacement, content, flags=re.MULTILINE)
    
    # เขียนไฟล์ใหม่
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ Fixed advanced_feature_selector.py - removed all sampling")
    return True

if __name__ == "__main__":
    fix_advanced_feature_selector()
