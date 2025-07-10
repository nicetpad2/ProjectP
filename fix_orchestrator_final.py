#!/usr/bin/env python3
"""
🔧 FINAL FIX: ElliottWavePipelineOrchestrator Arguments
แก้ไขปัญหา orchestrator ในทุกไฟล์ที่เป็นไปได้อย่างสมบูรณ์แบบ
"""

import os
import re
from pathlib import Path

def fix_orchestrator_calls():
    """แก้ไขการเรียกใช้ orchestrator ในทุกไฟล์"""
    project_root = Path('/content/drive/MyDrive/ProjectP')
    
    # Pattern ที่ต้องแก้ไข
    patterns_to_fix = [
        # Pattern 1: ElliottWavePipelineOrchestrator() - ไม่มี arguments
        (
            r'(\s*)(.*=\s*ElliottWavePipelineOrchestrator)\(\s*\)',
            r'\1\2(\n\1    data_processor=self.data_processor,\n\1    cnn_lstm_engine=self.cnn_lstm_engine,\n\1    dqn_agent=self.dqn_agent,\n\1    feature_selector=self.feature_selector,\n\1    ml_protection=self.ml_protection,\n\1    config=self.config,\n\1    logger=self.logger\n\1)'
        ),
        # Pattern 2: ElliottWavePipelineOrchestrator(config=..., logger=...) - ขาด components
        (
            r'(\s*)(.*=\s*ElliottWavePipelineOrchestrator)\(\s*config=([^,]+),\s*logger=([^)]+)\)',
            r'\1\2(\n\1    data_processor=self.data_processor,\n\1    cnn_lstm_engine=self.cnn_lstm_engine,\n\1    dqn_agent=self.dqn_agent,\n\1    feature_selector=self.feature_selector,\n\1    ml_protection=self.ml_protection,\n\1    config=\3,\n\1    logger=\4\n\1)'
        )
    ]
    
    # ไฟล์ที่ต้องแก้ไข
    files_to_check = [
        'menu_modules/menu_1_elliott_wave.py',
        'menu_modules/menu_1_elliott_wave_fixed.py',
        'menu_modules/menu_1_elliott_wave_advanced.py',
        'enhanced_menu_1_elliott_wave.py',
        'test_orchestrator_fix.py'
    ]
    
    fixed_files = []
    
    for file_path in files_to_check:
        full_path = project_root / file_path
        
        if not full_path.exists():
            print(f"⚠️ File not found: {file_path}")
            continue
            
        print(f"🔍 Checking: {file_path}")
        
        # อ่านไฟล์
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # ปรับแต่งแต่ละ pattern
        for pattern, replacement in patterns_to_fix:
            if re.search(pattern, content):
                print(f"  🔧 Fixing pattern in {file_path}")
                content = re.sub(pattern, replacement, content)
        
        # บันทึกถ้ามีการเปลี่ยนแปลง
        if content != original_content:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✅ Fixed: {file_path}")
            fixed_files.append(file_path)
        else:
            print(f"  ✅ No changes needed: {file_path}")
    
    return fixed_files

def create_summary_report():
    """สร้างรายงานสรุปการแก้ไข"""
    print("\n" + "="*60)
    print("📊 ORCHESTRATOR FIX SUMMARY REPORT")
    print("="*60)
    
    # ตรวจสอบไฟล์ทั้งหมดที่มี orchestrator
    project_root = Path('/content/drive/MyDrive/ProjectP')
    
    orchestrator_files = []
    for py_file in project_root.rglob('*.py'):
        if py_file.is_file():
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'ElliottWavePipelineOrchestrator' in content:
                        orchestrator_files.append(str(py_file.relative_to(project_root)))
            except:
                pass
    
    print(f"📁 Files containing ElliottWavePipelineOrchestrator: {len(orchestrator_files)}")
    for file_path in orchestrator_files:
        print(f"   - {file_path}")
    
    print(f"\n🎯 EXPECTED RESULT:")
    print(f"   ✅ All orchestrator calls should now have required arguments:")
    print(f"      - data_processor=self.data_processor")
    print(f"      - cnn_lstm_engine=self.cnn_lstm_engine") 
    print(f"      - dqn_agent=self.dqn_agent")
    print(f"      - feature_selector=self.feature_selector")
    print(f"      - ml_protection=self.ml_protection (optional)")
    print(f"      - config=self.config")
    print(f"      - logger=self.logger")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Test Menu 1 initialization")
    print(f"   2. Run full pipeline")
    print(f"   3. Verify no 'missing 4 required positional arguments' errors")

def main():
    """Main function"""
    print("🔧 STARTING COMPREHENSIVE ORCHESTRATOR FIX")
    print("="*50)
    
    # แก้ไขไฟล์
    fixed_files = fix_orchestrator_calls()
    
    if fixed_files:
        print(f"\n✅ Fixed {len(fixed_files)} files:")
        for file_path in fixed_files:
            print(f"   - {file_path}")
    else:
        print(f"\n✅ No files needed fixing (already correct)")
    
    # สร้างรายงาน
    create_summary_report()
    
    print(f"\n🎉 ORCHESTRATOR FIX COMPLETE!")
    print(f"🚀 Ready to test Menu 1 Elliott Wave Pipeline!")

if __name__ == "__main__":
    main()
