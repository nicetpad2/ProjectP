#!/usr/bin/env python3
"""
🔄 AUTO LOGGING SYSTEM UPDATER
สคริปต์อัตโนมัติสำหรับปรับปรุงระบบ Logging ในไฟล์ที่เหลือ

🎯 Target Files:
- elliott_wave_modules/data_processor.py
- elliott_wave_modules/feature_selector.py  
- elliott_wave_modules/feature_engineering.py
- menu_modules/menu_1_elliott_wave.py

🚀 Features:
- Auto-detect current logging implementation
- Backup original files
- Insert advanced logging imports
- Update constructor and methods
- Validate syntax after updates
"""

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

class LoggingSystemUpdater:
    """🔄 Automatic Logging System Updater"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backups" / f"logging_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Files to update
        self.target_files = [
            'elliott_wave_modules/data_processor.py',
            'elliott_wave_modules/feature_selector.py',
            'elliott_wave_modules/feature_engineering.py',
            'menu_modules/menu_1_elliott_wave.py'
        ]
        
        # Advanced logging imports template
        self.logging_imports = '''
# 🚀 Advanced Logging Integration
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel, ProcessStatus
    from core.real_time_progress_manager import get_progress_manager, ProgressType
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    import logging
'''
        
        # Constructor update template
        self.constructor_update = '''
        # 🚀 Initialize Advanced Logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
            self.progress_manager = get_progress_manager()
            self.logger.info(f"🚀 {self.__class__.__name__} initialized with Advanced Logging", 
                            self.__class__.__name__)
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_manager = None
'''

    def analyze_file(self, file_path: str) -> Dict[str, any]:
        """วิเคราะห์ไฟล์และสถานะ logging ปัจจุบัน"""
        
        file_full_path = self.project_root / file_path
        
        if not file_full_path.exists():
            return {'exists': False, 'error': f'File not found: {file_path}'}
        
        try:
            with open(file_full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'exists': True,
                'file_path': file_path,
                'has_advanced_logging': 'advanced_terminal_logger' in content,
                'has_basic_logging': 'import logging' in content,
                'has_constructor': 'def __init__' in content,
                'current_logging_type': 'none',
                'needs_update': True,
                'lines': len(content.split('\\n')),
                'size_kb': len(content) / 1024
            }
            
            # Determine current logging type
            if 'advanced_terminal_logger' in content:
                analysis['current_logging_type'] = 'advanced'
                analysis['needs_update'] = False
            elif 'menu1_logger' in content:
                analysis['current_logging_type'] = 'menu1_logger'
            elif 'beautiful_logging' in content:
                analysis['current_logging_type'] = 'beautiful_logging'
            elif 'import logging' in content:
                analysis['current_logging_type'] = 'basic'
            
            return analysis
            
        except Exception as e:
            return {'exists': False, 'error': str(e)}
    
    def backup_file(self, file_path: str) -> bool:
        """สำรองไฟล์ก่อนการแก้ไข"""
        try:
            source = self.project_root / file_path
            backup_file = self.backup_dir / file_path
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, backup_file)
            print(f"✅ Backed up: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed for {file_path}: {e}")
            return False
    
    def update_file_logging(self, file_path: str) -> Dict[str, any]:
        """ปรับปรุงระบบ logging ในไฟล์"""
        
        try:
            file_full_path = self.project_root / file_path
            
            # Read original content
            with open(file_full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            updated = False
            
            # 1. Add advanced logging imports (if not present)
            if 'advanced_terminal_logger' not in content:
                # Find the last import statement
                import_pattern = r'^(import .*|from .* import .*)'
                import_matches = list(re.finditer(import_pattern, content, re.MULTILINE))
                
                if import_matches:
                    last_import = import_matches[-1]
                    insert_pos = last_import.end()
                    content = content[:insert_pos] + self.logging_imports + content[insert_pos:]
                    updated = True
                    print(f"✅ Added advanced logging imports to {file_path}")
            
            # 2. Update constructor (if has __init__ and needs update)
            if 'def __init__' in content and 'ADVANCED_LOGGING_AVAILABLE' not in content:
                # Find __init__ method and update logger initialization
                init_pattern = r'(def __init__\(.*?\):.*?)(\n        self\.logger = .*)'
                
                def replace_logger_init(match):
                    method_def = match.group(1)
                    return method_def + self.constructor_update
                
                new_content = re.sub(init_pattern, replace_logger_init, content, flags=re.DOTALL)
                if new_content != content:
                    content = new_content
                    updated = True
                    print(f"✅ Updated constructor in {file_path}")
            
            # 3. Replace basic logging calls with advanced logging (selective)
            # This is more complex and should be done carefully
            
            # Save updated content
            if updated:
                with open(file_full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    'success': True,
                    'updated': True,
                    'file_path': file_path,
                    'changes': ['imports', 'constructor'] if updated else [],
                    'original_size': len(original_content),
                    'new_size': len(content)
                }
            else:
                return {
                    'success': True,
                    'updated': False,
                    'file_path': file_path,
                    'reason': 'No updates needed or already has advanced logging'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    def validate_syntax(self, file_path: str) -> bool:
        """ตรวจสอบ syntax หลังการแก้ไข"""
        try:
            file_full_path = self.project_root / file_path
            
            with open(file_full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile the code
            compile(content, str(file_full_path), 'exec')
            return True
            
        except SyntaxError as e:
            print(f"❌ Syntax error in {file_path}: Line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            print(f"⚠️ Validation warning for {file_path}: {e}")
            return True  # Allow other errors as they might be import-related
    
    def run_analysis(self) -> Dict[str, any]:
        """วิเคราะห์ไฟล์ทั้งหมดและสร้างรายงาน"""
        
        print("🔍 NICEGOLD LOGGING SYSTEM ANALYSIS")
        print("=" * 50)
        
        results = {}
        
        for file_path in self.target_files:
            print(f"\\n📁 Analyzing: {file_path}")
            analysis = self.analyze_file(file_path)
            results[file_path] = analysis
            
            if analysis['exists']:
                print(f"   Current logging: {analysis['current_logging_type']}")
                print(f"   Needs update: {'Yes' if analysis['needs_update'] else 'No'}")
                print(f"   Size: {analysis['size_kb']:.1f} KB, Lines: {analysis['lines']}")
            else:
                print(f"   ❌ {analysis.get('error', 'File not accessible')}")
        
        # Summary
        total_files = len(self.target_files)
        existing_files = len([r for r in results.values() if r['exists']])
        needs_update = len([r for r in results.values() if r.get('needs_update', False)])
        
        print(f"\\n📊 SUMMARY:")
        print(f"   Total files: {total_files}")
        print(f"   Existing files: {existing_files}")
        print(f"   Needs update: {needs_update}")
        print(f"   Already updated: {existing_files - needs_update}")
        
        return results
    
    def run_updates(self, dry_run: bool = True) -> Dict[str, any]:
        """ดำเนินการปรับปรุงไฟล์ทั้งหมด"""
        
        if dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        else:
            print("🚀 LIVE UPDATE MODE - Files will be modified")
        
        print("=" * 50)
        
        results = {}
        
        for file_path in self.target_files:
            print(f"\\n🔄 Processing: {file_path}")
            
            # Analyze first
            analysis = self.analyze_file(file_path)
            
            if not analysis['exists']:
                print(f"   ❌ Skipped: {analysis.get('error', 'File not found')}")
                results[file_path] = {'status': 'skipped', 'reason': 'file_not_found'}
                continue
            
            if not analysis['needs_update']:
                print(f"   ✅ Already updated: {analysis['current_logging_type']} logging")
                results[file_path] = {'status': 'already_updated', 'logging_type': analysis['current_logging_type']}
                continue
            
            if not dry_run:
                # Backup file
                if not self.backup_file(file_path):
                    print(f"   ❌ Backup failed, skipping update")
                    results[file_path] = {'status': 'failed', 'reason': 'backup_failed'}
                    continue
                
                # Update file
                update_result = self.update_file_logging(file_path)
                
                if update_result['success']:
                    if update_result['updated']:
                        # Validate syntax
                        if self.validate_syntax(file_path):
                            print(f"   ✅ Successfully updated and validated")
                            results[file_path] = {'status': 'updated', 'details': update_result}
                        else:
                            print(f"   ⚠️ Updated but syntax validation failed")
                            results[file_path] = {'status': 'updated_with_warnings', 'details': update_result}
                    else:
                        print(f"   ✅ No changes needed")
                        results[file_path] = {'status': 'no_changes_needed'}
                else:
                    print(f"   ❌ Update failed: {update_result.get('error', 'Unknown error')}")
                    results[file_path] = {'status': 'failed', 'error': update_result.get('error')}
            else:
                print(f"   🔍 Would update: {analysis['current_logging_type']} → advanced")
                results[file_path] = {'status': 'would_update', 'current': analysis['current_logging_type']}
        
        return results
    
    def generate_report(self, results: Dict[str, any]) -> str:
        """สร้างรายงานสรุป"""
        
        report = f"""
# 🔄 LOGGING SYSTEM UPDATE REPORT
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Project**: NICEGOLD Enterprise ProjectP

## 📊 Summary
"""
        
        status_counts = {}
        for result in results.values():
            status = result.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            report += f"- **{status.replace('_', ' ').title()}**: {count} files\\n"
        
        report += "\\n## 📋 Detailed Results\\n"
        
        for file_path, result in results.items():
            status = result.get('status', 'unknown')
            report += f"\\n### {file_path}\\n"
            report += f"**Status**: {status.replace('_', ' ').title()}\\n"
            
            if 'reason' in result:
                report += f"**Reason**: {result['reason']}\\n"
            if 'error' in result:
                report += f"**Error**: {result['error']}\\n"
            if 'current' in result:
                report += f"**Current**: {result['current']} logging\\n"
        
        report += f"""
## 🎯 Next Steps
1. Review any files with warnings or errors
2. Test the updated logging system
3. Run full system validation
4. Update documentation if needed

**Backup Location**: {self.backup_dir}
"""
        
        return report


def main():
    """Main execution function"""
    print("🚀 NICEGOLD ADVANCED LOGGING SYSTEM UPDATER")
    print("=" * 60)
    
    updater = LoggingSystemUpdater()
    
    # Step 1: Analysis
    print("\\n📊 Step 1: File Analysis")
    analysis_results = updater.run_analysis()
    
    # Step 2: Ask for confirmation
    print("\\n" + "=" * 60)
    response = input("🤔 Proceed with updates? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        # Step 3: Dry run first
        print("\\n🔍 Step 2: Dry Run")
        dry_results = updater.run_updates(dry_run=True)
        
        print("\\n" + "=" * 60)
        response = input("✅ Dry run complete. Proceed with actual updates? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            # Step 4: Actual updates
            print("\\n🚀 Step 3: Actual Updates")
            update_results = updater.run_updates(dry_run=False)
            
            # Step 5: Generate report
            report = updater.generate_report(update_results)
            
            # Save report
            report_file = Path(f"LOGGING_UPDATE_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\\n📋 Report saved: {report_file}")
            print("\\n✅ Logging system update completed!")
        else:
            print("\\n❌ Actual updates cancelled.")
    else:
        print("\\n❌ Updates cancelled.")


if __name__ == "__main__":
    main()
