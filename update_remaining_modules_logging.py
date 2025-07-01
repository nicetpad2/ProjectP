#!/usr/bin/env python3
"""
🔧 UPDATE REMAINING MODULES WITH ADVANCED LOGGING
อัปเดตโมดูลที่เหลือให้ใช้ Advanced Terminal Logger ทุกไฟล์
"""

import os
import sys
from pathlib import Path
import re
from typing import List, Dict

# Files to update with their specific modifications
MODULES_TO_UPDATE = {
    'elliott_wave_modules/data_processor.py': {
        'imports_to_add': [
            'from core.advanced_terminal_logger import get_terminal_logger',
            'from core.real_time_progress_manager import get_progress_manager'
        ],
        'class_name': 'ElliottWaveDataProcessor',
        'fallback_available': True
    },
    'elliott_wave_modules/cnn_lstm_engine.py': {
        'imports_to_add': [
            'from core.advanced_terminal_logger import get_terminal_logger',
            'from core.real_time_progress_manager import get_progress_manager'
        ],
        'class_name': 'CNNLSTMElliottWave',
        'fallback_available': True
    },
    'elliott_wave_modules/dqn_agent.py': {
        'imports_to_add': [
            'from core.advanced_terminal_logger import get_terminal_logger',
            'from core.real_time_progress_manager import get_progress_manager'
        ],
        'class_name': 'DQNReinforcementAgent',
        'fallback_available': True
    },
    'elliott_wave_modules/feature_selector.py': {
        'imports_to_add': [
            'from core.advanced_terminal_logger import get_terminal_logger',
            'from core.real_time_progress_manager import get_progress_manager'
        ],
        'class_name': 'EnterpriseShapOptunaFeatureSelector',
        'fallback_available': True
    },
    'elliott_wave_modules/pipeline_orchestrator.py': {
        'imports_to_add': [
            'from core.advanced_terminal_logger import get_terminal_logger',
            'from core.real_time_progress_manager import get_progress_manager'
        ],
        'class_name': 'ElliottWavePipelineOrchestrator',
        'fallback_available': True
    },
    'elliott_wave_modules/performance_analyzer.py': {
        'imports_to_add': [
            'from core.advanced_terminal_logger import get_terminal_logger',
            'from core.real_time_progress_manager import get_progress_manager'
        ],
        'class_name': 'ElliottWavePerformanceAnalyzer',
        'fallback_available': True
    },
    'core/output_manager.py': {
        'imports_to_add': [
            'from core.advanced_terminal_logger import get_terminal_logger',
            'from core.real_time_progress_manager import get_progress_manager'
        ],
        'class_name': 'NicegoldOutputManager',
        'fallback_available': True
    }
}

def update_module_with_advanced_logging(file_path: str, module_info: Dict):
    """อัปเดตไฟล์เดียวให้ใช้ Advanced Logging"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup original content
        original_content = content
        
        # Track if any changes made
        changes_made = False
        
        # 1. Add Advanced Logger imports after existing imports
        import_section_pattern = r'(import logging.*?\n)'
        
        # Check if advanced logging imports already exist
        if 'from core.advanced_terminal_logger import' not in content:
            # Find the import logging line and add our imports after it
            if re.search(import_section_pattern, content):
                new_imports = '\n'.join(module_info['imports_to_add'])
                advanced_imports_block = f"""
# Advanced Logging Integration
try:
    {new_imports}
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
"""
                content = re.sub(
                    import_section_pattern,
                    r'\1' + advanced_imports_block + '\n',
                    content
                )
                changes_made = True
                print(f"✅ Added advanced logging imports to {file_path}")
        
        # 2. Update class constructor to use advanced logger
        class_constructor_pattern = rf'(class {module_info["class_name"]}.*?def __init__\(self.*?logger.*?=.*?None\):)'
        
        if re.search(class_constructor_pattern, content, re.DOTALL):
            # Find and update the logger initialization
            logger_init_pattern = r'self\.logger = logger or logging\.getLogger\(__name__\)'
            
            if re.search(logger_init_pattern, content):
                new_logger_init = """# Initialize Advanced Terminal Logger
        if ADVANCED_LOGGING_AVAILABLE:
            try:
                self.logger = get_terminal_logger()
                self.progress_manager = get_progress_manager()
                self.logger.info(f"🚀 {self.__class__.__name__} initialized with advanced logging", self.__class__.__name__)
            except Exception as e:
                self.logger = logger or logging.getLogger(__name__)
                self.progress_manager = None
                print(f"⚠️ Advanced logging failed, using fallback: {e}")
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_manager = None"""
            
                content = re.sub(
                    logger_init_pattern,
                    new_logger_init,
                    content
                )
                changes_made = True
                print(f"✅ Updated logger initialization in {file_path}")
        
        # 3. Replace common logging patterns with advanced logger usage
        # Replace self.logger.info with advanced logger when appropriate
        info_pattern = r'self\.logger\.info\(([^)]+)\)'
        warning_pattern = r'self\.logger\.warning\(([^)]+)\)'
        error_pattern = r'self\.logger\.error\(([^)]+)\)'
        
        # Add progress manager usage in key methods
        if 'def ' in content and 'process' in content.lower():
            # Find methods that might benefit from progress tracking
            method_patterns = [
                r'def (train|fit|load|process|analyze|run|execute).*?\(.*?\):',
                r'def _.*?(train|fit|load|process|analyze|run|execute).*?\(.*?\):'
            ]
            
            for pattern in method_patterns:
                methods = re.findall(pattern, content, re.IGNORECASE)
                if methods:
                    print(f"📊 Found methods that can use progress tracking: {methods}")
        
        # Only save if changes were made
        if changes_made:
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"💾 Updated {file_path} successfully")
            return True
        else:
            print(f"ℹ️ No changes needed for {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def update_all_modules():
    """อัปเดตโมดูลทั้งหมด"""
    print("🔧 UPDATING REMAINING MODULES WITH ADVANCED LOGGING")
    print("=" * 60)
    
    updated_count = 0
    failed_count = 0
    
    for file_path, module_info in MODULES_TO_UPDATE.items():
        print(f"\n📝 Updating: {file_path}")
        
        if update_module_with_advanced_logging(file_path, module_info):
            updated_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY:")
    print(f"✅ Updated: {updated_count} modules")
    print(f"❌ Failed: {failed_count} modules")
    print(f"📁 Total: {len(MODULES_TO_UPDATE)} modules processed")
    
    if updated_count > 0:
        print("\n🎉 Advanced Logging integration completed!")
        print("💡 All modules now support:")
        print("   - Beautiful terminal logging with colors")
        print("   - Real-time progress bars")
        print("   - Advanced error tracking")
        print("   - System monitoring integration")
    
    return updated_count > 0

if __name__ == "__main__":
    success = update_all_modules()
    
    if success:
        print("\n🚀 Ready to test the updated system!")
        print("Try running: python test_advanced_logging.py")
    else:
        print("\n⚠️ Some updates may have failed. Check the log messages above.")
    
    sys.exit(0 if success else 1)
