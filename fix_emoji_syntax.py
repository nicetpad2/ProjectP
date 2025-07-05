#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix emoji-related syntax issues in advanced_feature_selector.py
"""

import re
import os

def fix_emoji_issues():
    """Fix emoji encoding issues that might cause syntax errors"""
    file_path = '/mnt/data/projects/ProjectP/advanced_feature_selector.py'
    
    # Backup the original file
    backup_path = file_path + '.backup'
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Replace problematic emojis with safe alternatives
        emoji_replacements = {
            '⚠️': 'WARNING',
            '❌': 'ERROR', 
            '✅': 'SUCCESS',
            '🎯': 'TARGET',
            '🏆': 'ACHIEVEMENT',
            '🎉': 'SUCCESS',
            '🚀': 'INFO',
            '📊': 'DATA',
            '🧠': 'AI',
            '🔧': 'FIX',
            '💡': 'INFO',
            '⭐': 'STAR',
            '🔍': 'SEARCH',
            '📈': 'TREND',
            '🎨': 'STYLE',
            '🛡️': 'PROTECTION'
        }
        
        # Apply replacements
        fixed_content = content
        for emoji, replacement in emoji_replacements.items():
            fixed_content = fixed_content.replace(emoji, replacement)
        
        # Write the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"✅ Fixed emoji issues in {file_path}")
        print(f"📄 Backup created at {backup_path}")
        
        # Test syntax
        import ast
        try:
            ast.parse(fixed_content)
            print("✅ Syntax validation passed!")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error still exists at line {e.lineno}: {e.msg}")
            # Restore backup
            with open(backup_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            print("🔄 Restored original file")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing emoji issues: {e}")
        return False

if __name__ == "__main__":
    fix_emoji_issues()
