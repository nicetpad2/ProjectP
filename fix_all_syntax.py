#!/usr/bin/env python3
"""
🔧 ENTERPRISE COMPREHENSIVE SYNTAX FIX
แก้ไข syntax errors ทั้งหมดใน advanced_feature_selector.py
"""

def fix_all_syntax_errors():
    """แก้ไข syntax errors ทั้งหมด"""
    
    file_path = "/mnt/data/projects/ProjectP/advanced_feature_selector.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("🔧 Fixing all syntax errors...")
        
        # Fix missing closing parentheses in fail_progress calls
        content = content.replace(
            'self.progress_manager.fail_progress(main_progress, str(e)',
            'self.progress_manager.fail_progress(main_progress, str(e))'
        )
        
        # Fix other potential syntax issues
        fixes = [
            # Fix any double closing parentheses that might have been created
            ('))', ')'),
            
            # Fix any broken method calls
            ('str(e))', 'str(e))'),
            ('str(e)))', 'str(e))'),
        ]
        
        for old, new in fixes:
            content = content.replace(old, new)
        
        # Write back the fixed content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ All syntax fixes applied")
        
        # Test syntax
        import ast
        try:
            ast.parse(content)
            print("✅ Syntax validation passed - file is now valid")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error still exists: {e}")
            print(f"📍 Line {e.lineno}: {e.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during fix: {e}")
        return False

if __name__ == "__main__":
    success = fix_all_syntax_errors()
    if success:
        print("🎉 All syntax errors fixed successfully!")
    else:
        print("❌ Some syntax errors remain")
