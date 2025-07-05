#!/usr/bin/env python3
"""
🔧 ENTERPRISE SYNTAX REPAIR
แก้ไข syntax errors ทั้งหมดในไฟล์ advanced_feature_selector.py
"""

import re

def repair_advanced_feature_selector():
    """แก้ไข syntax errors ใน advanced_feature_selector.py"""
    
    file_path = "/mnt/data/projects/ProjectP/advanced_feature_selector.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("🔧 Repairing advanced_feature_selector.py...")
        
        # Fix common indentation and syntax issues from regex replacements
        repairs = [
            # Fix double parentheses
            (r'len\(X_sample\)\)', 'len(X_sample)'),
            
            # Fix broken indentation from SHAP section
            (r'(\s+)# ✅ Enhanced SHAP values extraction with robust error handling\s*\n\s+shap_values = explainer\.shap_values\(X_sample\.iloc\[shap_idx\]\)\s*\n\s+# ✅ Robust handling',
             r'\1# ✅ Enhanced SHAP values extraction with robust error handling\n\1shap_values = explainer.shap_values(X_sample.iloc[shap_idx])\n\1\n\1# ✅ Robust handling'),
            
            # Fix any remaining broken parentheses
            (r'\)\)', ')'),
            
            # Fix extra spaces and formatting
            (r'\s+\n\s+\n\s+#', '\n\n            #'),
            
            # Fix method definitions that might be broken
            (r'def\s+(\w+)\s*\(\s*self\s*,([^)]*)\s*\)\s*->\s*([^:]+):\s*\n\s*"""([^"]+)"""\s*\n\s*([^#\n]*)\n\s*#',
             r'def \1(self,\2) -> \3:\n        """\4"""\n        \5\n        #'),
        ]
        
        original_content = content
        
        for pattern, replacement in repairs:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
        
        # Additional cleanup for broken structures
        content = re.sub(r'(\s+)shap_values = explainer\.shap_values\(X_sample\.iloc\[shap_idx\]\)\s*\n\s+# ✅ Robust handling',
                        r'\1shap_values = explainer.shap_values(X_sample.iloc[shap_idx])\n\1\n\1# ✅ Robust handling', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Syntax repairs applied")
        else:
            print("ℹ️ No repairs needed")
        
        # Test syntax
        import ast
        try:
            ast.parse(content)
            print("✅ Syntax validation passed")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error still exists: {e}")
            print(f"📍 Line {e.lineno}: {e.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during repair: {e}")
        return False

if __name__ == "__main__":
    repair_advanced_feature_selector()
