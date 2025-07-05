#!/usr/bin/env python3
"""Test script to identify the exact syntax error"""

import sys
import ast

def check_syntax(filename):
    """Check syntax of Python file and report exact error location"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to parse
        ast.parse(content)
        print("Syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"Syntax error found:")
        print(f"File: {filename}")
        print(f"Line: {e.lineno}")
        print(f"Column: {e.offset}")
        print(f"Message: {e.msg}")
        if e.text:
            print(f"Text: {e.text.strip()}")
        return False
        
    except Exception as e:
        print(f"Other error: {e}")
        return False

if __name__ == "__main__":
    check_syntax('/mnt/data/projects/ProjectP/advanced_feature_selector.py')
