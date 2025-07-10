#!/usr/bin/env python3
"""Simple test for Menu 1"""

import sys
import os
from pathlib import Path

# Add to path  
sys.path.append('/content/drive/MyDrive/ProjectP')

print("Testing import...")
try:
    from core.output_manager import NicegoldOutputManager
    print("✅ NicegoldOutputManager imported")
    
    # Test output manager methods
    om = NicegoldOutputManager()
    print("✅ OutputManager initialized")
    
    # Check methods
    if hasattr(om, 'save_report'):
        print("✅ save_report method exists")
    else:
        print("❌ save_report method missing")
        
    if hasattr(om, 'generate_report'):
        print("❌ generate_report method exists (should not)")
    else:
        print("✅ generate_report method does not exist (correct)")
        
    print("\nAvailable methods:")
    methods = [m for m in dir(om) if not m.startswith('_')]
    for m in methods:
        print(f"  • {m}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
