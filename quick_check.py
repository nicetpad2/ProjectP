#!/usr/bin/env python3
"""Quick validation of feature selector files"""

print("Checking feature selector files...")

# Check if files exist
import os
files_to_check = [
    'real_profit_feature_selector.py',
    'advanced_feature_selector.py', 
    'fast_feature_selector.py',
    'elliott_wave_modules/feature_selector.py'
]

for file in files_to_check:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✅ {file} exists ({size} bytes)")
    else:
        print(f"❌ {file} missing")

print("Basic validation complete.")
