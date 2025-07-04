#!/usr/bin/env python3
"""
Simple test to check basic imports
"""

import sys
import pandas as pd
import numpy as np

print("✅ Basic imports successful")

try:
    from fast_feature_selector import FastEnterpriseFeatureSelector
    print("✅ FastEnterpriseFeatureSelector import successful")
except Exception as e:
    print(f"❌ FastEnterpriseFeatureSelector import failed: {e}")

try:
    from advanced_feature_selector import AdvancedEnterpriseFeatureSelector
    print("✅ AdvancedEnterpriseFeatureSelector import successful")
except Exception as e:
    print(f"❌ AdvancedEnterpriseFeatureSelector import failed: {e}")

print("🎉 Simple test completed")
