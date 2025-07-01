#!/usr/bin/env python3
"""Test NumPy import and identify specific DLL errors"""
import sys
import traceback

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("=" * 50)

try:
    print("Testing NumPy import...")
    import numpy
    print("✓ NumPy import successful!")
    print("NumPy version:", numpy.__version__)
    print("NumPy location:", numpy.__file__)
    
    # Test the specific problematic import
    try:
        print("\nTesting _umath_linalg import...")
        from numpy.linalg import _umath_linalg
        print("✓ _umath_linalg import successful!")
    except Exception as e:
        print("✗ _umath_linalg import failed:", str(e))
        traceback.print_exc()
        
    # Test basic operations
    try:
        print("\nTesting NumPy operations...")
        arr = numpy.array([1, 2, 3])
        result = numpy.dot(arr, arr)
        print("✓ NumPy operations successful! Result:", result)
    except Exception as e:
        print("✗ NumPy operations failed:", str(e))
        traceback.print_exc()
        
except ImportError as e:
    print("✗ NumPy import failed with ImportError:", str(e))
    traceback.print_exc()
except Exception as e:
    print("✗ NumPy import failed with unexpected error:", str(e))
    traceback.print_exc()

print("\n" + "=" * 50)
print("Testing SHAP import...")
try:
    import shap
    print("✓ SHAP import successful!")
    print("SHAP version:", shap.__version__)
except Exception as e:
    print("✗ SHAP import failed:", str(e))
    traceback.print_exc()
