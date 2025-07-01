#!/usr/bin/env python3
"""Quick NumPy test"""
try:
    import numpy as np
    print("✓ NumPy import successful!")
    print(f"Version: {np.__version__}")
    
    # Test the problematic module
    from numpy.linalg import _umath_linalg
    print("✓ _umath_linalg import successful!")
    
    # Test operations
    test_array = np.array([1, 2, 3])
    result = np.dot(test_array, test_array)
    print(f"✓ NumPy operations work: {result}")
    
    # Test SHAP
    import shap
    print(f"✓ SHAP import successful! Version: {shap.__version__}")
    
    print("\n🎉 ALL TESTS PASSED! NumPy DLL issue is RESOLVED!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
