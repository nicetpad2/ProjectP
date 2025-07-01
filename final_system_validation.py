#!/usr/bin/env python3
"""
Final System Validation and Menu 1 Test
Run this after the NumPy fix is complete
"""
import sys
import os
from datetime import datetime

def test_numpy_complete():
    """Complete NumPy test including DLL issues"""
    print("🔍 TESTING NUMPY INSTALLATION")
    print("=" * 50)
    
    try:
        # Basic import
        import numpy as np
        print(f"✅ NumPy import successful: {np.__version__}")
        print(f"✅ NumPy location: {np.__file__}")
        
        # Test the specific problematic module
        print("\n🔍 Testing _umath_linalg (the problematic module)...")
        from numpy.linalg import _umath_linalg
        print("✅ _umath_linalg import successful!")
        
        # Test basic operations
        print("\n🔍 Testing NumPy operations...")
        arr = np.array([1, 2, 3, 4, 5])
        dot_result = np.dot(arr, arr)
        print(f"✅ Dot product: {dot_result}")
        
        # Test linear algebra (this is where the DLL error occurred)
        print("\n🔍 Testing NumPy linear algebra...")
        matrix = np.array([[1, 2], [3, 4]], dtype=float)
        det = np.linalg.det(matrix)
        eigenvals = np.linalg.eigvals(matrix)
        print(f"✅ Matrix determinant: {det}")
        print(f"✅ Eigenvalues: {eigenvals}")
        
        # Test matrix operations that use _umath_linalg
        print("\n🔍 Testing advanced linear algebra...")
        inv_matrix = np.linalg.inv(matrix)
        svd_result = np.linalg.svd(matrix)
        print(f"✅ Matrix inverse: {inv_matrix}")
        print(f"✅ SVD decomposition successful")
        
        print("\n🎉 ALL NUMPY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shap():
    """Test SHAP installation"""
    print("\n🔍 TESTING SHAP INSTALLATION")
    print("=" * 50)
    
    try:
        import shap
        print(f"✅ SHAP import successful: {shap.__version__}")
        print(f"✅ SHAP location: {shap.__file__}")
        
        # Test basic SHAP functionality
        print("\n🔍 Testing SHAP basic functionality...")
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        
        # Create simple test data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Test SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X[:5])  # Test with first 5 samples
        
        print(f"✅ SHAP explainer works: {shap_values.shape}")
        print("🎉 SHAP TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ SHAP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_menu_1():
    """Test Menu 1 import and initialization"""
    print("\n🔍 TESTING MENU 1 AVAILABILITY")
    print("=" * 50)
    
    try:
        # Set up minimal environment
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        # Test import
        from menu_modules.menu_1_elliott_wave import Menu1ElliottWave
        print("✅ Menu 1 import successful!")
        
        # Test initialization
        menu1 = Menu1ElliottWave()
        print("✅ Menu 1 initialization successful!")
        
        print("🎉 MENU 1 IS READY!")
        return True
        
    except Exception as e:
        print(f"❌ Menu 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_availability():
    """Test if data files are available"""
    print("\n🔍 TESTING DATA AVAILABILITY")
    print("=" * 50)
    
    data_files = [
        "datacsv/XAUUSD_M1.csv",
        "datacsv/XAUUSD_M15.csv"
    ]
    
    all_data_available = True
    for file_path in data_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {file_size:,} bytes")
        else:
            print(f"❌ {file_path}: Missing")
            all_data_available = False
    
    if all_data_available:
        print("🎉 ALL DATA FILES AVAILABLE!")
    else:
        print("⚠️ Some data files are missing")
    
    return all_data_available

def main():
    """Run complete system validation"""
    print("🏢 NICEGOLD ENTERPRISE SYSTEM VALIDATION")
    print("=" * 80)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Working Directory: {os.getcwd()}")
    print("=" * 80)
    
    results = {
        "numpy": test_numpy_complete(),
        "shap": test_shap(),
        "menu_1": test_menu_1(),
        "data": test_data_availability()
    }
    
    print("\n" + "=" * 80)
    print("📊 FINAL VALIDATION RESULTS")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name.upper()}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 SYSTEM IS READY FOR PRODUCTION!")
        print("\n✨ You can now run:")
        print("   python ProjectP.py")
        print("   Select Menu 1 - Full Pipeline")
        print("\n🏆 NICEGOLD ENTERPRISE SYSTEM IS 100% OPERATIONAL!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("🔧 Additional fixes may be required")
        
        # Provide specific guidance
        if not results["numpy"]:
            print("\n🔧 NumPy Fix Required:")
            print("   - Install Visual C++ Redistributables")
            print("   - Try: python windows_numpy_fix.py")
            print("   - Consider using Anaconda/Miniconda")
        
        if not results["shap"]:
            print("\n🔧 SHAP Fix Required:")
            print("   - Ensure NumPy is working first")
            print("   - Try: pip install shap==0.45.0")
        
        if not results["menu_1"]:
            print("\n🔧 Menu 1 Fix Required:")
            print("   - Ensure NumPy and SHAP are working")
            print("   - Check elliott_wave_modules dependencies")

if __name__ == "__main__":
    main()
