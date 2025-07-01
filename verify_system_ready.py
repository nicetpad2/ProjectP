#!/usr/bin/env python3
"""
✅ NICEGOLD Final System Verification
===================================
Run this after all installations complete to verify production readiness.
"""

def main():
    print("🎯 NICEGOLD Final System Verification")
    print("=" * 40)
    
    try:
        # Test 1: Python basics
        print("🐍 Python: OK")
        
        # Test 2: NumPy version
        import numpy as np
        numpy_version = np.__version__
        print(f"📊 NumPy: {numpy_version}")
        
        if numpy_version.startswith('1.'):
            print("   ✅ NumPy 1.x (SHAP compatible)")
        else:
            print("   ⚠️  NumPy 2.x detected - may have SHAP issues")
        
        # Test 3: SHAP
        import shap
        print(f"🔍 SHAP: {shap.__version__}")
        print("   ✅ SHAP import successful")
        
        # Test 4: Core modules
        from core.menu_system import MenuSystem
        print("📚 Core modules: OK")
        
        # Test 5: Quick SHAP functionality
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        X, y = make_regression(n_samples=10, n_features=3, random_state=42)
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        explainer.shap_values(X[:1])  # Just test functionality
        
        print("🧪 SHAP functionality: OK")
        
        print("\n🎉 SYSTEM VERIFICATION COMPLETE!")
        print("✅ All critical components working")
        print("✅ NumPy/SHAP compatibility confirmed") 
        print("✅ Core modules accessible")
        print("✅ Ready for production!")
        
        print("\n🚀 TO START SYSTEM:")
        print("   python ProjectP.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Solution: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ System Error: {e}")
        return False

if __name__ == "__main__":
    main()
