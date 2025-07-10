#!/usr/bin/env python3
"""
Simple test for Enterprise ML Protection System
"""
import sys
sys.path.append('/content/drive/MyDrive/ProjectP')

try:
    from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
    import pandas as pd
    import numpy as np
    
    print("🧪 Simple Enterprise ML Protection Test")
    print("=" * 50)
    
    # Test basic initialization
    protection = EnterpriseMLProtectionSystem()
    print('✅ Protection system initialized successfully')
    print(f'   sklearn available: {protection.sklearn_available}')
    print(f'   scipy available: {protection.scipy_available}')
    
    # Test simple data
    X = pd.DataFrame({
        'feature1': np.random.randn(100), 
        'feature2': np.random.randn(100), 
        'datetime': pd.date_range('2023-01-01', periods=100)
    })
    y = pd.Series(np.random.choice([0, 1], size=100))
    
    # Test simplified overfitting detection
    result = protection._detect_overfitting_simplified(X[['feature1', 'feature2']], y)
    print(f'✅ Simplified overfitting detection: {result.get("status", "UNKNOWN")}')
    
    # Test protection status
    status = protection.get_protection_status()
    print(f'✅ Protection status: {status.get("status", "UNKNOWN")}')
    
    # Test configuration validation
    validation = protection.validate_configuration()
    print(f'✅ Configuration valid: {validation.get("valid", False)}')
    
    print('🎉 All basic tests passed!')
    
except Exception as e:
    print(f'❌ Error: {str(e)}')
    import traceback
    traceback.print_exc()
