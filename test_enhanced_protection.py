#!/usr/bin/env python3
"""
🧪 Test Enhanced Enterprise ML Protection System
"""

import sys
import traceback
import os

# Add project root to path
sys.path.insert(0, '/mnt/data/projects/ProjectP')

def test_enterprise_protection():
    """Test the enhanced enterprise ML protection system"""
    try:
        print("🔄 Importing EnterpriseMLProtectionSystem...")
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        print("✅ Import successful")
        
        print("🔄 Initializing protection system...")
        protection_system = EnterpriseMLProtectionSystem()
        print("✅ Initialization successful")
        
        print("🔄 Checking configuration...")
        config = protection_system.protection_config
        print(f"✅ Configuration loaded with {len(config)} parameters")
        
        # Display ALL configuration values  
        print("\n📊 Full Configuration:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
        
        print("\n🧪 Testing with sample data...")
        import pandas as pd
        import numpy as np
        
        # Create sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 15
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y = pd.Series(np.random.randint(0, 2, n_samples))
        
        print(f"📊 Sample data: X={X.shape}, y={y.shape}")
        
        # Run basic analysis
        print("🔄 Running comprehensive analysis...")
        results = protection_system.comprehensive_protection_analysis(X, y)  # Use pandas DataFrames
        print("✅ Analysis completed successfully")
        
        # Display results
        overall = results.get('overall_assessment', {})
        print(f"\n🛡️ Protection Results:")
        print(f"  - Status: {overall.get('protection_status', 'Unknown')}")
        print(f"  - Risk Level: {overall.get('risk_level', 'Unknown')}")
        print(f"  - Enterprise Ready: {overall.get('enterprise_ready', False)}")
        
        # Check specific metrics
        overfitting = results.get('overfitting', {})
        noise = results.get('noise_analysis', {})
        leakage = results.get('data_leakage', {})
        
        print(f"\n📈 Detailed Analysis:")
        # Handle both numeric and string values
        overfitting_score = overfitting.get('overfitting_score', 'N/A')
        if isinstance(overfitting_score, (int, float)):
            print(f"  - Overfitting Score: {overfitting_score:.4f}")
        else:
            print(f"  - Overfitting Score: {overfitting_score}")
            
        noise_level = noise.get('noise_level', 'N/A')
        if isinstance(noise_level, (int, float)):
            print(f"  - Noise Level: {noise_level:.4f}")
        else:
            print(f"  - Noise Level: {noise_level}")
            
        leakage_score = leakage.get('leakage_score', 'N/A')
        if isinstance(leakage_score, (int, float)):
            print(f"  - Leakage Score: {leakage_score:.4f}")
        else:
            print(f"  - Leakage Score: {leakage_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Enhanced Enterprise ML Protection System Test")
    print("=" * 50)
    
    success = test_enterprise_protection()
    
    if success:
        print("\n🎉 All tests passed successfully!")
        print("✅ Enhanced protection system is working correctly")
    else:
        print("\n❌ Tests failed")
        print("🔧 Check the error messages above for debugging")
    
    print("=" * 50)
