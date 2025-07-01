#!/usr/bin/env python3
"""
🧪 ENTERPRISE ML PROTECTION SYSTEM TEST
ทดสอบระบบป้องกัน Overfitting, Noise Detection และ Data Leakage
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

# Add project path
sys.path.append('.')

def test_enterprise_protection():
    """ทดสอบระบบป้องกันระดับ Enterprise"""
    try:
        print("🛡️ Testing Enterprise ML Protection System...")
        
        # Import system
        from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
        print("✅ Successfully imported EnterpriseMLProtectionSystem")
        
        # Create test data with known issues
        print("📊 Creating test data with intentional issues...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create datetime series
        dates = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]
        dates.reverse()
        
        # Create features with different characteristics
        data = {
            'date': dates,
            'price': np.cumsum(np.random.randn(n_samples)) + 100,
            'volume': np.random.lognormal(mean=10, sigma=1, size=n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.randn(n_samples),
            
            # Features with issues for testing
            'perfect_predictor': np.zeros(n_samples),  # Will make this perfectly correlated
            'noisy_feature': np.random.randn(n_samples) * 100,  # High noise
            'future_price': np.zeros(n_samples),  # Will contain future information
            'constant_feature': np.ones(n_samples) * 5,  # No variance
        }
        
        df = pd.DataFrame(data)
        
        # Create target (price direction)
        df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
        df = df.dropna()
        
        # Introduce intentional issues for testing
        # 1. Perfect correlation (data leakage)
        df['perfect_predictor'] = df['target'] + np.random.randn(len(df)) * 0.01
        
        # 2. Future information leakage
        df['future_price'] = df['price'].shift(-5)  # Future price leak
        
        # 3. Add outliers (noise)
        outlier_indices = np.random.choice(len(df), size=50, replace=False)
        df.loc[outlier_indices, 'volume'] *= 100  # Extreme outliers
        
        print(f"📈 Created test dataset: {len(df)} samples, {len(df.columns)-1} features")
        
        # Initialize protection system
        protection_system = EnterpriseMLProtectionSystem()
        print("✅ Protection system initialized")
        
        # Prepare features and target
        feature_cols = ['price', 'volume', 'rsi', 'macd', 'perfect_predictor', 
                       'noisy_feature', 'future_price', 'constant_feature']
        X = df[feature_cols]
        y = df['target']
        
        print("🔍 Running comprehensive protection analysis...")
        
        # Run protection analysis
        results = protection_system.comprehensive_protection_analysis(
            X=X,
            y=y,
            model=None,
            datetime_col='date'
        )
        
        print("✅ Protection analysis completed!")
        
        # Analyze results
        print("\n📋 PROTECTION ANALYSIS RESULTS:")
        print("=" * 50)
        
        # Overall assessment
        overall = results.get('overall_assessment', {})
        print(f"🎯 Protection Status: {overall.get('protection_status', 'UNKNOWN')}")
        print(f"⚠️  Risk Level: {overall.get('risk_level', 'UNKNOWN')}")
        print(f"🏢 Enterprise Ready: {overall.get('enterprise_ready', False)}")
        print(f"📊 Overall Risk Score: {overall.get('overall_risk_score', 0):.3f}")
        print(f"⭐ Quality Score: {overall.get('quality_score', 0):.3f}")
        
        # Data leakage results
        leakage = results.get('data_leakage', {})
        print(f"\n🔍 Data Leakage Analysis:")
        print(f"   Status: {leakage.get('status', 'UNKNOWN')}")
        print(f"   Leakage Detected: {leakage.get('leakage_detected', False)}")
        print(f"   Leakage Score: {leakage.get('leakage_score', 0):.3f}")
        
        suspicious_corr = leakage.get('suspicious_correlations', [])
        if suspicious_corr:
            print(f"   Suspicious Correlations: {len(suspicious_corr)}")
            for corr in suspicious_corr[:3]:
                print(f"     • {corr['feature']}: {corr['correlation']:.3f} ({corr['risk_level']})")
        
        # Overfitting results
        overfitting = results.get('overfitting', {})
        print(f"\n📈 Overfitting Analysis:")
        print(f"   Status: {overfitting.get('status', 'UNKNOWN')}")
        print(f"   Overfitting Detected: {overfitting.get('overfitting_detected', False)}")
        print(f"   Overfitting Score: {overfitting.get('overfitting_score', 0):.3f}")
        
        cv_data = overfitting.get('cross_validation', {})
        if cv_data:
            print(f"   CV Mean Score: {cv_data.get('mean_cv_score', 0):.3f}")
            print(f"   CV Std Score: {cv_data.get('std_cv_score', 0):.3f}")
        
        # Noise analysis results
        noise = results.get('noise_analysis', {})
        print(f"\n🎯 Noise Analysis:")
        print(f"   Status: {noise.get('status', 'UNKNOWN')}")
        print(f"   Noise Detected: {noise.get('noise_detected', False)}")
        print(f"   Noise Level: {noise.get('noise_level', 0):.3f}")
        print(f"   Data Quality Score: {noise.get('data_quality_score', 0):.3f}")
        
        # Alerts and recommendations
        alerts = results.get('alerts', [])
        if alerts:
            print(f"\n⚠️  ALERTS ({len(alerts)}):")
            for i, alert in enumerate(alerts[:5], 1):
                print(f"   {i}. {alert}")
        
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. {rec}")
        
        # Test protection report generation
        print(f"\n📄 Generating protection report...")
        report_content = protection_system.generate_protection_report()
        
        if report_content and len(report_content) > 100:
            print("✅ Protection report generated successfully")
            print(f"📝 Report preview (first 200 chars):")
            print(report_content[:200] + "...")
        else:
            print("⚠️  Protection report generation may have issues")
        
        # Validation checks
        print(f"\n🧪 VALIDATION CHECKS:")
        print("=" * 30)
        
        # Check if system detected our intentional issues
        checks = []
        
        # 1. Should detect perfect predictor (data leakage)
        perfect_detected = any(
            corr['feature'] == 'perfect_predictor' and corr['correlation'] > 0.9
            for corr in leakage.get('suspicious_correlations', [])
        )
        checks.append(("Perfect predictor detection", perfect_detected))
        
        # 2. Should detect future information
        future_detected = 'future_price' in leakage.get('future_features', [])
        checks.append(("Future information detection", future_detected))
        
        # 3. Should detect high risk
        high_risk = overall.get('risk_level', 'LOW') in ['HIGH', 'CRITICAL']
        checks.append(("High risk detection", high_risk))
        
        # 4. Should not be enterprise ready with these issues
        not_enterprise_ready = not overall.get('enterprise_ready', True)
        checks.append(("Enterprise readiness (should be False)", not_enterprise_ready))
        
        # Display validation results
        for check_name, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {check_name}: {status}")
        
        # Overall test result
        all_passed = all(passed for _, passed in checks)
        
        print(f"\n🎉 TEST RESULTS:")
        print("=" * 20)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("🛡️ Enterprise ML Protection System is working correctly")
            print("🎯 System successfully detected all intentional issues")
        else:
            print("⚠️  SOME TESTS FAILED")
            print("🔍 Protection system may need adjustments")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 ENTERPRISE ML PROTECTION SYSTEM TEST")
    print("=" * 50)
    
    success = test_enterprise_protection()
    
    print(f"\n🏁 FINAL RESULT: {'SUCCESS' if success else 'FAILURE'}")
    
    if success:
        print("🎉 Enterprise ML Protection System ready for production!")
    else:
        print("🔧 System needs improvements before production use")
