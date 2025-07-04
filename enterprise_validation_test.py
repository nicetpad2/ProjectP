#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 COMPREHENSIVE ENTERPRISE VALIDATION TEST
ทดสอบระบบที่ปรับปรุงแล้วเพื่อให้ได้:
- AUC ≥ 70%
- No Noise
- No Data Leakage  
- No Overfitting

รายงานผลลัพธ์แบบครอบคลุมพร้อมคำแนะนำ
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add project paths
sys.path.append('.')
sys.path.append('elliott_wave_modules')

class EnterpriseQualityValidator:
    """🎯 ตัวตรวจสอบคุณภาพระดับองค์กร"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def create_high_quality_test_data(self, n_samples=2000, n_features=30):
        """สร้างข้อมูลทดสอบคุณภาพสูง"""
        print("📊 Creating high-quality test data...")
        
        np.random.seed(42)  # For reproducibility
        
        # Create meaningful features with different characteristics
        data = {}
        
        # 1. Technical indicators (good predictive features)
        data['rsi_14'] = np.random.uniform(20, 80, n_samples)
        data['macd_signal'] = np.random.randn(n_samples)
        data['bb_position'] = np.random.uniform(0, 1, n_samples)
        data['volume_ratio'] = np.random.lognormal(0, 0.5, n_samples)
        data['price_momentum'] = np.random.randn(n_samples)
        
        # 2. Elliott Wave features
        data['wave_position'] = np.random.uniform(0, 8, n_samples)  # Wave 1-8
        data['fibonacci_level'] = np.random.choice([0.236, 0.382, 0.618, 0.786], n_samples)
        data['trend_strength'] = np.random.uniform(0, 1, n_samples)
        
        # 3. Market structure features
        data['support_distance'] = np.random.exponential(0.1, n_samples)
        data['resistance_distance'] = np.random.exponential(0.1, n_samples)
        
        # 4. Add some noise features (should be filtered out)
        for i in range(10, n_features):
            data[f'noise_feature_{i}'] = np.random.randn(n_samples)
        
        X = pd.DataFrame(data)
        
        # Create target with realistic signal
        # Combine multiple features for realistic prediction
        signal_components = [
            X['rsi_14'] / 100,  # RSI normalized
            np.tanh(X['macd_signal']),  # MACD bounded
            X['bb_position'],  # Bollinger Band position
            X['trend_strength'],  # Trend strength
            X['wave_position'] / 8  # Elliott Wave position normalized
        ]
        
        # Combine signals with different weights
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        combined_signal = sum(w * comp for w, comp in zip(weights, signal_components))
        
        # Add controlled noise
        noise_level = 0.3  # 30% noise
        noise = np.random.randn(n_samples) * noise_level
        final_signal = combined_signal + noise
        
        # Convert to probability and then to binary target
        target_prob = 1 / (1 + np.exp(-final_signal))
        y = pd.Series((target_prob > 0.5).astype(int))
        
        print(f"✅ Created test data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Target distribution: {y.value_counts().to_dict()}")
        print(f"📊 Expected signal strength: Medium-High (realistic trading scenario)")
        
        return X, y
    
    def test_enhanced_feature_selector(self, X, y):
        """ทดสอบ Feature Selector ที่ปรับปรุงแล้ว"""
        print("\n🔧 Testing Enhanced Feature Selector...")
        print("=" * 50)
        
        try:
            from feature_selector import EnterpriseShapOptunaFeatureSelector
            
            # Initialize with enterprise parameters
            selector = EnterpriseShapOptunaFeatureSelector(
                target_auc=0.75,       # High target for enterprise quality
                max_features=20,       # Reasonable limit
                n_trials=20,           # Quick test (normally 150+)
                timeout=300            # 5 minutes for test
            )
            
            print(f"✅ Feature Selector initialized with enterprise settings:")
            print(f"   🎯 Target AUC: {selector.target_auc}")
            print(f"   📊 Max Features: {selector.max_features}")
            print(f"   🛡️ Anti-overfitting: {selector.monitor_train_val_gap}")
            print(f"   🔒 Data leakage check: {selector.check_future_leakage}")
            print(f"   📈 Validation strategy: {selector.validation_strategy}")
            
            print("\n⚡ Running feature selection...")
            start_time = time.time()
            
            selected_features, results = selector.select_features(X, y)
            
            execution_time = time.time() - start_time
            
            # Extract results
            best_auc = results.get('best_auc', 0.0)
            validation_results = results.get('validation_results', {})
            
            # Quality metrics
            enterprise_ready = validation_results.get('enterprise_ready', False)
            overfitting_controlled = validation_results.get('overfitting_controlled', False)
            data_leakage_detected = validation_results.get('data_leakage_detected', True)
            quality_score = validation_results.get('quality_score', 0.0)
            enterprise_grade = validation_results.get('enterprise_grade', 'F')
            
            print(f"\n📊 FEATURE SELECTION RESULTS:")
            print(f"   ✅ Selected Features: {len(selected_features)}")
            print(f"   🎯 Best AUC: {best_auc:.4f}")
            print(f"   ⏱️  Execution Time: {execution_time:.1f} seconds")
            print(f"   🏆 Enterprise Grade: {enterprise_grade}")
            print(f"   📈 Quality Score: {quality_score:.3f}")
            
            print(f"\n🛡️ QUALITY CHECKS:")
            print(f"   AUC ≥ 70%: {'✅' if best_auc >= 0.70 else '❌'} ({best_auc:.4f})")
            print(f"   No Overfitting: {'✅' if overfitting_controlled else '❌'}")
            print(f"   No Data Leakage: {'✅' if not data_leakage_detected else '❌'}")
            print(f"   Enterprise Ready: {'✅' if enterprise_ready else '❌'}")
            
            # Check specific thresholds
            checks = {
                'auc_target': best_auc >= 0.70,
                'auc_excellent': best_auc >= 0.75,
                'no_overfitting': overfitting_controlled,
                'no_data_leakage': not data_leakage_detected,
                'enterprise_ready': enterprise_ready,
                'quality_high': quality_score >= 0.8
            }
            
            # Calculate overall success
            critical_checks = ['auc_target', 'no_overfitting', 'no_data_leakage']
            critical_passed = all(checks[check] for check in critical_checks)
            
            overall_success = critical_passed and checks['enterprise_ready']
            
            self.results['feature_selection'] = {
                'success': overall_success,
                'auc': best_auc,
                'features_selected': len(selected_features),
                'execution_time': execution_time,
                'enterprise_grade': enterprise_grade,
                'quality_score': quality_score,
                'checks': checks,
                'selected_features': selected_features[:10],  # First 10 for display
                'validation_details': validation_results
            }
            
            return overall_success, results
            
        except Exception as e:
            print(f"❌ Feature Selection Test FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.results['feature_selection'] = {
                'success': False,
                'error': str(e)
            }
            return False, {}
    
    def test_data_quality_validation(self, X, y):
        """ทดสอบระบบตรวจสอบคุณภาพข้อมูล"""
        print("\n🔍 Testing Data Quality Validation...")
        print("=" * 50)
        
        try:
            # Calculate basic quality metrics
            missing_pct = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
            
            # Check for outliers (simple z-score method)
            from scipy import stats
            outlier_counts = []
            for col in X.select_dtypes(include=[np.number]).columns:
                z_scores = np.abs(stats.zscore(X[col].dropna()))
                outliers = (z_scores > 3).sum()
                outlier_counts.append(outliers)
            
            total_outliers = sum(outlier_counts)
            outlier_pct = total_outliers / (X.shape[0] * X.shape[1])
            
            # Check feature correlations
            corr_matrix = X.corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            # Check target correlations (potential data leakage)
            target_correlations = []
            for col in X.select_dtypes(include=[np.number]).columns:
                corr = abs(X[col].corr(y))
                if corr > 0.95:  # Suspicious correlation
                    target_correlations.append((col, corr))
            
            print(f"📊 DATA QUALITY METRICS:")
            print(f"   Missing Values: {missing_pct:.3%}")
            print(f"   Outliers: {outlier_pct:.3%}")
            print(f"   High Correlations: {len(high_corr_pairs)} pairs")
            print(f"   Suspicious Target Correlations: {len(target_correlations)}")
            
            # Quality assessment
            quality_checks = {
                'low_missing': missing_pct < 0.01,  # < 1% missing
                'low_outliers': outlier_pct < 0.05,  # < 5% outliers
                'low_multicollinearity': len(high_corr_pairs) < 5,
                'no_data_leakage': len(target_correlations) == 0
            }
            
            quality_score = sum(quality_checks.values()) / len(quality_checks)
            
            print(f"\n🛡️ QUALITY ASSESSMENT:")
            for check, passed in quality_checks.items():
                print(f"   {check}: {'✅' if passed else '❌'}")
            print(f"   Overall Quality Score: {quality_score:.1%}")
            
            self.results['data_quality'] = {
                'quality_score': quality_score,
                'missing_pct': missing_pct,
                'outlier_pct': outlier_pct,
                'high_correlations': len(high_corr_pairs),
                'suspicious_correlations': len(target_correlations),
                'checks': quality_checks
            }
            
            return quality_score >= 0.75  # 75% quality threshold
            
        except Exception as e:
            print(f"❌ Data Quality Test FAILED: {str(e)}")
            return False
    
    def generate_enterprise_report(self):
        """สร้างรายงานระดับองค์กร"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("🏆 ENTERPRISE VALIDATION REPORT")
        print("="*80)
        print(f"📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total Test Time: {total_time:.1f} seconds")
        
        # Feature Selection Results
        fs_results = self.results.get('feature_selection', {})
        if fs_results.get('success', False):
            print(f"\n✅ FEATURE SELECTION: SUCCESS")
            print(f"   🎯 AUC Achieved: {fs_results['auc']:.4f}")
            print(f"   📊 Features Selected: {fs_results['features_selected']}")
            print(f"   🏆 Enterprise Grade: {fs_results['enterprise_grade']}")
            print(f"   📈 Quality Score: {fs_results['quality_score']:.3f}")
        else:
            print(f"\n❌ FEATURE SELECTION: FAILED")
            print(f"   Error: {fs_results.get('error', 'Unknown error')}")
        
        # Data Quality Results
        dq_results = self.results.get('data_quality', {})
        if dq_results:
            print(f"\n📊 DATA QUALITY ASSESSMENT:")
            print(f"   Overall Quality: {dq_results['quality_score']:.1%}")
            print(f"   Missing Values: {dq_results['missing_pct']:.3%}")
            print(f"   Outlier Level: {dq_results['outlier_pct']:.3%}")
            print(f"   Data Leakage Risk: {'Low' if dq_results['suspicious_correlations'] == 0 else 'High'}")
        
        # Overall Assessment
        fs_success = fs_results.get('success', False)
        auc_target = fs_results.get('auc', 0) >= 0.70
        quality_good = dq_results.get('quality_score', 0) >= 0.75
        
        overall_success = fs_success and auc_target and quality_good
        
        print(f"\n🎯 OVERALL ASSESSMENT:")
        print(f"   AUC ≥ 70%: {'✅' if auc_target else '❌'}")
        print(f"   No Noise: {'✅' if quality_good else '❌'}")
        print(f"   No Data Leakage: {'✅' if dq_results.get('suspicious_correlations', 1) == 0 else '❌'}")
        print(f"   No Overfitting: {'✅' if fs_results.get('checks', {}).get('no_overfitting', False) else '❌'}")
        
        print(f"\n🏆 FINAL RESULT:")
        if overall_success:
            print("🎉 ENTERPRISE VALIDATION: SUCCESS!")
            print("✅ ระบบพร้อมใช้งานระดับองค์กรแล้ว")
            print("🚀 ผ่านการทดสอบทั้ง 4 เกณฑ์หลัก")
        else:
            print("⚠️ ENTERPRISE VALIDATION: PARTIAL SUCCESS")
            print("🔧 ระบบยังต้องปรับปรุงเพิ่มเติม")
        
        print("="*80)
        return overall_success

def main():
    """Main validation function"""
    print("🎯 COMPREHENSIVE ENTERPRISE VALIDATION TEST")
    print("=" * 80)
    print("เป้าหมาย: AUC ≥ 70%, No Noise, No Data Leakage, No Overfitting")
    print("=" * 80)
    
    validator = EnterpriseQualityValidator()
    
    # Create test data
    X, y = validator.create_high_quality_test_data()
    
    # Run tests
    test_results = []
    
    # Test 1: Data Quality
    dq_success = validator.test_data_quality_validation(X, y)
    test_results.append(("Data Quality", dq_success))
    
    # Test 2: Enhanced Feature Selection
    fs_success, fs_details = validator.test_enhanced_feature_selector(X, y)
    test_results.append(("Feature Selection", fs_success))
    
    # Generate comprehensive report
    overall_success = validator.generate_enterprise_report()
    
    # Summary
    print(f"\n📋 TEST SUMMARY:")
    for test_name, success in test_results:
        print(f"   {test_name}: {'✅ PASS' if success else '❌ FAIL'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
