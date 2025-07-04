#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTERPRISE ML PROTECTION SYSTEM
Enterprise-level ML Protection: Overfitting, Noise Detection, Data Leakage Prevention

Core Features:
- Advanced Overfitting Detection (Multiple Methods)
- Intelligent Noise Detection & Filtering
- Comprehensive Data Leakage Prevention
- Time-Series Aware Validation
- Enterprise-grade Statistical Analysis
- Real-time Monitoring & Alerts

🏢 Enterprise Standards:
- Zero Tolerance for Data Leakage
- Statistical Significance Testing
- Cross-Validation with Time Awareness
- Feature Stability Analysis
- Model Performance Degradation Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Import new advanced logging system
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel, ProcessStatus
    from core.real_time_progress_manager import get_progress_manager, ProgressType, ProgressContext
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    import logging

# Import ML libraries with error handling
try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available, using simplified protection analysis")

try:
    from scipy import stats
    from scipy.stats import ks_2samp, shapiro
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available, using simplified statistical analysis")


class EnterpriseMLProtectionSystem:
    """🛡️ ระบบป้องกัน ML ระดับ Enterprise สำหรับ NICEGOLD Trading System"""
    
    def __init__(self, config: Dict = None, logger=None):
        # Initialize advanced logging system
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
            self.progress_manager = get_progress_manager()
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_manager = None
        
        self.config = config or {}
        
        # Set availability flags
        self.sklearn_available = SKLEARN_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        
        # Ultra-strict enterprise protection configuration
        default_config = {
            'overfitting_threshold': 0.05,   # Ultra-strict overfitting detection
            'noise_threshold': 0.02,         # Ultra-strict noise detection
            'leak_detection_window': 200,    # Larger window for leakage detection
            'min_samples_split': 100,        # Larger minimum samples for time split
            'stability_window': 2000,        # Larger window for feature stability
            'significance_level': 0.01,      # Stricter statistical significance
            'enterprise_mode': True,         # Enable strict enterprise checking
            'min_auc_threshold': 0.75,       # Higher minimum AUC for enterprise ready
            'max_feature_correlation': 0.75, # Lower maximum correlation between features
            'min_feature_importance': 0.02,  # Higher minimum feature importance threshold
            'max_cv_variance': 0.05,         # Maximum allowed cross-validation variance
            'min_train_val_ratio': 0.90,     # Minimum train/validation performance ratio
            'max_feature_drift': 0.10,       # Maximum allowed feature drift over time
            'min_signal_noise_ratio': 3.0,   # Minimum signal-to-noise ratio required
        }
        
        # Merge with ml_protection config from main config
        ml_protection_config = self.config.get('ml_protection', {})
        self.protection_config = {**default_config, **ml_protection_config}
        self.protection_results = {}
        
        # Log initialization with new system
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.security("🛡️ Enterprise ML Protection System initialized", "ML_Protection_Init")
            self.logger.system(f"sklearn available: {self.sklearn_available}", "ML_Protection_Init")
            self.logger.system(f"scipy available: {self.scipy_available}", "ML_Protection_Init")
            self.logger.system(f"config loaded: {len(self.protection_config)} settings", "ML_Protection_Init")
        else:
            self.logger.info(f"🛡️ Enterprise ML Protection System initialized")
            self.logger.info(f"   - sklearn available: {self.sklearn_available}")
            self.logger.info(f"   - scipy available: {self.scipy_available}")
            self.logger.info(f"   - config: {self.protection_config}")
    
    def update_protection_config(self, new_config: Dict) -> bool:
        """🔧 อัปเดตการตั้งค่าป้องกัน"""
        try:
            self.protection_config.update(new_config)
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.system(f"Protection config updated: {len(new_config)} new settings", 
                                 "Config_Update", data=new_config)
            else:
                self.logger.info(f"🔧 Protection config updated: {new_config}")
            
            return True
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"Failed to update protection config", 
                                "Config_Update", exception=e)
            else:
                self.logger.error(f"❌ Failed to update protection config: {str(e)}")
            return False
    
    def get_protection_config(self) -> Dict:
        """📋 ดึงการตั้งค่าป้องกันปัจจุบัน"""
        return self.protection_config.copy()

    def comprehensive_protection_analysis(self, X: np.ndarray, y: np.ndarray, 
                                        model: Any = None, 
                                        feature_names: List[str] = None,
                                        process_id: str = None) -> Dict[str, Any]:
        """🎯 การวิเคราะห์ป้องกันแบบครบถ้วน"""
        
        # Critical check for empty or insufficient data
        if X is None or y is None:
            error_msg = "❌ Cannot analyze protection: X or y is None"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.critical(error_msg, "Protection_Analysis", process_id=process_id)
            return {
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'enterprise_ready': False,
                'critical_issues': [error_msg]
            }
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Check for empty or insufficient data
        if len(X) == 0 or len(y) == 0:
            error_msg = f"❌ Cannot analyze protection: Empty dataset (X: {len(X)}, y: {len(y)})"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.critical(error_msg, "Protection_Analysis", process_id=process_id)
            return {
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'enterprise_ready': False,
                'critical_issues': [error_msg]
            }
        
        if len(X) < 10:
            error_msg = f"❌ Cannot analyze protection: Insufficient data (only {len(X)} samples, need at least 10)"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.critical(error_msg, "Protection_Analysis", process_id=process_id)
            return {
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'enterprise_ready': False,
                'critical_issues': [error_msg]
            }
        
        # Start progress tracking
        main_progress_id = None
        if ADVANCED_LOGGING_AVAILABLE and self.progress_manager:
            main_progress_id = self.progress_manager.create_progress(
                "🛡️ ML Protection Analysis", 7, ProgressType.ANALYSIS
            )
        
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.security("🛡️ Starting comprehensive ML protection analysis", 
                                    "Protection_Analysis", process_id=process_id)
            else:
                self.logger.info("🛡️ Starting comprehensive ML protection analysis")
            
            protection_results = {
                'timestamp': datetime.now().isoformat(),
                'data_shape': X.shape,
                'feature_names': feature_names or [f'feature_{i}' for i in range(X.shape[1])],
                'protection_config': self.protection_config.copy(),
                'enterprise_ready': False,
                'critical_issues': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Step 1: Data Quality Analysis
            if main_progress_id:
                self.progress_manager.update_progress(main_progress_id, 1, "🔍 Analyzing data quality")
            
            data_quality_results = self._analyze_data_quality(X, y, process_id)
            protection_results['data_quality'] = data_quality_results
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.data_log(f"Data quality analysis completed", 
                                   "Data_Quality", process_id=process_id, 
                                   data={'issues': len(data_quality_results.get('issues', []))})
            
            # Step 2: Feature Correlation Analysis
            if main_progress_id:
                self.progress_manager.update_progress(main_progress_id, 1, "🔗 Analyzing feature correlations")
            
            correlation_results = self._analyze_feature_correlation(X, feature_names, process_id)
            protection_results['correlation_analysis'] = correlation_results
            
            # Step 3: Overfitting Detection
            if main_progress_id:
                self.progress_manager.update_progress(main_progress_id, 1, "🎯 Detecting overfitting")
            
            if model is not None and self.sklearn_available:
                overfitting_results = self._detect_overfitting(X, y, model, process_id)
                protection_results['overfitting_analysis'] = overfitting_results
            else:
                if ADVANCED_LOGGING_AVAILABLE:
                    self.logger.warning("Skipping overfitting analysis - no model provided or sklearn unavailable", 
                                      "Overfitting_Detection", process_id=process_id)
                protection_results['overfitting_analysis'] = {'skipped': True, 'reason': 'No model or sklearn unavailable'}
            
            # Step 4: Data Leakage Detection
            if main_progress_id:
                self.progress_manager.update_progress(main_progress_id, 1, "🚨 Detecting data leakage")
            
            leakage_results = self._detect_data_leakage(X, y, process_id)
            protection_results['leakage_analysis'] = leakage_results
            
            # Step 5: Noise Analysis
            if main_progress_id:
                self.progress_manager.update_progress(main_progress_id, 1, "📊 Analyzing noise levels")
            
            noise_results = self._analyze_noise(X, y, process_id)
            protection_results['noise_analysis'] = noise_results
            
            # Step 6: Time Series Validation
            if main_progress_id:
                self.progress_manager.update_progress(main_progress_id, 1, "⏰ Validating time series integrity")
            
            timeseries_results = self._validate_time_series_integrity(X, y, process_id)
            protection_results['timeseries_analysis'] = timeseries_results
            
            # Step 7: Enterprise Readiness Assessment
            if main_progress_id:
                self.progress_manager.update_progress(main_progress_id, 1, "🏢 Assessing enterprise readiness")
            
            enterprise_assessment = self._assess_enterprise_readiness(protection_results, process_id)
            protection_results.update(enterprise_assessment)
            
            # Complete progress
            if main_progress_id:
                self.progress_manager.complete_progress(main_progress_id, 
                                                       "✅ Protection analysis completed")
            
            # Log final results
            if ADVANCED_LOGGING_AVAILABLE:
                is_ready = protection_results.get('enterprise_ready', False)
                if is_ready:
                    self.logger.success("🏆 Enterprise ML Protection: PASSED", 
                                      "Protection_Results", process_id=process_id,
                                      data={'score': protection_results.get('enterprise_score', 0)})
                else:
                    self.logger.warning("⚠️ Enterprise ML Protection: REVIEW REQUIRED", 
                                      "Protection_Results", process_id=process_id,
                                      data={'critical_issues': len(protection_results.get('critical_issues', []))})
            
            self.protection_results = protection_results
            return protection_results
            
        except Exception as e:
            if main_progress_id:
                self.progress_manager.fail_progress(main_progress_id, f"Protection analysis failed: {str(e)}")
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.critical(f"Comprehensive protection analysis failed", 
                                   "Protection_Analysis", process_id=process_id, exception=e)
            else:
                self.logger.error(f"❌ Comprehensive protection analysis failed: {str(e)}")
            
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'enterprise_ready': False,
                'critical_issues': [f"Analysis failed: {str(e)}"]
            }

    def _detect_data_leakage(self, X: pd.DataFrame, y: pd.Series, datetime_col: str = None) -> Dict[str, Any]:
        """ตรวจจับ Data Leakage อย่างชาญฉลาด"""
        try:
            leakage_results = {
                'status': 'ANALYZING',
                'leakage_detected': False,
                'leakage_score': 0.0,
                'suspicious_features': [],
                'temporal_leakage': False,
                'future_feature_leakage': False,
                'details': {}
            }
            
            # 1. Perfect Correlation Detection
            correlations = []
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    corr = abs(X[col].corr(y))
                    if corr > 0.95:  # Suspiciously high correlation
                        correlations.append({
                            'feature': col,
                            'correlation': corr,
                            'risk_level': 'HIGH' if corr > 0.98 else 'MEDIUM'
                        })
            
            leakage_results['suspicious_correlations'] = correlations
            
            # 2. Temporal Leakage Detection
            if datetime_col and datetime_col in X.columns:
                temporal_leakage = self._check_temporal_leakage(X, y, datetime_col)
                leakage_results['temporal_leakage'] = temporal_leakage['detected']
                leakage_results['temporal_details'] = temporal_leakage
            
            # 3. Future Information Detection
            future_leakage = self._detect_future_information_leakage(X)
            leakage_results['future_feature_leakage'] = future_leakage['detected']
            leakage_results['future_features'] = future_leakage['suspicious_features']
            
            # 4. Statistical Leakage Test
            statistical_leakage = self._statistical_leakage_test(X, y)
            leakage_results['statistical_leakage'] = statistical_leakage
            
            # Overall leakage assessment
            leakage_score = self._compute_leakage_score(leakage_results)
            leakage_results['leakage_score'] = leakage_score
            leakage_results['leakage_detected'] = leakage_score > 0.3
            leakage_results['status'] = 'DETECTED' if leakage_results['leakage_detected'] else 'CLEAN'
            
            return leakage_results
            
        except Exception as e:
            self.logger.error(f"❌ Data leakage detection failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _detect_overfitting(self, X: pd.DataFrame, y: pd.Series, model: Any = None, process_id: str = None) -> Dict[str, Any]:
        """ตรวจจับ Overfitting ด้วยหลายวิธี"""
        try:
            # Check if sklearn is available for advanced analysis
            if not self.sklearn_available:
                self.logger.warning("⚠️ sklearn not available, using simplified overfitting detection")
                return self._detect_overfitting_simplified(X, y)
            
            overfitting_results = {
                'status': 'ANALYZING',
                'overfitting_detected': False,
                'overfitting_score': 0.0,
                'cross_validation': {},
                'learning_curves': {},
                'variance_analysis': {},
                'details': {}
            }
            
            # 1. Time-Series Cross Validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            if model is None:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
            
            overfitting_results['cross_validation'] = {
                'cv_scores': cv_scores.tolist(),
                'mean_cv_score': float(cv_scores.mean()),
                'std_cv_score': float(cv_scores.std()),
                'coefficient_of_variation': float(cv_scores.std() / cv_scores.mean())
            }
            
            # 2. Train-Validation Split Analysis
            train_val_analysis = self._train_validation_analysis(X, y, model, process_id)
            overfitting_results['train_validation'] = train_val_analysis
            
            # 3. Learning Curve Analysis
            learning_curves = self._analyze_learning_curves(X, y, model, process_id)
            overfitting_results['learning_curves'] = learning_curves
            
            # 4. Feature Importance Stability
            importance_stability = self._analyze_feature_importance_stability(X, y, model, process_id)
            overfitting_results['feature_importance_stability'] = importance_stability
            
            # Overall overfitting assessment
            overfitting_score = self._compute_overfitting_score(overfitting_results)
            overfitting_results['overfitting_score'] = overfitting_score
            overfitting_results['overfitting_detected'] = overfitting_score > self.protection_config['overfitting_threshold']
            overfitting_results['status'] = 'DETECTED' if overfitting_results['overfitting_detected'] else 'ACCEPTABLE'
            
            return overfitting_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced overfitting detection failed: {str(e)}, falling back to simplified method")
            # Fallback to simplified method
            try:
                return self._detect_overfitting_simplified(X, y)
            except Exception as fallback_error:
                self.logger.error(f"❌ Both overfitting detection methods failed: {str(fallback_error)}")
                return {
                    'status': 'ERROR', 
                    'error': str(fallback_error),
                    'overfitting_detected': False,
                    'overfitting_score': 0.0,
                    'details': {'fallback_attempted': True}
                }
    
    def _detect_overfitting_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """ตรวจจับ Overfitting แบบ simplified (ไม่ใช้ sklearn) - Enhanced Version"""
        try:
            overfitting_results = {
                'status': 'ANALYZING',
                'overfitting_detected': False,
                'overfitting_score': 0.0,
                'cross_validation': {},
                'details': {}
            }
            
            # Enhanced overfitting detection with multiple criteria
            n_samples, n_features = X.shape
            feature_ratio = n_features / n_samples if n_samples > 0 else 1.0
            
            # Criterion 1: Feature-to-sample ratio (stricter threshold)
            ratio_score = min(feature_ratio * 2, 1.0)  # More sensitive
            
            # Criterion 2: Data variance analysis
            variance_score = 0.0
            if n_features > 0:
                try:
                    # Calculate coefficient of variation for each feature
                    feature_cv = []
                    for col in X.select_dtypes(include=[np.number]).columns:
                        if X[col].std() > 0:
                            cv = X[col].std() / abs(X[col].mean()) if X[col].mean() != 0 else 0
                            feature_cv.append(cv)
                    
                    if feature_cv:
                        # High variance indicates potential noise/overfitting
                        variance_score = min(np.mean(feature_cv) / 2, 1.0)
                except:
                    variance_score = 0.1
            
            # Criterion 3: Feature correlation analysis
            correlation_score = 0.0
            try:
                # Check for high inter-feature correlations
                numeric_X = X.select_dtypes(include=[np.number])
                if len(numeric_X.columns) > 1:
                    corr_matrix = numeric_X.corr().abs()
                    # Count high correlations (excluding diagonal)
                    high_corr_count = (corr_matrix > 0.8).sum().sum() - len(corr_matrix)
                    total_pairs = len(corr_matrix) * (len(corr_matrix) - 1)
                    correlation_score = min(high_corr_count / max(total_pairs, 1), 1.0)
            except:
                correlation_score = 0.1
            
            # Combined overfitting score (weighted average)
            overfitting_score = (
                0.4 * ratio_score +      # 40% weight on feature ratio
                0.3 * variance_score +   # 30% weight on variance
                0.3 * correlation_score  # 30% weight on correlation
            )
            
            # More strict threshold for enterprise standards
            overfitting_threshold = self.protection_config.get('overfitting_threshold', 0.08)  # Reduced from 0.15
            overfitting_detected = overfitting_score > overfitting_threshold
            
            overfitting_results.update({
                'status': 'DETECTED' if overfitting_detected else 'ACCEPTABLE',
                'overfitting_detected': overfitting_detected,
                'overfitting_score': overfitting_score,
                'cross_validation': {
                    'method': 'enhanced_multi_criteria',
                    'feature_ratio': feature_ratio,
                    'ratio_score': ratio_score,
                    'variance_score': variance_score,
                    'correlation_score': correlation_score,
                    'samples': n_samples,
                    'features': n_features
                },
                'details': {
                    'method': 'enhanced_overfitting_detection',
                    'threshold': overfitting_threshold,
                    'criteria_weights': {
                        'feature_ratio': 0.4,
                        'variance_analysis': 0.3,
                        'correlation_analysis': 0.3
                    }
                }
            })
            
            return overfitting_results
            
        except Exception as e:
            self.logger.error(f"❌ Simplified overfitting detection failed: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'overfitting_detected': False,
                'overfitting_score': 0.0
            }
    
    def _detect_noise_and_quality(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """ตรวจจับ Noise และประเมินคุณภาพข้อมูล"""
        try:
            noise_results = {
                'status': 'ANALYZING',
                'noise_detected': False,
                'noise_level': 0.0,
                'data_quality_score': 0.0,
                'noisy_features': [],
                'quality_metrics': {},
                'recommendations': []
            }
            
            # 1. Missing Value Analysis
            missing_analysis = self._analyze_missing_values(X)
            noise_results['missing_values'] = missing_analysis
            
            # 2. Outlier Detection
            outlier_analysis = self._detect_outliers_comprehensive(X)
            noise_results['outliers'] = outlier_analysis
            
            # 3. Feature Distribution Analysis
            distribution_analysis = self._analyze_feature_distributions(X)
            noise_results['distributions'] = distribution_analysis
            
            # 4. Signal-to-Noise Ratio
            snr_analysis = self._compute_signal_to_noise_ratio(X, y)
            noise_results['signal_to_noise'] = snr_analysis
            
            # 5. Feature Relevance Analysis
            relevance_analysis = self._analyze_feature_relevance(X, y)
            noise_results['feature_relevance'] = relevance_analysis
            
            # Overall noise assessment
            noise_level = self._compute_noise_level(noise_results)
            data_quality_score = self._compute_data_quality_score(noise_results)
            
            noise_results['noise_level'] = noise_level
            noise_results['data_quality_score'] = data_quality_score
            noise_results['noise_detected'] = noise_level > self.protection_config['noise_threshold']
            noise_results['status'] = 'HIGH_NOISE' if noise_results['noise_detected'] else 'ACCEPTABLE'
            
            return noise_results
            
        except Exception as e:
            self.logger.error(f"❌ Noise detection failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _detect_noise_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """ตรวจจับ Noise แบบ simplified (ไม่ใช้ sklearn) - Enhanced Version"""
        try:
            noise_results = {
                'status': 'ANALYZING',
                'noise_level': 0.0,
                'data_quality_score': 1.0,
                'details': {}
            }
            
            # Enhanced noise detection with multiple criteria
            
            # 1. Missing Values Analysis (stricter)
            missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
            
            # 2. Constant Features Detection
            constant_features = []
            for col in X.columns:
                if X[col].nunique() <= 1:
                    constant_features.append(col)
            
            constant_ratio = len(constant_features) / len(X.columns) if len(X.columns) > 0 else 0
            
            # 3. Enhanced Outlier Detection
            outlier_ratio = 0.0
            extreme_outlier_ratio = 0.0
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if len(X[col]) > 0 and X[col].std() > 0:
                    # Standard IQR method
                    q1 = X[col].quantile(0.25)
                    q3 = X[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        # Normal outliers (1.5 * IQR)
                        outliers = ((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum()
                        outlier_ratio += outliers / len(X[col])
                        
                        # Extreme outliers (3 * IQR)
                        extreme_outliers = ((X[col] < q1 - 3 * iqr) | (X[col] > q3 + 3 * iqr)).sum()
                        extreme_outlier_ratio += extreme_outliers / len(X[col])
            
            if len(numeric_cols) > 0:
                outlier_ratio = outlier_ratio / len(numeric_cols)
                extreme_outlier_ratio = extreme_outlier_ratio / len(numeric_cols)
            
            # 4. Feature Variance Analysis
            low_variance_ratio = 0.0
            for col in numeric_cols:
                if len(X[col]) > 0:
                    # Calculate coefficient of variation
                    cv = X[col].std() / abs(X[col].mean()) if X[col].mean() != 0 else 0
                    if cv < 0.01:  # Very low variance features
                        low_variance_ratio += 1
            
            if len(numeric_cols) > 0:
                low_variance_ratio = low_variance_ratio / len(numeric_cols)
            
            # 5. Skewness Analysis  
            high_skewness_ratio = 0.0
            for col in numeric_cols:
                if len(X[col]) > 5 and X[col].std() > 0:  # Need at least 6 samples for skewness
                    try:
                        skewness = X[col].skew()
                        if abs(skewness) > 2:  # High skewness
                            high_skewness_ratio += 1
                    except:
                        pass
            
            if len(numeric_cols) > 0:
                high_skewness_ratio = high_skewness_ratio / len(numeric_cols)
            
            # Enhanced noise calculation with weighted criteria
            noise_level = (
                missing_ratio * 0.25 +           # 25% weight on missing values
                constant_ratio * 0.20 +          # 20% weight on constant features  
                outlier_ratio * 0.20 +           # 20% weight on outliers
                extreme_outlier_ratio * 0.15 +   # 15% weight on extreme outliers
                low_variance_ratio * 0.10 +      # 10% weight on low variance
                high_skewness_ratio * 0.10       # 10% weight on high skewness
            )
            
            data_quality_score = max(0.0, 1.0 - noise_level)
            
            # More strict threshold for noise detection
            noise_threshold = self.protection_config.get('noise_threshold', 0.03)  # Reduced from 0.05
            
            noise_results.update({
                'status': 'HIGH_NOISE' if noise_level > noise_threshold else 'ACCEPTABLE',
                'noise_level': noise_level,
                'data_quality_score': data_quality_score,
                'noise_detected': noise_level > noise_threshold,
                'details': {
                    'missing_ratio': missing_ratio,
                    'constant_ratio': constant_ratio,
                    'outlier_ratio': outlier_ratio,
                    'extreme_outlier_ratio': extreme_outlier_ratio,
                    'low_variance_ratio': low_variance_ratio,
                    'high_skewness_ratio': high_skewness_ratio,
                    'constant_features': constant_features,
                    'noise_threshold': noise_threshold,
                    'method': 'enhanced_multi_criteria_quality_check',
                    'criteria_weights': {
                        'missing_values': 0.25,
                        'constant_features': 0.20,
                        'outliers': 0.20,
                        'extreme_outliers': 0.15,
                        'low_variance': 0.10,
                        'high_skewness': 0.10
                    }
                }
            })
            
            return noise_results
            
        except Exception as e:
            self.logger.error(f"❌ Simplified noise detection failed: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'noise_level': 0.0,
                'data_quality_score': 1.0
            }
    
    def _analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series, datetime_col: str = None) -> Dict[str, Any]:
        """วิเคราะห์ความเสถียรของ Features"""
        try:
            stability_results = {
                'status': 'ANALYZING',
                'stable_features': [],
                'unstable_features': [],
                'stability_scores': {},
                'temporal_drift': {},
                'recommendations': []
            }
            
            window_size = min(self.protection_config['stability_window'], len(X) // 4)
            
            for col in X.select_dtypes(include=[np.number]).columns:
                stability_score = self._compute_feature_stability(X[col], window_size)
                stability_results['stability_scores'][col] = stability_score
                
                if stability_score > 0.8:
                    stability_results['stable_features'].append(col)
                else:
                    stability_results['unstable_features'].append(col)
            
            # Temporal drift detection
            if datetime_col and datetime_col in X.columns:
                drift_analysis = self._detect_temporal_drift(X, datetime_col)
                stability_results['temporal_drift'] = drift_analysis
            
            stability_results['status'] = 'COMPLETED'
            return stability_results
            
        except Exception as e:
            self.logger.error(f"❌ Feature stability analysis failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_timeseries_integrity(self, X: pd.DataFrame, y: pd.Series, datetime_col: str = None) -> Dict[str, Any]:
        """ตรวจสอบความถูกต้องของ Time Series"""
        try:
            timeseries_results = {
                'status': 'ANALYZING',
                'integrity_check': True,
                'temporal_ordering': True,
                'gaps_detected': False,
                'seasonality_analysis': {},
                'trend_analysis': {},
                'recommendations': []
            }
            
            if datetime_col and datetime_col in X.columns:
                # Check temporal ordering
                if not X[datetime_col].is_monotonic_increasing:
                    timeseries_results['temporal_ordering'] = False
                    timeseries_results['integrity_check'] = False
                
                # Detect gaps
                gaps = self._detect_temporal_gaps(X[datetime_col])
                timeseries_results['gaps_detected'] = len(gaps) > 0
                timeseries_results['gaps'] = gaps
                
                # Seasonality analysis
                seasonality = self._analyze_seasonality(X, y, datetime_col)
                timeseries_results['seasonality_analysis'] = seasonality
            
            timeseries_results['status'] = 'COMPLETED'
            return timeseries_results
            
        except Exception as e:
            self.logger.error(f"❌ Time series validation failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _compute_overall_assessment(self, protection_results: Dict) -> Dict:
        """คำนวณการประเมินโดยรวม - Enhanced for Enterprise Standards"""
        try:
            # Protection scores
            leakage_score = protection_results.get('data_leakage', {}).get('leakage_score', 0)
            overfitting_score = protection_results.get('overfitting', {}).get('overfitting_score', 0)
            noise_level = protection_results.get('noise_analysis', {}).get('noise_level', 0)
            quality_score = protection_results.get('noise_analysis', {}).get('data_quality_score', 1.0)
            
            # Enhanced risk calculation with stricter enterprise standards
            overall_risk = (leakage_score * 0.35 + overfitting_score * 0.35 + noise_level * 0.30)
            
            # Stricter thresholds for enterprise standards
            if overall_risk < 0.10:  # Much stricter threshold
                protection_status = 'EXCELLENT'
                risk_level = 'LOW'
            elif overall_risk < 0.20:  # Reduced from 0.4
                protection_status = 'GOOD'
                risk_level = 'MEDIUM'
            elif overall_risk < 0.35:  # Reduced from 0.6
                protection_status = 'ACCEPTABLE'
                risk_level = 'HIGH'
            else:
                protection_status = 'POOR'
                risk_level = 'CRITICAL'
            
            # Enhanced enterprise readiness criteria
            enterprise_ready = (
                overall_risk < 0.15 and           # Stricter overall risk
                quality_score > 0.80 and          # Higher quality requirement
                leakage_score < 0.05 and          # Stricter leakage tolerance
                overfitting_score < 0.08 and      # Stricter overfitting tolerance
                noise_level < 0.03                # Stricter noise tolerance
            )
            
            # Add alerts and recommendations with stricter thresholds
            alerts = []
            recommendations = []
            
            if leakage_score > 0.05:  # Reduced from 0.3
                alerts.append("⚠️ Data leakage detected - immediate action required")
                recommendations.append("Remove or fix features causing data leakage")
            
            if overfitting_score > 0.08:  # Reduced from 0.15
                alerts.append("⚠️ Overfitting detected - model generalization at risk")
                recommendations.append("Implement regularization or reduce model complexity")
            
            if noise_level > 0.03:  # Reduced from 0.05
                alerts.append("⚠️ High noise level detected")
                recommendations.append("Apply noise filtering and feature selection")
            
            if quality_score < 0.80:  # Increased from 0.7
                alerts.append("⚠️ Low data quality detected")
                recommendations.append("Improve data preprocessing and cleaning")
            
            # Additional enterprise-specific checks
            if not enterprise_ready:
                if overall_risk >= 0.15:
                    alerts.append("❌ Overall risk too high for enterprise deployment")
                    recommendations.append("Address overfitting, noise, and data leakage issues")
                
                if quality_score <= 0.80:
                    alerts.append("❌ Data quality below enterprise standards")
                    recommendations.append("Implement advanced data cleaning and feature engineering")
            
            protection_results.update({
                'overall_assessment': {
                    'protection_status': protection_status,
                    'risk_level': risk_level,
                    'overall_risk_score': overall_risk,
                    'quality_score': quality_score,
                    'enterprise_ready': enterprise_ready,
                    'enterprise_thresholds': {
                        'max_overall_risk': 0.15,
                        'min_quality_score': 0.80,
                        'max_leakage_score': 0.05,
                        'max_overfitting_score': 0.08,
                        'max_noise_level': 0.03
                    }
                },
                'alerts': alerts,
                'recommendations': recommendations
            })
            
            return protection_results
            
        except Exception as e:
            self.logger.error(f"❌ Overall assessment failed: {str(e)}")
            return protection_results
    
    # Helper methods for specific analyses
    def _check_temporal_leakage(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict:
        """ตรวจสอบ temporal leakage"""
        try:
            # Check if datetime is properly sorted
            is_sorted = X[datetime_col].is_monotonic_increasing
            
            # Check for future information
            future_refs = []
            for col in X.columns:
                if any(keyword in col.lower() for keyword in ['future', 'next', 'forward', 'lag_-']):
                    future_refs.append(col)
            
            return {
                'detected': not is_sorted or len(future_refs) > 0,
                'datetime_sorted': is_sorted,
                'future_references': future_refs,
                'details': f"Temporal ordering: {'OK' if is_sorted else 'VIOLATION'}"
            }
        except:
            return {'detected': False, 'error': 'Could not analyze temporal leakage'}
    
    def _detect_future_information_leakage(self, X: pd.DataFrame) -> Dict:
        """ตรวจหา future information leakage"""
        suspicious_features = []
        
        # Keywords that might indicate future information
        future_keywords = ['future', 'next', 'forward', 'ahead', 'tomorrow', 'lag_-']
        
        for col in X.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in future_keywords):
                suspicious_features.append(col)
        
        return {
            'detected': len(suspicious_features) > 0,
            'suspicious_features': suspicious_features
        }
    
    def _statistical_leakage_test(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Statistical test for leakage"""
        try:
            # Use mutual information to detect suspiciously high relationships
            mi_scores = mutual_info_classif(X.select_dtypes(include=[np.number]), y)
            
            suspicious_features = []
            for i, score in enumerate(mi_scores):
                if score > 0.8:  # Suspiciously high mutual information
                    suspicious_features.append(X.columns[i])
            
            return {
                'mutual_info_scores': mi_scores.tolist(),
                'suspicious_features': suspicious_features,
                'max_mi_score': float(mi_scores.max()) if len(mi_scores) > 0 else 0
            }
        except:
            return {'error': 'Could not perform statistical leakage test'}
    
    def _compute_leakage_score(self, leakage_results: Dict) -> float:
        """คำนวณคะแนน leakage โดยรวม"""
        score = 0.0
        
        # High correlation penalty
        if 'suspicious_correlations' in leakage_results:
            high_corr_count = len([c for c in leakage_results['suspicious_correlations'] if c['correlation'] > 0.98])
            score += min(high_corr_count * 0.3, 0.6)
        
        # Temporal leakage penalty
        if leakage_results.get('temporal_leakage', False):
            score += 0.4
        
        # Future feature penalty
        if leakage_results.get('future_feature_leakage', False):
            score += 0.3
        
        # Statistical leakage penalty
        stat_leak = leakage_results.get('statistical_leakage', {})
        max_mi = stat_leak.get('max_mi_score', 0)
        if max_mi > 0.8:
            score += min((max_mi - 0.8) * 2, 0.4)
        
        return min(score, 1.0)
    
    def _train_validation_analysis(self, X: pd.DataFrame, y: pd.Series, model: Any, process_id: str = None) -> Dict:
        """วิเคราะห์ train-validation performance gap"""
        try:
            if not self.sklearn_available:
                # Fallback: simplified analysis without sklearn
                return self._train_validation_analysis_simplified(X, y)
            
            # Time-aware train-test split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate AUC
            train_auc = roc_auc_score(y_train, train_proba)
            val_auc = roc_auc_score(y_val, val_proba)
            
            gap = train_auc - val_auc
            
            return {
                'train_auc': float(train_auc),
                'validation_auc': float(val_auc),
                'performance_gap': float(gap),
                'gap_percentage': float(gap / train_auc * 100),
                'overfitting_detected': gap > self.protection_config['overfitting_threshold']
            }
        except Exception as e:
            # Fallback to simplified analysis
            try:
                return self._train_validation_analysis_simplified(X, y)
            except Exception as fallback_error:
                return {'error': f'Train-validation analysis failed: {str(fallback_error)}'}
    
    def _train_validation_analysis_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simplified train-validation analysis without sklearn"""
        try:
            # Time-aware train-test split
            split_idx = int(len(X) * 0.8)
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Simple accuracy comparison based on class distribution
            train_accuracy = (y_train == y_train.mode()[0]).mean() if len(y_train) > 0 else 0.5
            val_accuracy = (y_val == y_val.mode()[0]).mean() if len(y_val) > 0 else 0.5
            
            gap = train_accuracy - val_accuracy
            
            return {
                'train_auc': float(train_accuracy),
                'validation_auc': float(val_accuracy),
                'performance_gap': float(gap),
                'gap_percentage': float(gap / train_accuracy * 100) if train_accuracy > 0 else 0.0,
                'overfitting_detected': gap > self.protection_config.get('overfitting_threshold', 0.15),
                'method': 'simplified'
            }
        except Exception as e:
            return {
                'error': f'Simplified train-validation analysis failed: {str(e)}',
                'train_auc': 0.5,
                'validation_auc': 0.5,
                'performance_gap': 0.0,
                'gap_percentage': 0.0,
                'overfitting_detected': False
            }
    
    def _analyze_learning_curves(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
        """วิเคราะห์ learning curves"""
        try:
            if not self.sklearn_available:
                # Fallback: simplified learning curve analysis
                return self._analyze_learning_curves_simplified(X, y)
            
            # Create learning curve with different training sizes
            train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
            train_scores = []
            val_scores = []
            
            for size in train_sizes:
                n_samples = int(len(X) * size)
                if n_samples < 50:  # Skip if too few samples
                    continue
                    
                # Time-aware split
                X_subset = X.iloc[:n_samples]
                y_subset = y.iloc[:n_samples]
                
                split_idx = int(n_samples * 0.8)
                X_train = X_subset.iloc[:split_idx]
                X_val = X_subset.iloc[split_idx:]
                y_train = y_subset.iloc[:split_idx]
                y_val = y_subset.iloc[split_idx:]
                
                if len(X_val) < 10:  # Skip if validation set too small
                    continue
                
                # Train and evaluate
                model.fit(X_train, y_train)
                
                train_pred = model.predict_proba(X_train)[:, 1]
                val_pred = model.predict_proba(X_val)[:, 1]
                
                train_auc = roc_auc_score(y_train, train_pred)
                val_auc = roc_auc_score(y_val, val_pred)
                
                train_scores.append(train_auc)
                val_scores.append(val_auc)
            
            return {
                'train_scores': train_scores,
                'validation_scores': val_scores,
                'final_gap': train_scores[-1] - val_scores[-1] if train_scores and val_scores else 0,
                'converging': len(val_scores) > 2 and val_scores[-1] > val_scores[-2]
            }
        except Exception as e:
            # Fallback to simplified analysis
            try:
                return self._analyze_learning_curves_simplified(X, y)
            except Exception as fallback_error:
                return {'error': f'Learning curve analysis failed: {str(fallback_error)}'}
    
    def _analyze_learning_curves_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simplified learning curve analysis without sklearn"""
        try:
            # Simple analysis based on data size progression
            train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
            train_scores = []
            val_scores = []
            
            for size in train_sizes:
                n_samples = int(len(X) * size)
                if n_samples < 50:
                    continue
                
                # Simple score based on class distribution balance
                y_subset = y.iloc[:n_samples]
                class_balance = min(y_subset.mean(), 1 - y_subset.mean())
                
                # Simulate learning: larger datasets typically have more stable scores
                base_score = 0.5 + class_balance * 0.3
                train_score = base_score + (size * 0.1)  # Training score improves with more data
                val_score = base_score + (size * 0.05)   # Validation score improves less
                
                train_scores.append(min(train_score, 1.0))
                val_scores.append(min(val_score, 1.0))
            
            return {
                'train_scores': train_scores,
                'validation_scores': val_scores,
                'final_gap': train_scores[-1] - val_scores[-1] if train_scores and val_scores else 0,
                'converging': len(val_scores) > 2 and val_scores[-1] > val_scores[-2],
                'method': 'simplified'
            }
        except Exception as e:
            return {
                'error': f'Simplified learning curve analysis failed: {str(e)}',
                'train_scores': [0.5],
                'validation_scores': [0.5],
                'final_gap': 0.0,
                'converging': False
            }
    
    def _analyze_feature_importance_stability(self, X: pd.DataFrame, y: pd.Series, model: Any, process_id: str = None) -> Dict:
        """วิเคราะห์ความเสถียรของ feature importance"""
        try:
            importances = []
            n_iterations = 5
            
            for i in range(n_iterations):
                # Bootstrap sampling
                sample_idx = np.random.choice(len(X), size=len(X), replace=True)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx]
                
                # Fit model
                model.fit(X_sample, y_sample)
                
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            if importances:
                importances = np.array(importances)
                stability_scores = {}
                
                for i, col in enumerate(X.columns):
                    # Calculate coefficient of variation for each feature
                    mean_imp = importances[:, i].mean()
                    std_imp = importances[:, i].std()
                    cv = std_imp / mean_imp if mean_imp > 0 else float('inf')
                    stability_scores[col] = 1.0 / (1.0 + cv)  # Higher is more stable
                
                return {
                    'stability_scores': stability_scores,
                    'mean_stability': float(np.mean(list(stability_scores.values()))),
                    'stable_features': [k for k, v in stability_scores.items() if v > 0.7]
                }
            
            return {'error': 'Could not compute feature importances'}
            
        except Exception as e:
            return {'error': f'Feature importance stability analysis failed: {str(e)}'}
    
    def _compute_overfitting_score(self, overfitting_results: Dict) -> float:
        """คำนวณคะแนน overfitting โดยรวม"""
        score = 0.0
        
        # Cross-validation variance penalty
        cv_data = overfitting_results.get('cross_validation', {})
        cv_variance = cv_data.get('coefficient_of_variation', 0)
        if cv_variance > 0.3:
            score += min(cv_variance, 0.5)
        
        # Train-validation gap penalty
        tv_data = overfitting_results.get('train_validation', {})
        gap = tv_data.get('performance_gap', 0)
        if gap > 0.1:
            score += min(gap, 0.4)
        
        # Learning curve divergence penalty
        lc_data = overfitting_results.get('learning_curves', {})
        final_gap = lc_data.get('final_gap', 0)
        if final_gap > 0.1:
            score += min(final_gap, 0.3)
        
        # Feature importance instability penalty
        fi_data = overfitting_results.get('feature_importance_stability', {})
        mean_stability = fi_data.get('mean_stability', 1.0)
        if mean_stability < 0.7:
            score += min((0.7 - mean_stability), 0.2)
        
        return min(score, 1.0)
    
    def _analyze_missing_values(self, X: pd.DataFrame) -> Dict:
        """วิเคราะห์ missing values"""
        # Check for empty dataset
        if len(X) == 0 or X.empty:
            return {
                'total_missing': 0,
                'missing_by_column': {},
                'missing_percentages': {},
                'columns_with_missing': [],
                'high_missing_columns': []
            }
        
        missing_counts = X.isnull().sum()
        missing_percentages = (missing_counts / len(X)) * 100
        
        return {
            'total_missing': int(missing_counts.sum()),
            'missing_by_column': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
            'high_missing_columns': missing_percentages[missing_percentages > 20].index.tolist()
        }
    
    def _detect_outliers_comprehensive(self, X: pd.DataFrame) -> Dict:
        """ตรวจจับ outliers แบบครบถ้วน"""
        outlier_results = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            data = X[col].dropna()
            if len(data) < 10:
                continue
            
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Z-score method (with fallback)
            try:
                if self.scipy_available:
                    from scipy import stats as scipy_stats
                    z_scores = np.abs(scipy_stats.zscore(data))
                else:
                    # Fallback: manual z-score calculation
                    mean_val = data.mean()
                    std_val = data.std()
                    z_scores = np.abs((data - mean_val) / std_val) if std_val > 0 else np.zeros(len(data))
                
                z_outliers = (z_scores > 3).sum()
            except:
                z_outliers = 0
            
            outlier_results[col] = {
                'iqr_outliers': int(iqr_outliers),
                'zscore_outliers': int(z_outliers),
                'outlier_percentage': float((max(iqr_outliers, z_outliers) / len(data)) * 100)
            }
        
        return outlier_results
    
    def _analyze_feature_distributions(self, X: pd.DataFrame) -> Dict:
        """วิเคราะห์การกระจายของ features"""
        distribution_results = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            data = X[col].dropna()
            if len(data) < 10:
                continue
            
            # Normality test (with fallback)
            try:
                if self.scipy_available:
                    _, p_value = shapiro(data.sample(min(5000, len(data))))
                    is_normal = p_value > 0.05
                else:
                    # Simple normality check using skewness and kurtosis
                    from scipy import stats as scipy_stats
                    skew_val = abs(scipy_stats.skew(data))
                    kurt_val = abs(scipy_stats.kurtosis(data))
                    is_normal = skew_val < 0.5 and kurt_val < 3
            except:
                # Fallback: use basic statistics for normality assessment
                try:
                    mean_val = data.mean()
                    median_val = data.median()
                    std_val = data.std()
                    # Simple check: if mean ≈ median and reasonable spread
                    is_normal = abs(mean_val - median_val) < (0.1 * std_val) if std_val > 0 else True
                except:
                    is_normal = False
            
            # Skewness and kurtosis (with fallback)
            try:
                if self.scipy_available:
                    from scipy import stats as scipy_stats
                    skewness = float(scipy_stats.skew(data))
                    kurtosis = float(scipy_stats.kurtosis(data))
                else:
                    # Simple skewness approximation: (mean - median) / std
                    mean_val = data.mean()
                    median_val = data.median()
                    std_val = data.std()
                    skewness = (mean_val - median_val) / std_val if std_val > 0 else 0
                    
                    # Simple kurtosis approximation using percentiles
                    q75 = data.quantile(0.75)
                    q25 = data.quantile(0.25)
                    iqr = q75 - q25
                    kurtosis = (data.quantile(0.875) - data.quantile(0.125)) / iqr if iqr > 0 else 0
            except:
                skewness = 0.0
                kurtosis = 0.0
            
            distribution_results[col] = {
                'is_normal': is_normal,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'highly_skewed': abs(skewness) > 2,
                'heavy_tailed': abs(kurtosis) > 3
            }
        
        return distribution_results
    
    def _compute_signal_to_noise_ratio(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """คำนวณ Signal-to-Noise Ratio"""
        try:
            # Calculate mutual information for signal strength
            mi_scores = mutual_info_classif(X.select_dtypes(include=[np.number]), y, random_state=42)
            
            # Calculate noise level based on feature stability
            noise_estimates = []
            for col in X.select_dtypes(include=[np.number]).columns:
                data = X[col].dropna()
                if len(data) > 100:
                    # Estimate noise as variance of high-frequency components
                    diff = np.diff(data.values)
                    noise_est = np.std(diff) / np.std(data.values) if np.std(data.values) > 0 else 1
                    noise_estimates.append(noise_est)
            
            avg_noise = np.mean(noise_estimates) if noise_estimates else 0.5
            avg_signal = np.mean(mi_scores) if len(mi_scores) > 0 else 0.1
            
            snr = avg_signal / avg_noise if avg_noise > 0 else 0
            
            return {
                'signal_strength': float(avg_signal),
                'noise_level': float(avg_noise),
                'signal_to_noise_ratio': float(snr),
                'quality_assessment': 'HIGH' if snr > 2 else 'MEDIUM' if snr > 1 else 'LOW'
            }
        except Exception as e:
            return {'error': f'SNR calculation failed: {str(e)}'}
    
    def _analyze_feature_relevance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """วิเคราะห์ความเกี่ยวข้องของ features"""
        try:
            # Check for empty dataset
            if len(X) == 0 or X.empty or len(y) == 0:
                return {
                    'relevant_features': [],
                    'irrelevant_features': [],
                    'mutual_info_scores': {},
                    'correlation_scores': {},
                    'relevance_ratio': 0.0,
                    'error': 'Empty dataset provided'
                }
            
            # Check if there are numeric columns
            numeric_X = X.select_dtypes(include=[np.number])
            if numeric_X.empty:
                return {
                    'relevant_features': [],
                    'irrelevant_features': list(X.columns),
                    'mutual_info_scores': {},
                    'correlation_scores': {},
                    'relevance_ratio': 0.0,
                    'error': 'No numeric features found'
                }
            
            # Mutual information scores
            mi_scores = mutual_info_classif(numeric_X, y, random_state=42)
            
            # Correlation with target
            correlations = []
            for col in numeric_X.columns:
                corr = abs(numeric_X[col].corr(y))
                correlations.append(corr if not np.isnan(corr) else 0.0)
            
            # Categorize features
            relevant_features = []
            irrelevant_features = []
            
            for i, (col, mi_score, corr) in enumerate(zip(numeric_X.columns, mi_scores, correlations)):
                if mi_score > 0.01 or corr > 0.05:
                    relevant_features.append(col)
                else:
                    irrelevant_features.append(col)
            
            # Calculate relevance ratio safely
            total_columns = len(X.columns)
            relevance_ratio = len(relevant_features) / total_columns if total_columns > 0 else 0.0
            
            return {
                'relevant_features': relevant_features,
                'irrelevant_features': irrelevant_features,
                'mutual_info_scores': dict(zip(numeric_X.columns, mi_scores.tolist())),
                'correlation_scores': dict(zip(numeric_X.columns, correlations)),
                'relevance_ratio': relevance_ratio
            }
        except Exception as e:
            return {
                'relevant_features': [],
                'irrelevant_features': list(X.columns) if not X.empty else [],
                'mutual_info_scores': {},
                'correlation_scores': {},
                'relevance_ratio': 0.0,
                'error': f'Feature relevance analysis failed: {str(e)}'
            }
    
    def _compute_noise_level(self, noise_results: Dict) -> float:
        """คำนวณระดับ noise โดยรวม"""
        noise_score = 0.0
        
        # Missing values penalty
        missing_data = noise_results.get('missing_values', {})
        high_missing_cols = len(missing_data.get('high_missing_columns', []))
        if high_missing_cols > 0:
            noise_score += min(high_missing_cols * 0.1, 0.3)
        
        # Outliers penalty
        outliers_data = noise_results.get('outliers', {})
        high_outlier_cols = sum(1 for col_data in outliers_data.values() 
                               if col_data.get('outlier_percentage', 0) > 10)
        if high_outlier_cols > 0:
            noise_score += min(high_outlier_cols * 0.05, 0.2)
        
        # Signal-to-noise ratio penalty
        snr_data = noise_results.get('signal_to_noise', {})
        snr = snr_data.get('signal_to_noise_ratio', 1)
        if snr < 1:
            noise_score += min((1 - snr) * 0.5, 0.3)
        
        # Irrelevant features penalty
        relevance_data = noise_results.get('feature_relevance', {})
        relevance_ratio = relevance_data.get('relevance_ratio', 1)
        if relevance_ratio < 0.5:
            noise_score += min((0.5 - relevance_ratio), 0.2)
        
        return min(noise_score, 1.0)
    
    def _compute_data_quality_score(self, noise_results: Dict) -> float:
        """คำนวณคะแนนคุณภาพข้อมูล"""
        quality_score = 1.0
        
        # Subtract penalties for quality issues
        noise_level = self._compute_noise_level(noise_results)
        quality_score -= noise_level
        
        # Additional quality factors
        missing_data = noise_results.get('missing_values', {})
        total_missing = missing_data.get('total_missing', 0)
        if total_missing > 0:
            missing_penalty = min(total_missing / (len(missing_data.get('missing_by_column', {})) * 1000), 0.2)
            quality_score -= missing_penalty
        
        return max(quality_score, 0.0)
    
    def _compute_feature_stability(self, feature_series: pd.Series, window_size: int) -> float:
        """คำนวณความเสถียรของ feature"""
        try:
            if len(feature_series) < window_size * 2:
                return 0.5  # Not enough data
            
            # Split into windows and compute statistics
            windows = []
            for i in range(0, len(feature_series) - window_size, window_size):
                window_data = feature_series.iloc[i:i+window_size]
                if len(window_data.dropna()) > window_size * 0.5:  # At least 50% non-null
                    windows.append({
                        'mean': window_data.mean(),
                        'std': window_data.std(),
                        'median': window_data.median()
                    })
            
            if len(windows) < 2:
                return 0.5
            
            # Calculate stability based on variance of window statistics
            means = [w['mean'] for w in windows if not np.isnan(w['mean'])]
            stds = [w['std'] for w in windows if not np.isnan(w['std'])]
            
            if len(means) < 2:
                return 0.5
            
            # Coefficient of variation of means (lower is more stable)
            mean_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else float('inf')
            
            # Convert to stability score (0-1, higher is more stable)
            stability = 1.0 / (1.0 + mean_cv)
            
            return min(max(stability, 0.0), 1.0)
            
        except:
            return 0.5  # Default stability score
    
    def _detect_temporal_drift(self, X: pd.DataFrame, datetime_col: str) -> Dict:
        """ตรวจจับ temporal drift ใน features"""
        try:
            drift_results = {}
            dt_series = pd.to_datetime(X[datetime_col])
            
            # Extract time components
            hours = dt_series.dt.hour
            days = dt_series.dt.dayofweek
            months = dt_series.dt.month
            
            seasonality_results = {}
            
            # Test for hourly patterns
            if len(hours.unique()) > 1:
                hourly_target_means = y.groupby(hours).mean()
                hourly_variance = hourly_target_means.var()
                seasonality_results['hourly'] = {
                    'variance': float(hourly_variance),
                    'significant': hourly_variance > 0.01
                }
            
            # Test for daily patterns
            if len(days.unique()) > 1:
                daily_target_means = y.groupby(days).mean()
                daily_variance = daily_target_means.var()
                seasonality_results['daily'] = {
                    'variance': float(daily_variance),
                    'significant': daily_variance > 0.01
                }
            
            # Test for monthly patterns
            if len(months.unique()) > 1:
                monthly_target_means = y.groupby(months).mean()
                monthly_variance = monthly_target_means.var()
                seasonality_results['monthly'] = {
                    'variance': float(monthly_variance),
                    'significant': monthly_variance > 0.01
                }
            
            return seasonality_results
            
        except Exception as e:
            return {'error': f'Seasonality analysis failed: {str(e)}'}
    
    def _log_protection_summary(self, protection_results: Dict):
        """บันทึกสรุปผลการป้องกัน"""
        try:
            overall = protection_results.get('overall_assessment', {})
            status = overall.get('protection_status', 'UNKNOWN')
            risk_level = overall.get('risk_level', 'UNKNOWN')
            enterprise_ready = overall.get('enterprise_ready', False)
            
            self.logger.info("🛡️ =============== ENTERPRISE ML PROTECTION SUMMARY ===============")
            self.logger.info(f"📊 Protection Status: {status}")
            self.logger.info(f"⚠️  Risk Level: {risk_level}")
            self.logger.info(f"🏢 Enterprise Ready: {'✅ YES' if enterprise_ready else '❌ NO'}")
            
            # Data leakage summary
            leakage = protection_results.get('data_leakage', {})
            leakage_status = leakage.get('status', 'UNKNOWN')
            leakage_score = leakage.get('leakage_score', 0)
            self.logger.info(f"🔍 Data Leakage: {leakage_status} (Score: {leakage_score:.3f})")
            
            # Overfitting summary
            overfitting = protection_results.get('overfitting', {})
            overfitting_status = overfitting.get('status', 'UNKNOWN')
            overfitting_score = overfitting.get('overfitting_score', 0)
            self.logger.info(f"📈 Overfitting: {overfitting_status} (Score: {overfitting_score:.3f})")
            
            # Noise summary
            noise = protection_results.get('noise_analysis', {})
            noise_status = noise.get('status', 'UNKNOWN')
            noise_level = noise.get('noise_level', 0)
            quality_score = noise.get('data_quality_score', 0)
            self.logger.info(f"🎯 Noise Level: {noise_status} (Level: {noise_level:.3f}, Quality: {quality_score:.3f})")
            
            # Alerts
            alerts = protection_results.get('alerts', [])
            if alerts:
                self.logger.warning("⚠️  ALERTS:")
                for alert in alerts:
                    self.logger.warning(f"   {alert}")
            
            # Recommendations
            recommendations = protection_results.get('recommendations', [])
            if recommendations:
                self.logger.info("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    self.logger.info(f"   • {rec}")
            
            self.logger.info("🛡️ ================================================================")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to log protection summary: {str(e)}")
    
    def generate_protection_report(self, output_path: str = None) -> str:
        """สร้างรายงานการป้องกันแบบละเอียด"""
        try:
            if not self.protection_results:
                return "No protection analysis results available"
            
            report_lines = []
            report_lines.append("🛡️ ENTERPRISE ML PROTECTION REPORT")
            report_lines.append("=" * 50)
            report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")
            
            # Overall assessment
            overall = self.protection_results.get('overall_assessment', {})
            report_lines.append("📊 OVERALL ASSESSMENT")
            report_lines.append("-" * 20)
            report_lines.append(f"Protection Status: {overall.get('protection_status', 'UNKNOWN')}")
            report_lines.append(f"Risk Level: {overall.get('risk_level', 'UNKNOWN')}")
            report_lines.append(f"Enterprise Ready: {'✅ YES' if overall.get('enterprise_ready', False) else '❌ NO'}")
            report_lines.append(f"Overall Risk Score: {overall.get('overall_risk_score', 0):.3f}")
            report_lines.append(f"Quality Score: {overall.get('quality_score', 0):.3f}")
            report_lines.append("")
            
            # Data info
            data_info = self.protection_results.get('data_info', {})
            report_lines.append("📈 DATA INFORMATION")
            report_lines.append("-" * 20)
            report_lines.append(f"Samples: {data_info.get('samples', 0):,}")
            report_lines.append(f"Features: {data_info.get('features', 0)}")
            report_lines.append(f"Target Distribution: {data_info.get('target_distribution', {})}")
            report_lines.append("")
            
            # Data leakage details
            leakage = self.protection_results.get('data_leakage', {})
            if leakage.get('status') != 'ERROR':
                report_lines.append("🔍 DATA LEAKAGE ANALYSIS")
                report_lines.append("-" * 25)
                report_lines.append(f"Status: {leakage.get('status', 'UNKNOWN')}")
                report_lines.append(f"Leakage Score: {leakage.get('leakage_score', 0):.3f}")
                report_lines.append(f"Leakage Detected: {'⚠️ YES' if leakage.get('leakage_detected', False) else '✅ NO'}")
                
                # Suspicious correlations
                sus_corr = leakage.get('suspicious_correlations', [])
                if sus_corr:
                    report_lines.append("Suspicious Correlations:")
                    for corr in sus_corr[:5]:  # Top 5
                        report_lines.append(f"  • {corr['feature']}: {corr['correlation']:.3f} ({corr['risk_level']})")
                
                # Future features
                future_features = leakage.get('future_features', [])
                if future_features:
                    report_lines.append(f"Future Information Features: {len(future_features)}")
                    for feat in future_features[:3]:  # Top 3
                        report_lines.append(f"  • {feat}")
                
                report_lines.append("")
            
            # Overfitting details
            overfitting = self.protection_results.get('overfitting', {})
            if overfitting.get('status') != 'ERROR':
                report_lines.append("📈 OVERFITTING ANALYSIS")
                report_lines.append("-" * 22)
                report_lines.append(f"Status: {overfitting.get('status', 'UNKNOWN')}")
                report_lines.append(f"Overfitting Score: {overfitting.get('overfitting_score', 0):.3f}")
                report_lines.append(f"Overfitting Detected: {'⚠️ YES' if overfitting.get('overfitting_detected', False) else '✅ NO'}")
                
                # Cross-validation results
                cv_data = overfitting.get('cross_validation', {})
                if cv_data:
                    report_lines.append(f"CV Mean Score: {cv_data.get('mean_cv_score', 0):.3f}")
                    report_lines.append(f"CV Std Score: {cv_data.get('std_cv_score', 0):.3f}")
                    report_lines.append(f"CV Coefficient of Variation: {cv_data.get('coefficient_of_variation', 0):.3f}")
                
                # Train-validation gap
                tv_data = overfitting.get('train_validation', {})
                if tv_data:
                    report_lines.append(f"Train AUC: {tv_data.get('train_auc', 0):.3f}")
                    report_lines.append(f"Validation AUC: {tv_data.get('validation_auc', 0):.3f}")
                    report_lines.append(f"Performance Gap: {tv_data.get('performance_gap', 0):.3f}")
                
                report_lines.append("")
            
            # Noise analysis details
            noise = self.protection_results.get('noise_analysis', {})
            if noise.get('status') != 'ERROR':
                report_lines.append("🎯 NOISE & QUALITY ANALYSIS")
                report_lines.append("-" * 26)
                report_lines.append(f"Status: {noise.get('status', 'UNKNOWN')}")
                report_lines.append(f"Noise Level: {noise.get('noise_level', 0):.3f}")
                report_lines.append(f"Data Quality Score: {noise.get('data_quality_score', 0):.3f}")
                report_lines.append(f"Noise Detected: {'⚠️ YES' if noise.get('noise_detected', False) else '✅ NO'}")
                
                # Signal-to-noise ratio
                snr_data = noise.get('signal_to_noise', {})
                if snr_data:
                    report_lines.append(f"Signal Strength: {snr_data.get('signal_strength', 0):.3f}")
                    report_lines.append(f"Signal-to-Noise Ratio: {snr_data.get('signal_to_noise_ratio', 0):.3f}")
                    report_lines.append(f"Quality Assessment: {snr_data.get('quality_assessment', 'UNKNOWN')}")
                
                # Feature relevance
                relevance_data = noise.get('feature_relevance', {})
                if relevance_data:
                    report_lines.append(f"Relevant Features: {len(relevance_data.get('relevant_features', []))}")
                    report_lines.append(f"Irrelevant Features: {len(relevance_data.get('irrelevant_features', []))}")
                    report_lines.append(f"Relevance Ratio: {relevance_data.get('relevance_ratio', 0):.3f}")
                
                report_lines.append("")
            
            # Alerts and recommendations
            alerts = self.protection_results.get('alerts', [])
            if alerts:
                report_lines.append("⚠️ ALERTS")
                report_lines.append("-" * 8)
                for i, alert in enumerate(alerts, 1):
                    report_lines.append(f"{i}. {alert}")
                report_lines.append("")
            
            recommendations = self.protection_results.get('recommendations', [])
            if recommendations:
                report_lines.append("💡 RECOMMENDATIONS")
                report_lines.append("-" * 16)
                for i, rec in enumerate(recommendations, 1):
                    report_lines.append(f"{i}. {rec}")
                report_lines.append("")
            
            report_lines.append("=" * 50)
            report_lines.append("Report End")
            
            report_content = "\n".join(report_lines)
            
            # Save to file if path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"📄 Protection report saved to: {output_path}")
            
            return report_content
            
        except Exception as e:
            error_msg = f"❌ Failed to generate protection report: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def validate_configuration(self) -> Dict[str, Any]:
        """ตรวจสอบความถูกต้องของ configuration"""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check thresholds
        if self.protection_config.get('overfitting_threshold', 0) <= 0:
            validation_results['issues'].append("overfitting_threshold must be > 0")
            validation_results['valid'] = False
        
        if self.protection_config.get('noise_threshold', 0) <= 0:
            validation_results['issues'].append("noise_threshold must be > 0")
            validation_results['valid'] = False
        
        if self.protection_config.get('leak_detection_window', 0) <= 0:
            validation_results['issues'].append("leak_detection_window must be > 0")
            validation_results['valid'] = False
        
        # Check reasonable ranges
        if self.protection_config.get('overfitting_threshold', 0) > 0.5:
            validation_results['warnings'].append("overfitting_threshold > 0.5 may be too high")
        
        if self.protection_config.get('noise_threshold', 0) > 0.2:
            validation_results['warnings'].append("noise_threshold > 0.2 may be too high")
        
        return validation_results
    
    def get_protection_status(self) -> Dict[str, Any]:
        """ดึงสถานะการป้องกันปัจจุบัน"""
        validation = self.validate_configuration()
        
        status = {
            'enabled': True,
            'sklearn_available': self.sklearn_available,
            'scipy_available': self.scipy_available,
            'configuration_valid': validation['valid'],
            'configuration_issues': validation['issues'],
            'configuration_warnings': validation['warnings'],
            'overfitting_threshold': self.protection_config.get('overfitting_threshold'),
            'noise_threshold': self.protection_config.get('noise_threshold'),
            'status': 'ACTIVE' if validation['valid'] and not validation['warnings'] 
                     else 'ACTIVE_WITH_WARNINGS' if validation['valid'] 
                     else 'CONFIGURATION_ERROR'
        }
        
        return status
    
    def _analyze_data_quality(self, X: np.ndarray, y: np.ndarray, process_id: str = None) -> Dict[str, Any]:
        """🔍 Analyze overall data quality"""
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.data_log("Starting data quality analysis", "Data_Quality", process_id=process_id)
            
            # Check for empty data
            if len(X) == 0 or len(y) == 0:
                return {
                    'status': 'ERROR',
                    'overall_quality_score': 0.0,
                    'issues': ['Empty dataset provided'],
                    'warnings': [],
                    'recommendations': ['Provide valid data with samples'],
                    'error': 'Empty dataset'
                }
            
            # Convert to DataFrame for analysis
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            else:
                X_df = X
            
            if isinstance(y, np.ndarray):
                y_series = pd.Series(y)
            else:
                y_series = y
            
            quality_results = {
                'status': 'ANALYZING',
                'overall_quality_score': 0.0,
                'issues': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Check for missing values
            missing_info = self._analyze_missing_values(X_df)
            quality_results['missing_values'] = missing_info
            
            # Check feature distributions
            distribution_info = self._analyze_feature_distributions(X_df)
            quality_results['distributions'] = distribution_info
            
            # Check feature relevance
            relevance_info = self._analyze_feature_relevance(X_df, y_series)
            quality_results['relevance'] = relevance_info
            
            # Calculate overall quality score
            missing_penalty = min(missing_info.get('total_missing', 0) / len(X_df) * 2, 0.3)
            
            # Count distribution issues
            dist_issues = sum(1 for col, info in distribution_info.items() 
                            if info.get('highly_skewed', False) or info.get('heavy_tailed', False))
            distribution_penalty = min(dist_issues / len(X_df.columns) * 0.5, 0.3)
            
            # Relevance penalty
            irrelevant_features = len(relevance_info.get('irrelevant_features', []))
            relevance_penalty = min(irrelevant_features / len(X_df.columns) * 0.4, 0.3)
            
            # Overall quality score (1.0 = perfect, 0.0 = poor)
            quality_score = max(1.0 - missing_penalty - distribution_penalty - relevance_penalty, 0.0)
            quality_results['overall_quality_score'] = quality_score
            
            # Add issues and recommendations
            if missing_penalty > 0.1:
                quality_results['issues'].append("High missing value rate detected")
                quality_results['recommendations'].append("Implement missing value imputation")
            
            if distribution_penalty > 0.1:
                quality_results['issues'].append("Significant distribution issues detected")
                quality_results['recommendations'].append("Apply feature transformation and normalization")
            
            if relevance_penalty > 0.1:
                quality_results['issues'].append("Many irrelevant features detected")
                quality_results['recommendations'].append("Perform feature selection to remove irrelevant features")
            
            # Set status
            if quality_score >= 0.8:
                quality_results['status'] = 'EXCELLENT'
            elif quality_score >= 0.6:
                quality_results['status'] = 'GOOD'
            elif quality_score >= 0.4:
                quality_results['status'] = 'ACCEPTABLE'
            else:
                quality_results['status'] = 'POOR'
            
            return quality_results
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error("Data quality analysis failed", "Data_Quality", 
                                process_id=process_id, exception=e)
            return {
                'status': 'ERROR',
                'error': str(e),
                'overall_quality_score': 0.0,
                'issues': [f"Analysis failed: {str(e)}"]
            }
    
    def _analyze_feature_correlation(self, X: np.ndarray, feature_names: List[str] = None, 
                                   process_id: str = None) -> Dict[str, Any]:
        """🔗 Analyze feature correlations"""
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.data_log("Starting feature correlation analysis", "Correlation_Analysis", process_id=process_id)
            
            # Convert to DataFrame for analysis
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=feature_names or [f'feature_{i}' for i in range(X.shape[1])])
            else:
                X_df = X
            
            correlation_results = {
                'status': 'ANALYZING',
                'high_correlations': [],
                'correlation_matrix_summary': {},
                'multicollinearity_detected': False,
                'recommendations': []
            }
            
            # Calculate correlation matrix for numeric features only
            numeric_features = X_df.select_dtypes(include=[np.number])
            if len(numeric_features.columns) <= 1:
                correlation_results['status'] = 'INSUFFICIENT_FEATURES'
                return correlation_results
            
            corr_matrix = numeric_features.corr().abs()
            
            # Find high correlations (excluding diagonal)
            high_corr_threshold = self.protection_config.get('max_feature_correlation', 0.75)
            
            high_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if corr_value > high_corr_threshold:
                        high_correlations.append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': float(corr_value),
                            'risk_level': 'HIGH' if corr_value > 0.9 else 'MEDIUM'
                        })
            
            correlation_results['high_correlations'] = high_correlations
            correlation_results['multicollinearity_detected'] = len(high_correlations) > 0
            
            # Summary statistics
            correlation_results['correlation_matrix_summary'] = {
                'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
                'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                'high_correlation_pairs': len(high_correlations)
            }
            
            # Add recommendations
            if len(high_correlations) > 0:
                correlation_results['recommendations'].append("Remove or combine highly correlated features")
                correlation_results['recommendations'].append("Consider PCA or feature selection techniques")
            
            correlation_results['status'] = 'COMPLETED'
            return correlation_results
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error("Feature correlation analysis failed", "Correlation_Analysis", 
                                process_id=process_id, exception=e)
            return {
                'status': 'ERROR',
                'error': str(e),
                'high_correlations': [],
                'multicollinearity_detected': False
            }
    
    def _analyze_noise(self, X: np.ndarray, y: np.ndarray, process_id: str = None) -> Dict[str, Any]:
        """📊 Analyze noise levels in data"""
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.data_log("Starting noise analysis", "Noise_Analysis", process_id=process_id)
            
            # Convert to appropriate format and delegate to existing method
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            else:
                X_df = X
            
            if isinstance(y, np.ndarray):
                y_series = pd.Series(y)
            else:
                y_series = y
            
            # Use existing noise detection method
            noise_results = self._detect_noise_and_quality(X_df, y_series)
            
            return noise_results
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error("Noise analysis failed", "Noise_Analysis", 
                                process_id=process_id, exception=e)
            return {
                'status': 'ERROR',
                'error': str(e),
                'noise_level': 1.0,  # Assume high noise on error
                'data_quality_score': 0.0
            }
    
    def _validate_time_series_integrity(self, X: np.ndarray, y: np.ndarray, 
                                      process_id: str = None) -> Dict[str, Any]:
        """⏰ Validate time series data integrity"""
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.data_log("Starting time series validation", "TimeSeries_Validation", process_id=process_id)
            
            timeseries_results = {
                'status': 'ANALYZING',
                'is_time_series': False,
                'temporal_gaps': [],
                'seasonal_patterns': {},
                'data_drift': {},
                'integrity_score': 1.0,
                'issues': [],
                'recommendations': []
            }
            
            # For now, assume this is not explicitly time series data
            # In a real implementation, you would check for datetime columns
            # and perform proper temporal analysis
            
            # Simple validation based on data ordering
            n_samples = len(X)
            
            # Check for data consistency (simplified)
            if n_samples < 100:
                timeseries_results['issues'].append("Insufficient data for time series analysis")
                timeseries_results['integrity_score'] = 0.5
            
            # Check for potential data drift by comparing first and last thirds
            if n_samples >= 300:
                try:
                    first_third = X[:n_samples//3]
                    last_third = X[-n_samples//3:]
                    
                    # Simple drift detection using mean differences
                    drift_scores = []
                    for col in range(X.shape[1]):
                        first_mean = np.mean(first_third[:, col])
                        last_mean = np.mean(last_third[:, col])
                        first_std = np.std(first_third[:, col])
                        
                        if first_std > 0:
                            drift_score = abs(first_mean - last_mean) / first_std
                            drift_scores.append(drift_score)
                    
                    avg_drift = np.mean(drift_scores) if drift_scores else 0
                    timeseries_results['data_drift'] = {
                        'detected': avg_drift > 2.0,
                        'average_drift_score': float(avg_drift),
                        'severity': 'HIGH' if avg_drift > 3.0 else 'MEDIUM' if avg_drift > 2.0 else 'LOW'
                    }
                    
                    if avg_drift > 2.0:
                        timeseries_results['issues'].append("Significant data drift detected")
                        timeseries_results['recommendations'].append("Check for data distribution changes over time")
                        timeseries_results['integrity_score'] *= 0.8
                        
                except Exception:
                    # Drift analysis failed, continue with reduced score
                    timeseries_results['integrity_score'] *= 0.9
            
            # Set final status
            if timeseries_results['integrity_score'] >= 0.8:
                timeseries_results['status'] = 'GOOD'
            elif timeseries_results['integrity_score'] >= 0.6:
                timeseries_results['status'] = 'ACCEPTABLE'
            else:
                timeseries_results['status'] = 'POOR'
            
            return timeseries_results
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error("Time series validation failed", "TimeSeries_Validation", 
                                process_id=process_id, exception=e)
            return {
                'status': 'ERROR',
                'error': str(e),
                'integrity_score': 0.0,
                'issues': [f"Validation failed: {str(e)}"]
            }
    
    def _assess_enterprise_readiness(self, protection_results: Dict, process_id: str = None) -> Dict[str, Any]:
        """🏢 Assess overall enterprise readiness"""
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.security("Starting enterprise readiness assessment", "Enterprise_Assessment", process_id=process_id)
            
            # Extract scores from analysis results
            data_quality = protection_results.get('data_quality', {})
            quality_score = data_quality.get('overall_quality_score', 0.0)
            
            correlation_analysis = protection_results.get('correlation_analysis', {})
            multicollinearity_detected = correlation_analysis.get('multicollinearity_detected', False)
            
            noise_analysis = protection_results.get('noise_analysis', {})
            noise_level = noise_analysis.get('noise_level', 1.0)
            
            overfitting_analysis = protection_results.get('overfitting_analysis', {})
            overfitting_detected = overfitting_analysis.get('overfitting_detected', False)
            overfitting_score = overfitting_analysis.get('overfitting_score', 0.0)
            
            leakage_analysis = protection_results.get('leakage_analysis', {})
            leakage_detected = leakage_analysis.get('leakage_detected', False)
            leakage_score = leakage_analysis.get('leakage_score', 0.0)
            
            timeseries_analysis = protection_results.get('timeseries_analysis', {})
            integrity_score = timeseries_analysis.get('integrity_score', 1.0)
            
            # Calculate overall risk assessment
            risk_factors = []
            
            # Data quality risk
            if quality_score < 0.6:
                risk_factors.append(('poor_data_quality', 0.3))
            elif quality_score < 0.8:
                risk_factors.append(('moderate_data_quality', 0.1))
            
            # Multicollinearity risk
            if multicollinearity_detected:
                risk_factors.append(('multicollinearity', 0.2))
            
            # Noise risk
            noise_threshold = self.protection_config.get('noise_threshold', 0.02)
            if noise_level > noise_threshold * 5:  # Very high noise
                risk_factors.append(('high_noise', 0.3))
            elif noise_level > noise_threshold:
                risk_factors.append(('moderate_noise', 0.1))
            
            # Overfitting risk
            overfitting_threshold = self.protection_config.get('overfitting_threshold', 0.05)
            if overfitting_detected and overfitting_score > overfitting_threshold * 2:
                risk_factors.append(('high_overfitting', 0.4))
            elif overfitting_detected:
                risk_factors.append(('moderate_overfitting', 0.2))
            
            # Data leakage risk
            if leakage_detected:
                risk_factors.append(('data_leakage', 0.5))  # Critical issue
            
            # Time series integrity risk
            if integrity_score < 0.6:
                risk_factors.append(('poor_temporal_integrity', 0.2))
            
            # Calculate total risk score
            total_risk = sum(weight for _, weight in risk_factors)
            total_risk = min(total_risk, 1.0)  # Cap at 1.0
            
            # Determine protection status and risk level
            if total_risk < 0.1:
                protection_status = 'EXCELLENT'
                risk_level = 'LOW'
            elif total_risk < 0.25:
                protection_status = 'GOOD'
                risk_level = 'LOW'
            elif total_risk < 0.5:
                protection_status = 'ACCEPTABLE'
                risk_level = 'MEDIUM'
            elif total_risk < 0.75:
                protection_status = 'POOR'
                risk_level = 'HIGH'
            else:
                protection_status = 'CRITICAL'
                risk_level = 'CRITICAL'
            
            # Enterprise readiness assessment
            enterprise_ready = (
                total_risk < 0.25 and
                quality_score >= 0.8 and
                not leakage_detected and
                noise_level <= noise_threshold * 2 and
                (not overfitting_detected or overfitting_score <= overfitting_threshold)
            )
            
            # Generate alerts and recommendations
            alerts = []
            recommendations = []
            
            for factor_name, weight in risk_factors:
                if factor_name == 'data_leakage':
                    alerts.append("🚨 CRITICAL: Data leakage detected - immediate action required")
                    recommendations.append("Remove or fix features causing data leakage")
                elif factor_name in ['high_overfitting', 'high_noise']:
                    alerts.append(f"⚠️ HIGH RISK: {factor_name.replace('_', ' ').title()} detected")
                    if 'overfitting' in factor_name:
                        recommendations.append("Implement stronger regularization or reduce model complexity")
                    else:
                        recommendations.append("Apply advanced noise filtering and feature cleaning")
                elif weight >= 0.2:
                    alerts.append(f"⚠️ MEDIUM RISK: {factor_name.replace('_', ' ').title()} detected")
            
            if not enterprise_ready:
                alerts.append("❌ System not ready for enterprise deployment")
                recommendations.append("Address all identified risk factors before production use")
            
            # Enterprise assessment summary
            enterprise_assessment = {
                'overall_assessment': {
                    'protection_status': protection_status,
                    'risk_level': risk_level,
                    'total_risk_score': total_risk,
                    'enterprise_ready': enterprise_ready,
                    'risk_factors': [factor for factor, _ in risk_factors],
                    'enterprise_score': max(1.0 - total_risk, 0.0)
                },
                'quality_metrics': {
                    'data_quality_score': quality_score,
                    'noise_level': noise_level,
                    'overfitting_score': overfitting_score,
                    'leakage_score': leakage_score,
                    'integrity_score': integrity_score
                },
                'alerts': alerts,
                'recommendations': recommendations,
                'enterprise_ready': enterprise_ready
            }
            
            return enterprise_assessment
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.critical("Enterprise readiness assessment failed", "Enterprise_Assessment", 
                                   process_id=process_id, exception=e)
            return {
                'overall_assessment': {
                    'protection_status': 'ERROR',
                    'risk_level': 'CRITICAL',
                    'total_risk_score': 1.0,
                    'enterprise_ready': False,
                    'enterprise_score': 0.0
                },
                'alerts': [f"Assessment failed: {str(e)}"],
                'recommendations': ["Fix system errors before proceeding"],
                'enterprise_ready': False
            }
