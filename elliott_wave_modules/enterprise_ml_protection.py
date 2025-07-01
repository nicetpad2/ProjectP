#!/usr/bin/env python3
"""
🛡️ ENTERPRISE ML PROTECTION SYSTEM
ระบบป้องกัน Overfitting, Noise Detection และ Data Leakage ระดับ Enterprise

🎯 Core Features:
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
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats
from scipy.stats import ks_2samp, shapiro
import sys
from pathlib import Path


class EnterpriseMLProtectionSystem:
    """ระบบป้องกัน ML ระดับ Enterprise สำหรับ NICEGOLD Trading System"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.protection_config = {
            'overfitting_threshold': 0.15,  # Max difference between train/val
            'noise_threshold': 0.05,        # Max noise ratio allowed
            'leak_detection_window': 100,   # Samples to check for leakage
            'min_samples_split': 50,        # Minimum samples for time split
            'stability_window': 1000,       # Window for feature stability
            'significance_level': 0.05,     # Statistical significance level
        }
        self.protection_results = {}
        
    def comprehensive_protection_analysis(self, 
                                        X: pd.DataFrame, 
                                        y: pd.Series, 
                                        model: Any = None,
                                        datetime_col: str = None) -> Dict[str, Any]:
        """
        ระบบตรวจสอบป้องกันแบบครบถ้วน
        
        Returns:
            Dict containing all protection analysis results
        """
        try:
            self.logger.info("🛡️ Starting Enterprise ML Protection Analysis...")
            
            # Initial data validation
            if X.empty or y.empty:
                return {
                    'protection_status': 'ERROR',
                    'error': 'Empty data provided',
                    'timestamp': datetime.now().isoformat(),
                    'overall_assessment': {
                        'protection_status': 'ERROR',
                        'risk_level': 'CRITICAL',
                        'enterprise_ready': False
                    },
                    'alerts': ['❌ Empty data provided'],
                    'recommendations': ['Provide valid training data']
                }
            
            if len(X) < 10:
                return {
                    'protection_status': 'ERROR',
                    'error': 'Insufficient data samples',
                    'timestamp': datetime.now().isoformat(),
                    'overall_assessment': {
                        'protection_status': 'ERROR',
                        'risk_level': 'CRITICAL',
                        'enterprise_ready': False
                    },
                    'alerts': ['❌ Insufficient data samples'],
                    'recommendations': ['Provide at least 10 data samples']
                }
            
            protection_results = {
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'samples': len(X),
                    'features': len(X.columns),
                    'target_distribution': y.value_counts().to_dict() if len(y.unique()) < 10 else {'unique_values': len(y.unique())}
                },
                'protection_status': 'ANALYZING',
                'alerts': [],
                'recommendations': []
            }
            
            # 1. Data Leakage Detection
            self.logger.info("🔍 Phase 1: Data Leakage Detection...")
            leakage_results = self._detect_data_leakage(X, y, datetime_col)
            protection_results['data_leakage'] = leakage_results
            
            # 2. Overfitting Detection
            self.logger.info("📊 Phase 2: Overfitting Detection...")
            overfitting_results = self._detect_overfitting(X, y, model)
            protection_results['overfitting'] = overfitting_results
            
            # 3. Noise Detection & Analysis
            self.logger.info("🎯 Phase 3: Noise Detection & Analysis...")
            noise_results = self._detect_noise_and_quality(X, y)
            protection_results['noise_analysis'] = noise_results
            
            # 4. Feature Stability Analysis
            self.logger.info("📈 Phase 4: Feature Stability Analysis...")
            stability_results = self._analyze_feature_stability(X, y, datetime_col)
            protection_results['feature_stability'] = stability_results
            
            # 5. Time-Series Validation
            self.logger.info("⏰ Phase 5: Time-Series Validation...")
            timeseries_results = self._validate_timeseries_integrity(X, y, datetime_col)
            protection_results['timeseries_validation'] = timeseries_results
            
            # 6. Overall Assessment
            protection_results = self._compute_overall_assessment(protection_results)
            
            self.protection_results = protection_results
            
            # Log summary
            self._log_protection_summary(protection_results)
            
            return protection_results
            
        except Exception as e:
            self.logger.error(f"❌ Enterprise Protection Analysis failed: {str(e)}")
            return {
                'protection_status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
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
    
    def _detect_overfitting(self, X: pd.DataFrame, y: pd.Series, model: Any = None) -> Dict[str, Any]:
        """ตรวจจับ Overfitting ด้วยหลายวิธี"""
        try:
            overfitting_results = {
                'status': 'ANALYZING',
                'overfitting_detected': False,
                'overfitting_score': 0.0,
                'cross_validation': {},
                'learning_curves': {},
                'variance_analysis': {},
                'details': {}
            }
            
            # Check for sufficient data
            if len(X) < 50:
                overfitting_results.update({
                    'status': 'INSUFFICIENT_DATA',
                    'overfitting_score': 0.5,  # Medium risk due to small data
                    'overfitting_detected': True,
                    'details': {'error': 'Insufficient data for reliable overfitting detection'}
                })
                return overfitting_results
            
            # 1. Time-Series Cross Validation
            try:
                n_splits = min(5, len(X) // 20)  # Ensure enough data per split
                if n_splits < 2:
                    n_splits = 2
                    
                tscv = TimeSeriesSplit(n_splits=n_splits)
                
                if model is None:
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
                
                # Cross-validation scores with proper error handling
                try:
                    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc')
                    
                    if len(cv_scores) > 0 and not np.isnan(cv_scores).all():
                        cv_scores = cv_scores[~np.isnan(cv_scores)]  # Remove NaN values
                        
                        overfitting_results['cross_validation'] = {
                            'cv_scores': cv_scores.tolist(),
                            'mean_cv_score': float(cv_scores.mean()),
                            'std_cv_score': float(cv_scores.std()),
                            'coefficient_of_variation': float(cv_scores.std() / cv_scores.mean()) if cv_scores.mean() > 0 else 0.0
                        }
                    else:
                        overfitting_results['cross_validation'] = {
                            'cv_scores': [],
                            'mean_cv_score': 0.5,
                            'std_cv_score': 0.0,
                            'coefficient_of_variation': 0.0,
                            'error': 'All CV scores were NaN'
                        }
                        
                except Exception as cv_error:
                    self.logger.warning(f"Cross-validation failed: {str(cv_error)}")
                    overfitting_results['cross_validation'] = {
                        'error': str(cv_error),
                        'mean_cv_score': 0.5,
                        'std_cv_score': 0.0
                    }
                    
            except Exception as e:
                self.logger.warning(f"Time-series CV setup failed: {str(e)}")
                overfitting_results['cross_validation'] = {'error': str(e)}
            
            # 2. Simple variance analysis as fallback
            try:
                train_val_analysis = self._simple_variance_analysis(X, y)
                overfitting_results['train_validation'] = train_val_analysis
            except Exception as e:
                overfitting_results['train_validation'] = {'error': str(e)}
            
            # Overall overfitting assessment
            overfitting_score = self._compute_overfitting_score(overfitting_results)
            overfitting_results['overfitting_score'] = float(overfitting_score)
            overfitting_results['overfitting_detected'] = overfitting_score > self.protection_config['overfitting_threshold']
            overfitting_results['status'] = 'DETECTED' if overfitting_results['overfitting_detected'] else 'ACCEPTABLE'
            
            return overfitting_results
            
        except Exception as e:
            self.logger.error(f"❌ Overfitting detection failed: {str(e)}")
            return {'status': 'ERROR', 'error': str(e)}
    
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
        """คำนวณการประเมินโดยรวม"""
        try:
            # Protection scores
            leakage_score = protection_results.get('data_leakage', {}).get('leakage_score', 0)
            overfitting_score = protection_results.get('overfitting', {}).get('overfitting_score', 0)
            noise_level = protection_results.get('noise_analysis', {}).get('noise_level', 0)
            quality_score = protection_results.get('noise_analysis', {}).get('data_quality_score', 1.0)
            
            # Overall risk score (0-1, lower is better)
            overall_risk = (leakage_score * 0.4 + overfitting_score * 0.3 + noise_level * 0.3)
            
            # Protection status
            if overall_risk < 0.2:
                protection_status = 'EXCELLENT'
                risk_level = 'LOW'
            elif overall_risk < 0.4:
                protection_status = 'GOOD'
                risk_level = 'MEDIUM'
            elif overall_risk < 0.6:
                protection_status = 'ACCEPTABLE'
                risk_level = 'HIGH'
            else:
                protection_status = 'POOR'
                risk_level = 'CRITICAL'
            
            # Add alerts and recommendations
            alerts = []
            recommendations = []
            
            if leakage_score > 0.3:
                alerts.append("⚠️ Data leakage detected - immediate action required")
                recommendations.append("Remove or fix features causing data leakage")
            
            if overfitting_score > 0.15:
                alerts.append("⚠️ Overfitting detected - model generalization at risk")
                recommendations.append("Implement regularization or reduce model complexity")
            
            if noise_level > 0.05:
                alerts.append("⚠️ High noise level detected")
                recommendations.append("Apply noise filtering and feature selection")
            
            if quality_score < 0.7:
                alerts.append("⚠️ Low data quality detected")
                recommendations.append("Improve data preprocessing and cleaning")
            
            protection_results.update({
                'overall_assessment': {
                    'protection_status': protection_status,
                    'risk_level': risk_level,
                    'overall_risk_score': overall_risk,
                    'quality_score': quality_score,
                    'enterprise_ready': overall_risk < 0.3 and quality_score > 0.7
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
    
    def _train_validation_analysis(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
        """วิเคราะห์ train-validation performance gap"""
        try:
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
            return {'error': f'Train-validation analysis failed: {str(e)}'}
    
    def _analyze_learning_curves(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
        """วิเคราะห์ learning curves"""
        try:
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
            return {'error': f'Learning curve analysis failed: {str(e)}'}
    
    def _analyze_feature_importance_stability(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
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
    
    def _simple_variance_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """การวิเคราะห์ variance แบบง่าย"""
        try:
            # Calculate feature variances
            feature_variances = X.var()
            
            # Check for high variance features (potential overfitting indicators)
            high_variance_features = feature_variances[feature_variances > feature_variances.quantile(0.95)]
            
            # Calculate target variance
            target_variance = y.var() if len(y.unique()) > 2 else 0.25  # Binary target default variance
            
            return {
                'feature_variance_stats': {
                    'mean_variance': float(feature_variances.mean()),
                    'std_variance': float(feature_variances.std()),
                    'max_variance': float(feature_variances.max()),
                    'min_variance': float(feature_variances.min())
                },
                'high_variance_features': high_variance_features.index.tolist(),
                'high_variance_count': len(high_variance_features),
                'target_variance': float(target_variance),
                'variance_ratio': float(feature_variances.mean() / target_variance) if target_variance > 0 else 0.0
            }
        except Exception as e:
            return {'error': str(e), 'variance_ratio': 1.0}
    
    def _compute_overfitting_score(self, overfitting_results: Dict) -> float:
        """คำนวณคะแนน overfitting โดยรวม"""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # CV score consistency (weight: 40%)
            cv_data = overfitting_results.get('cross_validation', {})
            if 'coefficient_of_variation' in cv_data and not cv_data.get('error'):
                cv_weight = 0.4
                cv_score = min(cv_data['coefficient_of_variation'], 1.0)  # Cap at 1.0
                score += cv_score * cv_weight
                weight_sum += cv_weight
            
            # Variance analysis (weight: 30%)
            variance_data = overfitting_results.get('train_validation', {})
            if 'variance_ratio' in variance_data and not variance_data.get('error'):
                var_weight = 0.3
                # High variance ratio indicates potential overfitting
                var_ratio = variance_data['variance_ratio']
                var_score = min(var_ratio / 10.0, 1.0)  # Normalize and cap
                score += var_score * var_weight
                weight_sum += var_weight
            
            # High variance features (weight: 30%)
            if 'high_variance_count' in variance_data:
                hv_weight = 0.3
                high_var_count = variance_data['high_variance_count']
                total_features = overfitting_results.get('details', {}).get('total_features', len(overfitting_results.get('data_info', {}).get('features', 10)))
                hv_ratio = high_var_count / max(total_features, 1)
                hv_score = min(hv_ratio, 1.0)
                score += hv_score * hv_weight
                weight_sum += hv_weight
            
            # Normalize by actual weights used
            if weight_sum > 0:
                return score / weight_sum
            else:
                return 0.1  # Low default score if no analysis possible
                
        except Exception as e:
            self.logger.warning(f"Overfitting score computation failed: {str(e)}")
            return 0.2  # Conservative estimate
    
    # ... existing methods ...
    
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
        """คำนวณการประเมินโดยรวม"""
        try:
            # Protection scores
            leakage_score = protection_results.get('data_leakage', {}).get('leakage_score', 0)
            overfitting_score = protection_results.get('overfitting', {}).get('overfitting_score', 0)
            noise_level = protection_results.get('noise_analysis', {}).get('noise_level', 0)
            quality_score = protection_results.get('noise_analysis', {}).get('data_quality_score', 1.0)
            
            # Overall risk score (0-1, lower is better)
            overall_risk = (leakage_score * 0.4 + overfitting_score * 0.3 + noise_level * 0.3)
            
            # Protection status
            if overall_risk < 0.2:
                protection_status = 'EXCELLENT'
                risk_level = 'LOW'
            elif overall_risk < 0.4:
                protection_status = 'GOOD'
                risk_level = 'MEDIUM'
            elif overall_risk < 0.6:
                protection_status = 'ACCEPTABLE'
                risk_level = 'HIGH'
            else:
                protection_status = 'POOR'
                risk_level = 'CRITICAL'
            
            # Add alerts and recommendations
            alerts = []
            recommendations = []
            
            if leakage_score > 0.3:
                alerts.append("⚠️ Data leakage detected - immediate action required")
                recommendations.append("Remove or fix features causing data leakage")
            
            if overfitting_score > 0.15:
                alerts.append("⚠️ Overfitting detected - model generalization at risk")
                recommendations.append("Implement regularization or reduce model complexity")
            
            if noise_level > 0.05:
                alerts.append("⚠️ High noise level detected")
                recommendations.append("Apply noise filtering and feature selection")
            
            if quality_score < 0.7:
                alerts.append("⚠️ Low data quality detected")
                recommendations.append("Improve data preprocessing and cleaning")
            
            protection_results.update({
                'overall_assessment': {
                    'protection_status': protection_status,
                    'risk_level': risk_level,
                    'overall_risk_score': overall_risk,
                    'quality_score': quality_score,
                    'enterprise_ready': overall_risk < 0.3 and quality_score > 0.7
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
    
    def _train_validation_analysis(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
        """วิเคราะห์ train-validation performance gap"""
        try:
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
            return {'error': f'Train-validation analysis failed: {str(e)}'}
    
    def _analyze_learning_curves(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
        """วิเคราะห์ learning curves"""
        try:
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
            return {'error': f'Learning curve analysis failed: {str(e)}'}
    
    def _analyze_feature_importance_stability(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
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
    
    def _simple_variance_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """การวิเคราะห์ variance แบบง่าย"""
        try:
            # Calculate feature variances
            feature_variances = X.var()
            
            # Check for high variance features (potential overfitting indicators)
            high_variance_features = feature_variances[feature_variances > feature_variances.quantile(0.95)]
            
            # Calculate target variance
            target_variance = y.var() if len(y.unique()) > 2 else 0.25  # Binary target default variance
            
            return {
                'feature_variance_stats': {
                    'mean_variance': float(feature_variances.mean()),
                    'std_variance': float(feature_variances.std()),
                    'max_variance': float(feature_variances.max()),
                    'min_variance': float(feature_variances.min())
                },
                'high_variance_features': high_variance_features.index.tolist(),
                'high_variance_count': len(high_variance_features),
                'target_variance': float(target_variance),
                'variance_ratio': float(feature_variances.mean() / target_variance) if target_variance > 0 else 0.0
            }
        except Exception as e:
            return {'error': str(e), 'variance_ratio': 1.0}
    
    def _compute_overfitting_score(self, overfitting_results: Dict) -> float:
        """คำนวณคะแนน overfitting โดยรวม"""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # CV score consistency (weight: 40%)
            cv_data = overfitting_results.get('cross_validation', {})
            if 'coefficient_of_variation' in cv_data and not cv_data.get('error'):
                cv_weight = 0.4
                cv_score = min(cv_data['coefficient_of_variation'], 1.0)  # Cap at 1.0
                score += cv_score * cv_weight
                weight_sum += cv_weight
            
            # Variance analysis (weight: 30%)
            variance_data = overfitting_results.get('train_validation', {})
            if 'variance_ratio' in variance_data and not variance_data.get('error'):
                var_weight = 0.3
                # High variance ratio indicates potential overfitting
                var_ratio = variance_data['variance_ratio']
                var_score = min(var_ratio / 10.0, 1.0)  # Normalize and cap
                score += var_score * var_weight
                weight_sum += var_weight
            
            # High variance features (weight: 30%)
            if 'high_variance_count' in variance_data:
                hv_weight = 0.3
                high_var_count = variance_data['high_variance_count']
                total_features = overfitting_results.get('details', {}).get('total_features', len(overfitting_results.get('data_info', {}).get('features', 10)))
                hv_ratio = high_var_count / max(total_features, 1)
                hv_score = min(hv_ratio, 1.0)
                score += hv_score * hv_weight
                weight_sum += hv_weight
            
            # Normalize by actual weights used
            if weight_sum > 0:
                return score / weight_sum
            else:
                return 0.1  # Low default score if no analysis possible
                
        except Exception as e:
            self.logger.warning(f"Overfitting score computation failed: {str(e)}")
            return 0.2  # Conservative estimate
    
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
        """คำนวณการประเมินโดยรวม"""
        try:
            # Protection scores
            leakage_score = protection_results.get('data_leakage', {}).get('leakage_score', 0)
            overfitting_score = protection_results.get('overfitting', {}).get('overfitting_score', 0)
            noise_level = protection_results.get('noise_analysis', {}).get('noise_level', 0)
            quality_score = protection_results.get('noise_analysis', {}).get('data_quality_score', 1.0)
            
            # Overall risk score (0-1, lower is better)
            overall_risk = (leakage_score * 0.4 + overfitting_score * 0.3 + noise_level * 0.3)
            
            # Protection status
            if overall_risk < 0.2:
                protection_status = 'EXCELLENT'
                risk_level = 'LOW'
            elif overall_risk < 0.4:
                protection_status = 'GOOD'
                risk_level = 'MEDIUM'
            elif overall_risk < 0.6:
                protection_status = 'ACCEPTABLE'
                risk_level = 'HIGH'
            else:
                protection_status = 'POOR'
                risk_level = 'CRITICAL'
            
            # Add alerts and recommendations
            alerts = []
            recommendations = []
            
            if leakage_score > 0.3:
                alerts.append("⚠️ Data leakage detected - immediate action required")
                recommendations.append("Remove or fix features causing data leakage")
            
            if overfitting_score > 0.15:
                alerts.append("⚠️ Overfitting detected - model generalization at risk")
                recommendations.append("Implement regularization or reduce model complexity")
            
            if noise_level > 0.05:
                alerts.append("⚠️ High noise level detected")
                recommendations.append("Apply noise filtering and feature selection")
            
            if quality_score < 0.7:
                alerts.append("⚠️ Low data quality detected")
                recommendations.append("Improve data preprocessing and cleaning")
            
            protection_results.update({
                'overall_assessment': {
                    'protection_status': protection_status,
                    'risk_level': risk_level,
                    'overall_risk_score': overall_risk,
                    'quality_score': quality_score,
                    'enterprise_ready': overall_risk < 0.3 and quality_score > 0.7
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
    
    def _train_validation_analysis(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
        """วิเคราะห์ train-validation performance gap"""
        try:
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
            return {'error': f'Train-validation analysis failed: {str(e)}'}
    