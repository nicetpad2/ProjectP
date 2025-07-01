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
    
    def __init__(self, config: Dict = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or {}
        
        # Default protection configuration
        default_config = {
            'overfitting_threshold': 0.15,  # Max difference between train/val
            'noise_threshold': 0.05,        # Max noise ratio allowed
            'leak_detection_window': 100,   # Samples to check for leakage
            'min_samples_split': 50,        # Minimum samples for time split
            'stability_window': 1000,       # Window for feature stability
            'significance_level': 0.05,     # Statistical significance level
        }
        
        # Merge with ml_protection config from main config
        ml_protection_config = self.config.get('ml_protection', {})
        self.protection_config = {**default_config, **ml_protection_config}
        self.protection_results = {}
        
        # Log configuration
        self.logger.info(f"🛡️ Enterprise ML Protection System initialized with config: {self.protection_config}")
    
    def update_protection_config(self, new_config: Dict):
        """อัปเดตการตั้งค่าป้องกัน"""
        self.protection_config.update(new_config)
        self.logger.info(f"🔧 Protection config updated: {new_config}")
    
    def get_protection_config(self) -> Dict:
        """ดึงการตั้งค่าป้องกันปัจจุบัน"""
        return self.protection_config.copy()
        
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
            
            protection_results = {
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'samples': len(X),
                    'features': len(X.columns),
                    'target_distribution': y.value_counts().to_dict()
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
            train_val_analysis = self._train_validation_analysis(X, y, model)
            overfitting_results['train_validation'] = train_val_analysis
            
            # 3. Learning Curve Analysis
            learning_curves = self._analyze_learning_curves(X, y, model)
            overfitting_results['learning_curves'] = learning_curves
            
            # 4. Feature Importance Stability
            importance_stability = self._analyze_feature_importance_stability(X, y, model)
            overfitting_results['feature_importance_stability'] = importance_stability
            
            # Overall overfitting assessment
            overfitting_score = self._compute_overfitting_score(overfitting_results)
            overfitting_results['overfitting_score'] = overfitting_score
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
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_outliers = (z_scores > 3).sum()
            
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
            
            # Normality test
            try:
                _, p_value = shapiro(data.sample(min(5000, len(data))))
                is_normal = p_value > 0.05
            except:
                is_normal = False
            
            # Skewness and kurtosis
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
            
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
            # Mutual information scores
            mi_scores = mutual_info_classif(X.select_dtypes(include=[np.number]), y, random_state=42)
            
            # Correlation with target
            correlations = []
            for col in X.select_dtypes(include=[np.number]).columns:
                corr = abs(X[col].corr(y))
                correlations.append(corr)
            
            # Categorize features
            relevant_features = []
            irrelevant_features = []
            
            for i, (col, mi_score, corr) in enumerate(zip(X.columns, mi_scores, correlations)):
                if mi_score > 0.01 or corr > 0.05:
                    relevant_features.append(col)
                else:
                    irrelevant_features.append(col)
            
            return {
                'relevant_features': relevant_features,
                'irrelevant_features': irrelevant_features,
                'mutual_info_scores': dict(zip(X.columns, mi_scores.tolist())),
                'correlation_scores': dict(zip(X.columns, correlations)),
                'relevance_ratio': len(relevant_features) / len(X.columns)
            }
        except Exception as e:
            return {'error': f'Feature relevance analysis failed: {str(e)}'}
    
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
            
            # Sort by datetime
            X_sorted = X.sort_values(datetime_col)
            
            # Split into first and last thirds
            n = len(X_sorted)
            first_third = X_sorted.iloc[:n//3]
            last_third = X_sorted.iloc[2*n//3:]
            
            for col in X_sorted.select_dtypes(include=[np.number]).columns:
                if col == datetime_col:
                    continue
                
                # Get non-null values
                first_values = first_third[col].dropna()
                last_values = last_third[col].dropna()
                
                if len(first_values) < 10 or len(last_values) < 10:
                    continue
                
                # Kolmogorov-Smirnov test for distribution change
                try:
                    ks_stat, p_value = ks_2samp(first_values, last_values)
                    drift_detected = p_value < 0.05
                    
                    drift_results[col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'drift_detected': drift_detected,
                        'severity': 'HIGH' if ks_stat > 0.3 else 'MEDIUM' if ks_stat > 0.1 else 'LOW'
                    }
                except:
                    pass
            
            return drift_results
            
        except Exception as e:
            return {'error': f'Temporal drift detection failed: {str(e)}'}
    
    def _detect_temporal_gaps(self, datetime_series: pd.Series) -> List[Dict]:
        """ตรวจหาช่องว่างใน time series"""
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(datetime_series):
                datetime_series = pd.to_datetime(datetime_series)
            
            # Sort and find gaps
            sorted_times = datetime_series.sort_values()
            time_diffs = sorted_times.diff()
            
            # Define expected frequency (assume most common difference)
            median_diff = time_diffs.median()
            
            # Find gaps larger than 2x expected frequency
            large_gaps = time_diffs[time_diffs > median_diff * 2]
            
            gaps = []
            for idx, gap in large_gaps.items():
                gaps.append({
                    'start_time': sorted_times.iloc[idx-1],
                    'end_time': sorted_times.iloc[idx],
                    'gap_duration': gap,
                    'gap_ratio': gap / median_diff
                })
            
            return gaps
            
        except:
            return []
    
    def _analyze_seasonality(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict:
        """วิเคราะห์ seasonality ใน data"""
        try:
            # This is a simplified seasonality analysis
            # In production, you might want to use more sophisticated methods
            
            if datetime_col not in X.columns:
                return {'error': 'Datetime column not found'}
            
            # Convert to datetime
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


# Alias for backward compatibility
DataProcessor = EnterpriseMLProtectionSystem
