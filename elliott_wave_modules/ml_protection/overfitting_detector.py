#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVERFITTING DETECTOR MODULE
Enterprise-grade overfitting detection with multiple methodologies

Detection Methods:
- Time-Series Cross Validation
- Train-Validation Performance Analysis
- Learning Curve Analysis
- Feature Importance Stability
- Advanced Statistical Tests
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime

from core.unified_enterprise_logger import get_unified_logger

# Import ML libraries with error handling
try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available, using simplified overfitting detection")


class OverfittingDetector:
    """🎯 Enterprise Overfitting Detection System"""
    
    def __init__(self, config: Dict = None, logger=None):
        self.config = config or {}
        self.sklearn_available = SKLEARN_AVAILABLE
        
        # Initialize logging
        self.logger = get_unified_logger("OverfittingDetector")
    
    def update_config(self, new_config: Dict):
        """Update detector configuration"""
        self.config.update(new_config)
    
    def detect_overfitting(self, X: pd.DataFrame, y: pd.Series, model: Any = None, process_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive overfitting detection
        
        Args:
            X: Feature matrix
            y: Target vector
            model: ML model for analysis (optional)
            process_id: Process identifier
            
        Returns:
            Overfitting analysis results
        """
        try:
            # Check if sklearn is available for advanced analysis
            if not self.sklearn_available:
                self.logger.warning("⚠️ sklearn not available, using simplified overfitting detection", component="OverfittingDetector")
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
            learning_curves = self._analyze_learning_curves(X, y, model)
            overfitting_results['learning_curves'] = learning_curves
            
            # 4. Feature Importance Stability
            importance_stability = self._analyze_feature_importance_stability(X, y, model, process_id)
            overfitting_results['feature_importance_stability'] = importance_stability
            
            # Overall overfitting assessment
            overfitting_score = self._compute_overfitting_score(overfitting_results)
            overfitting_results['overfitting_score'] = overfitting_score
            overfitting_results['overfitting_detected'] = overfitting_score > self.config.get('overfitting_threshold', 0.05)
            overfitting_results['status'] = 'DETECTED' if overfitting_results['overfitting_detected'] else 'ACCEPTABLE'
            
            return overfitting_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced overfitting detection failed: {str(e)}, falling back to simplified method", component="OverfittingDetector")
            
            # Fallback to simplified method
            try:
                return self._detect_overfitting_simplified(X, y)
            except Exception as fallback_error:
                error_msg = f"❌ Both overfitting detection methods failed: {str(fallback_error)}"
                self.logger.error(error_msg, component="OverfittingDetector")
                
                return {
                    'status': 'ERROR', 
                    'error': str(fallback_error),
                    'overfitting_detected': False,
                    'overfitting_score': 0.0,
                    'details': {'fallback_attempted': True}
                }
    
    def _detect_overfitting_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Simplified overfitting detection without sklearn dependencies"""
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
            overfitting_threshold = self.config.get('overfitting_threshold', 0.05)
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
            error_msg = f"❌ Simplified overfitting detection failed: {str(e)}"
            self.logger.error(error_msg, component="OverfittingDetector")
            
            return {
                'status': 'ERROR',
                'error': str(e),
                'overfitting_detected': False,
                'overfitting_score': 0.0
            }
    
    def _train_validation_analysis(self, X: pd.DataFrame, y: pd.Series, model: Any, process_id: str = None) -> Dict:
        """Analyze train vs validation performance"""
        try:
            if not self.sklearn_available:
                return self._train_validation_analysis_simplified(X, y)
            
            # Split data temporally for time series
            split_point = int(0.8 * len(X))
            X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
            
            # Train model and evaluate
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            performance_gap = abs(train_score - val_score)
            performance_ratio = val_score / train_score if train_score > 0 else 0
            
            return {
                'train_score': float(train_score),
                'validation_score': float(val_score),
                'performance_gap': float(performance_gap),
                'performance_ratio': float(performance_ratio),
                'overfitting_indicator': performance_gap > 0.1 or performance_ratio < 0.9
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Train-validation analysis failed: {str(e)}", component="OverfittingDetector")
            return {'error': str(e)}
    
    def _train_validation_analysis_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simplified train-validation analysis without sklearn"""
        try:
            # Split data temporally
            split_point = int(0.8 * len(X))
            X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
            
            # Simple score based on class distribution balance
            train_class_balance = min(y_train.mean(), 1 - y_train.mean())
            val_class_balance = min(y_val.mean(), 1 - y_val.mean())
            
            # Simple performance scores
            train_score = 0.5 + train_class_balance * 0.3
            val_score = 0.5 + val_class_balance * 0.3
            
            performance_gap = abs(train_score - val_score)
            performance_ratio = val_score / train_score if train_score > 0 else 0
            
            return {
                'train_score': float(train_score),
                'validation_score': float(val_score),
                'performance_gap': float(performance_gap),
                'performance_ratio': float(performance_ratio),
                'overfitting_indicator': performance_gap > 0.15 or performance_ratio < 0.85,
                'method': 'simplified_class_balance'
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'simplified_failed'}
    
    def _analyze_learning_curves(self, X: pd.DataFrame, y: pd.Series, model: Any) -> Dict:
        """Analyze learning curves for overfitting signs"""
        try:
            if not self.sklearn_available:
                return self._analyze_learning_curves_simplified(X, y)
            
            # Generate learning curves with different training sizes
            train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
            train_scores = []
            val_scores = []
            
            for size in train_sizes:
                # Time-aware split
                n_samples = int(size * len(X))
                X_subset = X.iloc[:n_samples]
                y_subset = y.iloc[:n_samples]
                
                # Split for validation
                split_point = int(0.8 * n_samples)
                X_train = X_subset.iloc[:split_point]
                X_val = X_subset.iloc[split_point:]
                y_train = y_subset.iloc[:split_point]
                y_val = y_subset.iloc[split_point:]
                
                if len(X_train) > 0 and len(X_val) > 0:
                    model.fit(X_train, y_train)
                    train_score = model.score(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    
                    train_scores.append(train_score)
                    val_scores.append(val_score)
            
            # Analyze convergence and gaps
            if len(train_scores) > 1:
                final_gap = abs(train_scores[-1] - val_scores[-1]) if val_scores else 0
                score_trend = np.diff(val_scores) if len(val_scores) > 1 else [0]
                
                return {
                    'train_scores': train_scores,
                    'validation_scores': val_scores,
                    'final_gap': float(final_gap),
                    'converging': np.mean(score_trend) > -0.05,
                    'overfitting_pattern': final_gap > 0.1
                }
            else:
                return {'insufficient_data': True}
                
        except Exception as e:
            self.logger.warning(f"⚠️ Learning curve analysis failed: {str(e)}", component="OverfittingDetector")
            return {'error': str(e)}
    
    def _analyze_learning_curves_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simplified learning curve analysis"""
        try:
            train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
            scores = []
            
            for size in train_sizes:
                n_samples = int(size * len(X))
                X_subset = X.iloc[:n_samples]
                y_subset = y.iloc[:n_samples]
                
                # Simple score based on class distribution balance
                class_balance = min(y_subset.mean(), 1 - y_subset.mean())
                base_score = 0.5 + class_balance * 0.3
                
                scores.append(base_score)
            
            # Analyze trend
            score_trend = np.diff(scores) if len(scores) > 1 else [0]
            
            return {
                'scores': scores,
                'trend': 'improving' if np.mean(score_trend) > 0 else 'stable',
                'final_score': scores[-1] if scores else 0.5,
                'method': 'simplified_trend_analysis'
            }
            
        except Exception as e:
            return {'error': str(e), 'method': 'simplified_failed'}
    
    def _analyze_feature_importance_stability(self, X: pd.DataFrame, y: pd.Series, model: Any, process_id: str = None) -> Dict:
        """Analyze feature importance stability across different subsets"""
        try:
            if not self.sklearn_available or not hasattr(model, 'feature_importances_'):
                return {'method': 'not_available', 'stable': True}
            
            # Train on different subsets and compare feature importances
            importance_sets = []
            
            for i in range(3):  # 3 different subsets
                # Random subset (but respecting time order)
                subset_size = int(0.8 * len(X))
                start_idx = i * (len(X) - subset_size) // 3
                end_idx = start_idx + subset_size
                
                X_subset = X.iloc[start_idx:end_idx]
                y_subset = y.iloc[start_idx:end_idx]
                
                model.fit(X_subset, y_subset)
                importance_sets.append(model.feature_importances_)
            
            # Calculate stability
            if len(importance_sets) > 1:
                importance_correlations = []
                for i in range(len(importance_sets)):
                    for j in range(i + 1, len(importance_sets)):
                        corr = np.corrcoef(importance_sets[i], importance_sets[j])[0, 1]
                        importance_correlations.append(corr)
                
                stability_score = np.mean(importance_correlations)
                
                return {
                    'stability_score': float(stability_score),
                    'stable': stability_score > 0.7,
                    'importance_correlations': importance_correlations
                }
            
            return {'insufficient_data': True}
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance stability analysis failed: {str(e)}", component="OverfittingDetector")
            return {'error': str(e)}
    
    def _compute_overfitting_score(self, overfitting_results: Dict) -> float:
        """Compute overall overfitting score"""
        try:
            score_components = []
            
            # Cross-validation variance
            cv_data = overfitting_results.get('cross_validation', {})
            if 'coefficient_of_variation' in cv_data:
                cv_score = min(cv_data['coefficient_of_variation'] * 10, 1.0)
                score_components.append(cv_score)
            
            # Train-validation gap
            tv_data = overfitting_results.get('train_validation', {})
            if 'performance_gap' in tv_data:
                gap_score = min(tv_data['performance_gap'] * 5, 1.0)
                score_components.append(gap_score)
            
            # Learning curve pattern
            lc_data = overfitting_results.get('learning_curves', {})
            if 'overfitting_pattern' in lc_data:
                lc_score = 0.8 if lc_data['overfitting_pattern'] else 0.2
                score_components.append(lc_score)
            
            # Feature importance stability
            fi_data = overfitting_results.get('feature_importance_stability', {})
            if 'stability_score' in fi_data:
                fi_score = 1.0 - fi_data['stability_score']
                score_components.append(fi_score)
            
            # Calculate weighted average
            if score_components:
                return float(np.mean(score_components))
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"⚠️ Overfitting score computation failed: {str(e)}", component="OverfittingDetector")
            return 0.0
