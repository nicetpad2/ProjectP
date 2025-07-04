#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEATURE STABILITY ANALYZER MODULE
Enterprise-grade feature stability and drift analysis

Analysis Methods:
- Feature Stability Over Time
- Feature Drift Detection
- Feature Correlation Stability
- Feature Importance Stability
- Window-based Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime

# Import advanced logging system
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    import logging


class FeatureStabilityAnalyzer:
    """📈 Enterprise Feature Stability Analysis System"""
    
    def __init__(self, config: Dict = None, logger=None):
        self.config = config or {}
        
        # Initialize logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
        else:
            self.logger = logger or logging.getLogger(__name__)
    
    def update_config(self, new_config: Dict):
        """Update analyzer configuration"""
        self.config.update(new_config)
    
    def analyze_feature_stability(self, X: pd.DataFrame, y: pd.Series, datetime_col: str = None) -> Dict[str, Any]:
        """
        Comprehensive feature stability analysis
        
        Args:
            X: Feature matrix
            y: Target vector
            datetime_col: Name of datetime column for temporal analysis
            
        Returns:
            Feature stability analysis results
        """
        try:
            stability_results = {
                'status': 'ANALYZING',
                'stability_score': 0.0,
                'unstable_features': [],
                'drift_detected': False,
                'feature_analysis': {},
                'temporal_analysis': {},
                'recommendations': []
            }
            
            # 1. Basic Feature Stability Analysis
            basic_stability = self._analyze_basic_stability(X, y)
            stability_results['basic_stability'] = basic_stability
            
            # 2. Feature Drift Analysis
            drift_analysis = self._analyze_feature_drift(X, datetime_col)
            stability_results['drift_analysis'] = drift_analysis
            
            # 3. Correlation Stability Analysis
            correlation_stability = self._analyze_correlation_stability(X, y)
            stability_results['correlation_stability'] = correlation_stability
            
            # 4. Window-based Stability Analysis
            if datetime_col and datetime_col in X.columns:
                window_analysis = self._analyze_windowed_stability(X, y, datetime_col)
                stability_results['window_analysis'] = window_analysis
            
            # Overall stability assessment
            overall_stability = self._compute_overall_stability(stability_results)
            stability_results['stability_score'] = overall_stability['score']
            stability_results['drift_detected'] = overall_stability['drift_detected']
            stability_results['unstable_features'] = overall_stability['unstable_features']
            stability_results['status'] = 'UNSTABLE' if overall_stability['drift_detected'] else 'STABLE'
            
            # Generate recommendations
            recommendations = self._generate_stability_recommendations(stability_results)
            stability_results['recommendations'] = recommendations
            
            return stability_results
            
        except Exception as e:
            error_msg = f"❌ Feature stability analysis failed: {str(e)}"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(error_msg, "FeatureAnalyzer")
            else:
                self.logger.error(error_msg)
            return {'status': 'ERROR', 'error': str(e)}
    
    def _analyze_basic_stability(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze basic feature stability metrics"""
        try:
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for stability analysis'}
            
            stability_metrics = {
                'feature_stability': {},
                'overall_variance_score': 0.0,
                'high_variance_features': []
            }
            
            variance_scores = []
            
            for col in numeric_X.columns:
                try:
                    col_data = numeric_X[col].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Calculate coefficient of variation
                    if col_data.std() > 0 and col_data.mean() != 0:
                        cv = col_data.std() / abs(col_data.mean())
                    else:
                        cv = 0.0
                    
                    # Calculate stability score (inverse of CV)
                    stability_score = 1.0 / (1.0 + cv)
                    
                    # Additional metrics
                    feature_metrics = {
                        'coefficient_of_variation': float(cv),
                        'stability_score': float(stability_score),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'range_ratio': float((col_data.max() - col_data.min()) / max(abs(col_data.mean()), 1e-8))
                    }
                    
                    stability_metrics['feature_stability'][col] = feature_metrics
                    variance_scores.append(stability_score)
                    
                    # Check for high variance
                    max_cv_threshold = 2.0  # Configurable threshold
                    if cv > max_cv_threshold:
                        stability_metrics['high_variance_features'].append({
                            'feature': col,
                            'coefficient_of_variation': float(cv),
                            'stability_score': float(stability_score),
                            'severity': 'HIGH' if cv > max_cv_threshold * 2 else 'MEDIUM'
                        })
                
                except Exception as col_error:
                    stability_metrics['feature_stability'][col] = {
                        'error': f'Stability analysis failed: {str(col_error)}'
                    }
            
            # Overall variance score
            if variance_scores:
                stability_metrics['overall_variance_score'] = float(np.mean(variance_scores))
            
            return stability_metrics
            
        except Exception as e:
            return {'error': f'Basic stability analysis failed: {str(e)}'}
    
    def _analyze_feature_drift(self, X: pd.DataFrame, datetime_col: str = None) -> Dict:
        """Analyze feature drift over time"""
        try:
            if not datetime_col or datetime_col not in X.columns:
                return self._analyze_feature_drift_no_time(X)
            
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for drift analysis'}
            
            # Sort by datetime
            sorted_data = X.sort_values(datetime_col)
            
            drift_results = {
                'feature_drift': {},
                'overall_drift_score': 0.0,
                'drifted_features': []
            }
            
            drift_scores = []
            
            # Analyze drift using window-based comparison
            window_size = min(len(sorted_data) // 4, self.config.get('stability_window', 2000))
            window_size = max(window_size, 100)  # Minimum window size
            
            for col in numeric_X.columns:
                try:
                    col_data = sorted_data[col].dropna()
                    if len(col_data) < window_size * 2:
                        continue
                    
                    # Compare first and last windows
                    first_window = col_data.iloc[:window_size]
                    last_window = col_data.iloc[-window_size:]
                    
                    # Statistical comparison
                    mean_diff = abs(last_window.mean() - first_window.mean())
                    std_diff = abs(last_window.std() - first_window.std())
                    
                    # Normalize differences
                    mean_scale = max(abs(first_window.mean()), 1e-8)
                    std_scale = max(first_window.std(), 1e-8)
                    
                    normalized_mean_diff = mean_diff / mean_scale
                    normalized_std_diff = std_diff / std_scale
                    
                    # Drift score
                    drift_score = min((normalized_mean_diff + normalized_std_diff) / 2, 1.0)
                    
                    feature_drift = {
                        'drift_score': float(drift_score),
                        'mean_drift': float(normalized_mean_diff),
                        'std_drift': float(normalized_std_diff),
                        'first_window_mean': float(first_window.mean()),
                        'last_window_mean': float(last_window.mean()),
                        'first_window_std': float(first_window.std()),
                        'last_window_std': float(last_window.std())
                    }
                    
                    drift_results['feature_drift'][col] = feature_drift
                    drift_scores.append(drift_score)
                    
                    # Check for significant drift
                    max_drift_threshold = self.config.get('max_feature_drift', 0.10)
                    if drift_score > max_drift_threshold:
                        drift_results['drifted_features'].append({
                            'feature': col,
                            'drift_score': float(drift_score),
                            'severity': 'HIGH' if drift_score > max_drift_threshold * 2 else 'MEDIUM'
                        })
                
                except Exception as col_error:
                    drift_results['feature_drift'][col] = {
                        'error': f'Drift analysis failed: {str(col_error)}'
                    }
            
            # Overall drift score
            if drift_scores:
                drift_results['overall_drift_score'] = float(np.mean(drift_scores))
            
            return drift_results
            
        except Exception as e:
            return {'error': f'Feature drift analysis failed: {str(e)}'}
    
    def _analyze_feature_drift_no_time(self, X: pd.DataFrame) -> Dict:
        """Analyze feature drift without temporal information"""
        try:
            numeric_X = X.select_dtypes(include=[np.number])
            
            # Split data into chunks and compare
            n_chunks = 4
            chunk_size = len(numeric_X) // n_chunks
            
            if chunk_size < 50:
                return {'error': 'Insufficient data for drift analysis'}
            
            drift_results = {
                'feature_drift': {},
                'overall_drift_score': 0.0,
                'drifted_features': [],
                'method': 'chunk_based_no_time'
            }
            
            drift_scores = []
            
            for col in numeric_X.columns:
                try:
                    col_data = numeric_X[col].dropna()
                    if len(col_data) < chunk_size * 2:
                        continue
                    
                    # Compare first and last chunks
                    first_chunk = col_data.iloc[:chunk_size]
                    last_chunk = col_data.iloc[-chunk_size:]
                    
                    # Statistical comparison
                    mean_diff = abs(last_chunk.mean() - first_chunk.mean())
                    std_diff = abs(last_chunk.std() - first_chunk.std())
                    
                    # Normalize differences
                    mean_scale = max(abs(first_chunk.mean()), 1e-8)
                    std_scale = max(first_chunk.std(), 1e-8)
                    
                    normalized_mean_diff = mean_diff / mean_scale
                    normalized_std_diff = std_diff / std_scale
                    
                    # Drift score
                    drift_score = min((normalized_mean_diff + normalized_std_diff) / 2, 1.0)
                    
                    drift_results['feature_drift'][col] = {
                        'drift_score': float(drift_score),
                        'mean_drift': float(normalized_mean_diff),
                        'std_drift': float(normalized_std_diff)
                    }
                    
                    drift_scores.append(drift_score)
                    
                    # Check for significant drift
                    if drift_score > 0.15:  # Higher threshold without time info
                        drift_results['drifted_features'].append({
                            'feature': col,
                            'drift_score': float(drift_score),
                            'severity': 'HIGH' if drift_score > 0.3 else 'MEDIUM'
                        })
                
                except Exception as col_error:
                    drift_results['feature_drift'][col] = {
                        'error': f'Drift analysis failed: {str(col_error)}'
                    }
            
            # Overall drift score
            if drift_scores:
                drift_results['overall_drift_score'] = float(np.mean(drift_scores))
            
            return drift_results
            
        except Exception as e:
            return {'error': f'No-time drift analysis failed: {str(e)}'}
    
    def _analyze_correlation_stability(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze stability of feature correlations"""
        try:
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for correlation stability analysis'}
            
            correlation_results = {
                'target_correlation_stability': {},
                'inter_feature_correlation_stability': {},
                'overall_correlation_stability': 0.0,
                'unstable_correlations': []
            }
            
            # Split data into parts for comparison
            split_point = len(numeric_X) // 2
            if split_point < 50:
                return {'error': 'Insufficient data for correlation stability analysis'}
            
            X_first = numeric_X.iloc[:split_point]
            X_second = numeric_X.iloc[split_point:]
            y_first = y.iloc[:split_point]
            y_second = y.iloc[split_point:]
            
            stability_scores = []
            
            # Target correlation stability
            for col in numeric_X.columns:
                try:
                    corr_first = X_first[col].corr(y_first)
                    corr_second = X_second[col].corr(y_second)
                    
                    if pd.isna(corr_first):
                        corr_first = 0.0
                    if pd.isna(corr_second):
                        corr_second = 0.0
                    
                    correlation_diff = abs(corr_first - corr_second)
                    stability_score = 1.0 - min(correlation_diff, 1.0)
                    
                    correlation_results['target_correlation_stability'][col] = {
                        'first_half_correlation': float(corr_first),
                        'second_half_correlation': float(corr_second),
                        'correlation_difference': float(correlation_diff),
                        'stability_score': float(stability_score)
                    }
                    
                    stability_scores.append(stability_score)
                    
                    # Check for unstable correlations
                    if correlation_diff > 0.2:
                        correlation_results['unstable_correlations'].append({
                            'feature': col,
                            'correlation_difference': float(correlation_diff),
                            'stability_score': float(stability_score),
                            'type': 'target_correlation'
                        })
                
                except Exception as col_error:
                    correlation_results['target_correlation_stability'][col] = {
                        'error': f'Correlation stability analysis failed: {str(col_error)}'
                    }
            
            # Overall correlation stability
            if stability_scores:
                correlation_results['overall_correlation_stability'] = float(np.mean(stability_scores))
            
            return correlation_results
            
        except Exception as e:
            return {'error': f'Correlation stability analysis failed: {str(e)}'}
    
    def _analyze_windowed_stability(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict:
        """Analyze stability using sliding windows"""
        try:
            # Sort by datetime
            sorted_data = X.sort_values(datetime_col)
            sorted_y = y.loc[sorted_data.index]
            
            window_size = min(len(sorted_data) // 5, 1000)
            window_size = max(window_size, 100)
            
            if len(sorted_data) < window_size * 2:
                return {'error': 'Insufficient data for windowed analysis'}
            
            numeric_X = sorted_data.select_dtypes(include=[np.number])
            
            window_results = {
                'window_statistics': [],
                'stability_trends': {},
                'overall_trend_stability': 0.0
            }
            
            # Create overlapping windows
            step_size = window_size // 2
            n_windows = (len(sorted_data) - window_size) // step_size + 1
            
            window_stats = []
            
            for i in range(min(n_windows, 10)):  # Limit to 10 windows for performance
                start_idx = i * step_size
                end_idx = start_idx + window_size
                
                window_X = numeric_X.iloc[start_idx:end_idx]
                window_y = sorted_y.iloc[start_idx:end_idx]
                
                # Calculate window statistics
                window_stat = {
                    'window_id': i,
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'feature_means': {},
                    'feature_stds': {},
                    'target_correlations': {}
                }
                
                for col in window_X.columns:
                    try:
                        col_data = window_X[col].dropna()
                        if len(col_data) > 0:
                            window_stat['feature_means'][col] = float(col_data.mean())
                            window_stat['feature_stds'][col] = float(col_data.std())
                            
                            corr = col_data.corr(window_y.loc[col_data.index])
                            window_stat['target_correlations'][col] = float(corr) if not pd.isna(corr) else 0.0
                    except:
                        continue
                
                window_stats.append(window_stat)
            
            window_results['window_statistics'] = window_stats
            
            # Analyze trends across windows
            if len(window_stats) > 1:
                for col in numeric_X.columns:
                    means = [w['feature_means'].get(col, 0) for w in window_stats if col in w['feature_means']]
                    stds = [w['feature_stds'].get(col, 0) for w in window_stats if col in w['feature_stds']]
                    
                    if len(means) > 1:
                        mean_trend = np.polyfit(range(len(means)), means, 1)[0] if len(means) > 1 else 0
                        std_trend = np.polyfit(range(len(stds)), stds, 1)[0] if len(stds) > 1 else 0
                        
                        window_results['stability_trends'][col] = {
                            'mean_trend': float(mean_trend),
                            'std_trend': float(std_trend),
                            'trend_stability': float(1.0 - min(abs(mean_trend) + abs(std_trend), 1.0))
                        }
            
            return window_results
            
        except Exception as e:
            return {'error': f'Windowed stability analysis failed: {str(e)}'}
    
    def _compute_overall_stability(self, stability_results: Dict) -> Dict:
        """Compute overall stability assessment"""
        try:
            assessment = {
                'score': 0.0,
                'drift_detected': False,
                'unstable_features': [],
                'stability_level': 'UNKNOWN'
            }
            
            stability_components = []
            
            # Basic stability component
            basic_stability = stability_results.get('basic_stability', {})
            overall_variance = basic_stability.get('overall_variance_score', 0.5)
            stability_components.append(overall_variance)
            
            # Drift analysis component
            drift_analysis = stability_results.get('drift_analysis', {})
            drift_score = drift_analysis.get('overall_drift_score', 0.0)
            drift_stability = 1.0 - drift_score
            stability_components.append(drift_stability)
            
            # Correlation stability component
            corr_stability = stability_results.get('correlation_stability', {})
            corr_score = corr_stability.get('overall_correlation_stability', 0.5)
            stability_components.append(corr_score)
            
            # Overall stability score
            if stability_components:
                assessment['score'] = float(np.mean(stability_components))
            
            # Determine drift detection
            max_drift_threshold = self.config.get('max_feature_drift', 0.10)
            assessment['drift_detected'] = drift_score > max_drift_threshold
            
            # Collect unstable features
            unstable_features = []
            
            # From high variance features
            high_variance = basic_stability.get('high_variance_features', [])
            unstable_features.extend([f['feature'] for f in high_variance])
            
            # From drifted features
            drifted_features = drift_analysis.get('drifted_features', [])
            unstable_features.extend([f['feature'] for f in drifted_features])
            
            # From unstable correlations
            unstable_corrs = corr_stability.get('unstable_correlations', [])
            unstable_features.extend([f['feature'] for f in unstable_corrs])
            
            assessment['unstable_features'] = list(set(unstable_features))
            
            # Determine stability level
            if assessment['drift_detected'] or len(assessment['unstable_features']) > 0:
                assessment['stability_level'] = 'UNSTABLE'
            elif assessment['score'] >= 0.8:
                assessment['stability_level'] = 'HIGHLY_STABLE'
            elif assessment['score'] >= 0.6:
                assessment['stability_level'] = 'STABLE'
            else:
                assessment['stability_level'] = 'MODERATE'
            
            return assessment
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.warning(f"⚠️ Overall stability computation failed: {str(e)}", "FeatureAnalyzer")
            else:
                self.logger.warning(f"⚠️ Overall stability computation failed: {str(e)}")
            
            return {
                'score': 0.5,
                'drift_detected': False,
                'unstable_features': [],
                'stability_level': 'UNKNOWN'
            }
    
    def _generate_stability_recommendations(self, stability_results: Dict) -> List[str]:
        """Generate feature stability improvement recommendations"""
        try:
            recommendations = []
            
            # High variance features
            basic_stability = stability_results.get('basic_stability', {})
            if len(basic_stability.get('high_variance_features', [])) > 0:
                recommendations.append("Stabilize high-variance features: Consider robust scaling or feature transformation")
            
            # Feature drift
            drift_analysis = stability_results.get('drift_analysis', {})
            if len(drift_analysis.get('drifted_features', [])) > 0:
                recommendations.append("Address feature drift: Consider retraining models or adaptive feature engineering")
            
            # Correlation instability
            corr_stability = stability_results.get('correlation_stability', {})
            if len(corr_stability.get('unstable_correlations', [])) > 0:
                recommendations.append("Improve correlation stability: Consider feature selection or regularization")
            
            # Overall stability
            overall_score = stability_results.get('stability_score', 0.5)
            if overall_score < 0.6:
                recommendations.append("Overall stability is low: Consider comprehensive feature engineering review")
            
            return recommendations
            
        except Exception as e:
            return [f"Could not generate stability recommendations: {str(e)}"]
