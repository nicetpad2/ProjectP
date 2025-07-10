#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOISE ANALYZER MODULE
Enterprise-grade noise detection and data quality analysis

Analysis Methods:
- Missing Value Analysis
- Outlier Detection (Multiple Methods)
- Feature Distribution Analysis
- Signal-to-Noise Ratio Computation
- Feature Relevance Analysis
- Data Quality Scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime

# Import ML libraries with error handling
try:
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available, using simplified noise analysis")

try:
    from scipy import stats
    from scipy.stats import ks_2samp, shapiro
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available, using simplified statistical analysis")

# Import advanced logging system
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    import logging


class NoiseQualityAnalyzer:
    """📊 Enterprise Noise Detection and Data Quality Analysis System"""
    
    def __init__(self, config: Dict = None, logger=None):
        self.config = config or {}
        self.sklearn_available = SKLEARN_AVAILABLE
        self.scipy_available = SCIPY_AVAILABLE
        
        # Initialize logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
        else:
            self.logger = logger or logging.getLogger(__name__)
    
    def update_config(self, new_config: Dict):
        """Update analyzer configuration"""
        self.config.update(new_config)
    
    def analyze_noise_and_quality(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive noise detection and data quality analysis
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Noise and quality analysis results
        """
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
            noise_results['noise_detected'] = noise_level > self.config.get('noise_threshold', 0.02)
            noise_results['status'] = 'HIGH_NOISE' if noise_results['noise_detected'] else 'ACCEPTABLE'
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(noise_results)
            noise_results['recommendations'] = recommendations
            
            return noise_results
            
        except Exception as e:
            error_msg = f"❌ Noise detection failed: {str(e)}"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(error_msg, "NoiseAnalyzer")
            else:
                self.logger.error(error_msg)
            return {'status': 'ERROR', 'error': str(e)}
    
    def _analyze_missing_values(self, X: pd.DataFrame) -> Dict:
        """Analyze missing value patterns"""
        try:
            total_cells = X.shape[0] * X.shape[1]
            missing_cells = X.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
            
            # Per-feature missing analysis
            feature_missing = []
            for col in X.columns:
                missing_count = X[col].isnull().sum()
                missing_pct = missing_count / len(X) if len(X) > 0 else 0
                
                if missing_pct > 0:
                    feature_missing.append({
                        'feature': col,
                        'missing_count': int(missing_count),
                        'missing_percentage': float(missing_pct),
                        'severity': 'HIGH' if missing_pct > 0.3 else 'MEDIUM' if missing_pct > 0.1 else 'LOW'
                    })
            
            # Missing pattern analysis
            missing_patterns = {}
            if len(feature_missing) > 0:
                # Find features with similar missing patterns
                missing_mask = X.isnull()
                pattern_groups = []
                
                for i, col1 in enumerate(missing_mask.columns):
                    for j, col2 in enumerate(missing_mask.columns[i+1:], i+1):
                        # Check if missing patterns are similar
                        overlap = (missing_mask[col1] & missing_mask[col2]).sum()
                        total_missing = (missing_mask[col1] | missing_mask[col2]).sum()
                        
                        if total_missing > 0:
                            similarity = overlap / total_missing
                            if similarity > 0.8:  # High similarity
                                pattern_groups.append({
                                    'features': [col1, col2],
                                    'similarity': float(similarity),
                                    'overlap_count': int(overlap)
                                })
                
                missing_patterns['similar_patterns'] = pattern_groups
            
            return {
                'total_missing_ratio': float(missing_ratio),
                'total_missing_cells': int(missing_cells),
                'features_with_missing': feature_missing,
                'missing_patterns': missing_patterns,
                'quality_impact': 'HIGH' if missing_ratio > 0.2 else 'MEDIUM' if missing_ratio > 0.05 else 'LOW'
            }
            
        except Exception as e:
            return {'error': f'Missing value analysis failed: {str(e)}'}
    
    def _detect_outliers_comprehensive(self, X: pd.DataFrame) -> Dict:
        """Comprehensive outlier detection using multiple methods"""
        try:
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for outlier analysis'}
            
            outlier_results = {
                'methods_used': [],
                'feature_outliers': {},
                'overall_outlier_ratio': 0.0
            }
            
            total_outliers = 0
            total_observations = len(numeric_X)
            
            for col in numeric_X.columns:
                feature_outliers = {
                    'iqr_outliers': [],
                    'zscore_outliers': [],
                    'modified_zscore_outliers': [],
                    'combined_outliers': []
                }
                
                try:
                    col_data = numeric_X[col].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Method 1: IQR-based outlier detection
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        feature_outliers['iqr_outliers'] = iqr_outliers.index.tolist()
                    
                    # Method 2: Z-score based (|z| > 3)
                    if col_data.std() > 0:
                        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                        zscore_outliers = col_data[z_scores > 3]
                        feature_outliers['zscore_outliers'] = zscore_outliers.index.tolist()
                    
                    # Method 3: Modified Z-score (using median)
                    median = col_data.median()
                    mad = np.median(np.abs(col_data - median))
                    
                    if mad > 0:
                        modified_z_scores = 0.6745 * (col_data - median) / mad
                        mod_zscore_outliers = col_data[np.abs(modified_z_scores) > 3.5]
                        feature_outliers['modified_zscore_outliers'] = mod_zscore_outliers.index.tolist()
                    
                    # Combine all methods
                    all_outlier_indices = set()
                    all_outlier_indices.update(feature_outliers['iqr_outliers'])
                    all_outlier_indices.update(feature_outliers['zscore_outliers'])
                    all_outlier_indices.update(feature_outliers['modified_zscore_outliers'])
                    
                    feature_outliers['combined_outliers'] = list(all_outlier_indices)
                    feature_outliers['outlier_count'] = len(all_outlier_indices)
                    feature_outliers['outlier_percentage'] = len(all_outlier_indices) / len(col_data) if len(col_data) > 0 else 0
                    
                    total_outliers += len(all_outlier_indices)
                    
                except Exception as col_error:
                    feature_outliers['error'] = f'Analysis failed: {str(col_error)}'
                
                outlier_results['feature_outliers'][col] = feature_outliers
            
            # Overall outlier ratio
            outlier_results['overall_outlier_ratio'] = total_outliers / (total_observations * len(numeric_X.columns)) if (total_observations * len(numeric_X.columns)) > 0 else 0
            outlier_results['methods_used'] = ['IQR', 'Z-Score', 'Modified Z-Score']
            
            return outlier_results
            
        except Exception as e:
            return {'error': f'Outlier detection failed: {str(e)}'}
    
    def _analyze_feature_distributions(self, X: pd.DataFrame) -> Dict:
        """Analyze feature distributions for normality and skewness"""
        try:
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for distribution analysis'}
            
            distribution_results = {
                'feature_distributions': {},
                'overall_normality_score': 0.0,
                'highly_skewed_features': []
            }
            
            normality_scores = []
            
            for col in numeric_X.columns:
                try:
                    col_data = numeric_X[col].dropna()
                    if len(col_data) < 3:
                        continue
                    
                    feature_dist = {
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'normality_tests': {}
                    }
                    
                    # Normality tests
                    if self.scipy_available and len(col_data) >= 3:
                        try:
                            # Shapiro-Wilk test (for small samples)
                            if len(col_data) <= 5000:
                                shapiro_stat, shapiro_p = shapiro(col_data)
                                feature_dist['normality_tests']['shapiro_wilk'] = {
                                    'statistic': float(shapiro_stat),
                                    'p_value': float(shapiro_p),
                                    'is_normal': shapiro_p > 0.05
                                }
                                normality_scores.append(shapiro_p)
                        except:
                            pass
                    
                    # Simple normality assessment based on skewness and kurtosis
                    skew_abs = abs(feature_dist['skewness'])
                    kurt_abs = abs(feature_dist['kurtosis'])
                    
                    # Rough normality score based on skewness and kurtosis
                    normality_score = max(0, 1 - (skew_abs / 3 + kurt_abs / 7))
                    feature_dist['simple_normality_score'] = float(normality_score)
                    normality_scores.append(normality_score)
                    
                    # Check for high skewness
                    if skew_abs > 2:
                        distribution_results['highly_skewed_features'].append({
                            'feature': col,
                            'skewness': float(feature_dist['skewness']),
                            'severity': 'HIGH' if skew_abs > 3 else 'MEDIUM'
                        })
                    
                    distribution_results['feature_distributions'][col] = feature_dist
                    
                except Exception as col_error:
                    distribution_results['feature_distributions'][col] = {
                        'error': f'Distribution analysis failed: {str(col_error)}'
                    }
            
            # Overall normality score
            if normality_scores:
                distribution_results['overall_normality_score'] = float(np.mean(normality_scores))
            
            return distribution_results
            
        except Exception as e:
            return {'error': f'Distribution analysis failed: {str(e)}'}
    
    def _compute_signal_to_noise_ratio(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Compute signal-to-noise ratio for features"""
        try:
            if not self.sklearn_available:
                return self._compute_signal_to_noise_ratio_simplified(X, y)
            
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for SNR analysis'}
            
            # Use mutual information as signal measure
            mi_scores = mutual_info_classif(numeric_X, y, random_state=42)
            
            snr_results = {
                'feature_snr': {},
                'overall_snr': 0.0,
                'low_snr_features': []
            }
            
            snr_values = []
            
            for i, col in enumerate(numeric_X.columns):
                try:
                    col_data = numeric_X[col].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Signal: mutual information with target
                    signal = mi_scores[i]
                    
                    # Noise: coefficient of variation
                    if col_data.std() > 0 and col_data.mean() != 0:
                        noise = col_data.std() / abs(col_data.mean())
                    else:
                        noise = 1.0
                    
                    # SNR calculation
                    snr = signal / max(noise, 0.001)  # Prevent division by zero
                    
                    snr_results['feature_snr'][col] = {
                        'signal': float(signal),
                        'noise': float(noise),
                        'snr': float(snr)
                    }
                    
                    snr_values.append(snr)
                    
                    # Check for low SNR
                    min_snr_threshold = self.config.get('min_signal_noise_ratio', 3.0)
                    if snr < min_snr_threshold:
                        snr_results['low_snr_features'].append({
                            'feature': col,
                            'snr': float(snr),
                            'severity': 'HIGH' if snr < min_snr_threshold * 0.5 else 'MEDIUM'
                        })
                
                except Exception as col_error:
                    snr_results['feature_snr'][col] = {
                        'error': f'SNR computation failed: {str(col_error)}'
                    }
            
            # Overall SNR
            if snr_values:
                snr_results['overall_snr'] = float(np.mean(snr_values))
            
            return snr_results
            
        except Exception as e:
            return {'error': f'SNR computation failed: {str(e)}'}
    
    def _compute_signal_to_noise_ratio_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simplified SNR computation without sklearn"""
        try:
            numeric_X = X.select_dtypes(include=[np.number])
            
            snr_results = {
                'feature_snr': {},
                'overall_snr': 0.0,
                'low_snr_features': [],
                'method': 'simplified_correlation'
            }
            
            snr_values = []
            
            for col in numeric_X.columns:
                try:
                    col_data = numeric_X[col].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Signal: correlation with target
                    signal = abs(col_data.corr(y.loc[col_data.index]))
                    
                    if np.isnan(signal):
                        signal = 0.0
                    
                    # Noise: coefficient of variation
                    if col_data.std() > 0 and col_data.mean() != 0:
                        noise = col_data.std() / abs(col_data.mean())
                    else:
                        noise = 1.0
                    
                    # SNR calculation
                    snr = signal / max(noise, 0.001)
                    
                    snr_results['feature_snr'][col] = {
                        'signal': float(signal),
                        'noise': float(noise),
                        'snr': float(snr)
                    }
                    
                    snr_values.append(snr)
                    
                    # Check for low SNR
                    if snr < 1.0:  # Lower threshold for simplified method
                        snr_results['low_snr_features'].append({
                            'feature': col,
                            'snr': float(snr),
                            'severity': 'HIGH' if snr < 0.5 else 'MEDIUM'
                        })
                
                except Exception as col_error:
                    snr_results['feature_snr'][col] = {
                        'error': f'SNR computation failed: {str(col_error)}'
                    }
            
            # Overall SNR
            if snr_values:
                snr_results['overall_snr'] = float(np.mean(snr_values))
            
            return snr_results
            
        except Exception as e:
            return {'error': f'Simplified SNR computation failed: {str(e)}'}
    
    def _analyze_feature_relevance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Analyze feature relevance to target variable"""
        try:
            if not self.sklearn_available:
                return self._analyze_feature_relevance_simplified(X, y)
            
            numeric_X = X.select_dtypes(include=[np.number])
            
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for relevance analysis'}
            
            # Mutual information analysis
            mi_scores = mutual_info_classif(numeric_X, y, random_state=42)
            
            relevance_results = {
                'feature_relevance': {},
                'low_relevance_features': [],
                'overall_relevance_score': 0.0
            }
            
            for i, col in enumerate(numeric_X.columns):
                relevance_score = float(mi_scores[i])
                
                relevance_results['feature_relevance'][col] = {
                    'mutual_info_score': relevance_score,
                    'relevance_level': (
                        'HIGH' if relevance_score > 0.1 else
                        'MEDIUM' if relevance_score > 0.05 else
                        'LOW'
                    )
                }
                
                # Check for low relevance
                min_relevance = self.config.get('min_feature_importance', 0.02)
                if relevance_score < min_relevance:
                    relevance_results['low_relevance_features'].append({
                        'feature': col,
                        'relevance_score': relevance_score,
                        'severity': 'HIGH' if relevance_score < min_relevance * 0.5 else 'MEDIUM'
                    })
            
            # Overall relevance score
            relevance_results['overall_relevance_score'] = float(np.mean(mi_scores))
            
            return relevance_results
            
        except Exception as e:
            return {'error': f'Feature relevance analysis failed: {str(e)}'}
    
    def _analyze_feature_relevance_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simplified feature relevance analysis using correlation"""
        try:
            numeric_X = X.select_dtypes(include=[np.number])
            
            relevance_results = {
                'feature_relevance': {},
                'low_relevance_features': [],
                'overall_relevance_score': 0.0,
                'method': 'simplified_correlation'
            }
            
            correlations = []
            
            for col in numeric_X.columns:
                try:
                    corr = abs(numeric_X[col].corr(y))
                    
                    if np.isnan(corr):
                        corr = 0.0
                    
                    relevance_results['feature_relevance'][col] = {
                        'correlation_score': float(corr),
                        'relevance_level': (
                            'HIGH' if corr > 0.3 else
                            'MEDIUM' if corr > 0.1 else
                            'LOW'
                        )
                    }
                    
                    correlations.append(corr)
                    
                    # Check for low relevance
                    if corr < 0.05:
                        relevance_results['low_relevance_features'].append({
                            'feature': col,
                            'correlation_score': float(corr),
                            'severity': 'HIGH' if corr < 0.02 else 'MEDIUM'
                        })
                
                except:
                    relevance_results['feature_relevance'][col] = {
                        'correlation_score': 0.0,
                        'relevance_level': 'LOW'
                    }
                    correlations.append(0.0)
            
            # Overall relevance score
            if correlations:
                relevance_results['overall_relevance_score'] = float(np.mean(correlations))
            
            return relevance_results
            
        except Exception as e:
            return {'error': f'Simplified feature relevance analysis failed: {str(e)}'}
    
    def _compute_noise_level(self, noise_results: Dict) -> float:
        """Compute overall noise level score"""
        try:
            noise_components = []
            
            # Missing values component (25% weight)
            missing_data = noise_results.get('missing_values', {})
            missing_ratio = missing_data.get('total_missing_ratio', 0.0)
            noise_components.append(min(missing_ratio * 2, 1.0))
            
            # Outliers component (25% weight)
            outlier_data = noise_results.get('outliers', {})
            outlier_ratio = outlier_data.get('overall_outlier_ratio', 0.0)
            noise_components.append(min(outlier_ratio * 5, 1.0))
            
            # Distribution quality component (25% weight)
            dist_data = noise_results.get('distributions', {})
            normality_score = dist_data.get('overall_normality_score', 1.0)
            noise_components.append(1.0 - normality_score)
            
            # Signal-to-noise ratio component (25% weight)
            snr_data = noise_results.get('signal_to_noise', {})
            overall_snr = snr_data.get('overall_snr', 5.0)
            min_snr = self.config.get('min_signal_noise_ratio', 3.0)
            snr_component = max(0, 1.0 - (overall_snr / min_snr))
            noise_components.append(snr_component)
            
            # Weighted average
            return float(np.mean(noise_components))
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.warning(f"⚠️ Noise level computation failed: {str(e)}", "NoiseAnalyzer")
            else:
                self.logger.warning(f"⚠️ Noise level computation failed: {str(e)}")
            return 0.5  # Default moderate noise level
    
    def _compute_data_quality_score(self, noise_results: Dict) -> float:
        """Compute overall data quality score (inverse of noise level)"""
        try:
            noise_level = noise_results.get('noise_level', 0.5)
            
            # Base quality score (inverse of noise)
            base_quality = 1.0 - noise_level
            
            # Additional quality factors
            quality_factors = []
            
            # Feature relevance factor
            relevance_data = noise_results.get('feature_relevance', {})
            overall_relevance = relevance_data.get('overall_relevance_score', 0.5)
            quality_factors.append(overall_relevance)
            
            # Data completeness factor
            missing_data = noise_results.get('missing_values', {})
            completeness = 1.0 - missing_data.get('total_missing_ratio', 0.0)
            quality_factors.append(completeness)
            
            # Weighted combination
            if quality_factors:
                adjusted_quality = 0.6 * base_quality + 0.4 * np.mean(quality_factors)
                return float(min(max(adjusted_quality, 0.0), 1.0))
            else:
                return float(base_quality)
                
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.warning(f"⚠️ Data quality score computation failed: {str(e)}", "NoiseAnalyzer")
            else:
                self.logger.warning(f"⚠️ Data quality score computation failed: {str(e)}")
            return 0.5  # Default moderate quality
    
    def _generate_quality_recommendations(self, noise_results: Dict) -> List[str]:
        """Generate data quality improvement recommendations"""
        try:
            recommendations = []
            
            # Missing values recommendations
            missing_data = noise_results.get('missing_values', {})
            if missing_data.get('total_missing_ratio', 0) > 0.1:
                recommendations.append("Address missing values: Consider imputation or feature engineering")
            
            # Outlier recommendations
            outlier_data = noise_results.get('outliers', {})
            if outlier_data.get('overall_outlier_ratio', 0) > 0.05:
                recommendations.append("Handle outliers: Consider robust scaling or outlier removal")
            
            # Distribution recommendations
            dist_data = noise_results.get('distributions', {})
            if len(dist_data.get('highly_skewed_features', [])) > 0:
                recommendations.append("Transform skewed features: Consider log transformation or Box-Cox")
            
            # SNR recommendations
            snr_data = noise_results.get('signal_to_noise', {})
            if len(snr_data.get('low_snr_features', [])) > 0:
                recommendations.append("Improve signal-to-noise ratio: Consider feature selection or noise reduction")
            
            # Relevance recommendations
            relevance_data = noise_results.get('feature_relevance', {})
            if len(relevance_data.get('low_relevance_features', [])) > 0:
                recommendations.append("Remove low-relevance features: Consider feature selection methods")
            
            return recommendations
            
        except Exception as e:
            return [f"Could not generate recommendations: {str(e)}"]
