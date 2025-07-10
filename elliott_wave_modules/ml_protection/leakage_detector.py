#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DATA LEAKAGE DETECTOR MODULE
Enterprise-grade data leakage detection system

Detection Methods:
- Perfect Correlation Detection
- Temporal Leakage Analysis
- Future Information Detection
- Statistical Leakage Tests
- Feature Naming Pattern Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime

from core.unified_enterprise_logger import get_unified_logger

# Import ML libraries with error handling
try:
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available, using simplified leakage detection")


class DataLeakageDetector:
    """🔍 Enterprise Data Leakage Detection System"""
    
    def __init__(self, config: Dict = None, logger=None):
        self.config = config or {}
        self.sklearn_available = SKLEARN_AVAILABLE
        
        # Initialize logging
        self.logger = get_unified_logger("DataLeakageDetector")
    
    def update_config(self, new_config: Dict):
        """Update detector configuration"""
        self.config.update(new_config)
    
    def detect_data_leakage(self, X: pd.DataFrame, y: pd.Series, datetime_col: str = None) -> Dict[str, Any]:
        """
        Comprehensive data leakage detection
        
        Args:
            X: Feature matrix
            y: Target vector
            datetime_col: Name of datetime column for temporal analysis
            
        Returns:
            Data leakage analysis results
        """
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
                    try:
                        corr = abs(X[col].corr(y))
                        if not np.isnan(corr) and corr > 0.95:  # Suspiciously high correlation
                            correlations.append({
                                'feature': col,
                                'correlation': corr,
                                'risk_level': 'HIGH' if corr > 0.98 else 'MEDIUM'
                            })
                    except:
                        continue
            
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
            
            # 5. Feature Naming Pattern Analysis
            naming_analysis = self._analyze_feature_naming_patterns(X)
            leakage_results['naming_patterns'] = naming_analysis
            
            # Overall leakage assessment
            leakage_score = self._compute_leakage_score(leakage_results)
            leakage_results['leakage_score'] = leakage_score
            leakage_results['leakage_detected'] = leakage_score > 0.3
            leakage_results['status'] = 'DETECTED' if leakage_results['leakage_detected'] else 'CLEAN'
            
            # Collect all suspicious features
            suspicious_features = []
            suspicious_features.extend([c['feature'] for c in correlations])
            suspicious_features.extend(future_leakage['suspicious_features'])
            if 'suspicious_features' in statistical_leakage:
                suspicious_features.extend(statistical_leakage['suspicious_features'])
            if naming_analysis.get('suspicious_features'):
                suspicious_features.extend(naming_analysis['suspicious_features'])
            
            leakage_results['suspicious_features'] = list(set(suspicious_features))
            
            return leakage_results
            
        except Exception as e:
            error_msg = f"❌ Data leakage detection failed: {str(e)}"
                self.logger.error(error_msg, component="LeakageDetector")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _check_temporal_leakage(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict:
        """Check for temporal leakage patterns"""
        try:
            # Check if datetime is properly sorted
            is_sorted = X[datetime_col].is_monotonic_increasing
            
            # Check for future information in column names
            future_refs = []
            for col in X.columns:
                if any(keyword in col.lower() for keyword in ['future', 'next', 'forward', 'lag_-']):
                    future_refs.append(col)
            
            # Check for time-based patterns that might indicate leakage
            time_gaps = []
            if is_sorted and len(X) > 1:
                time_diff = X[datetime_col].diff()
                # Look for unusual time gaps
                median_gap = time_diff.median()
                large_gaps = time_diff[time_diff > median_gap * 10].index.tolist()
                time_gaps = large_gaps
            
            return {
                'detected': not is_sorted or len(future_refs) > 0 or len(time_gaps) > len(X) * 0.1,
                'datetime_sorted': is_sorted,
                'future_references': future_refs,
                'unusual_time_gaps': time_gaps,
                'details': f"Temporal ordering: {'OK' if is_sorted else 'VIOLATION'}"
            }
        except Exception as e:
            return {
                'detected': False, 
                'error': f'Could not analyze temporal leakage: {str(e)}'
            }
    
    def _detect_future_information_leakage(self, X: pd.DataFrame) -> Dict:
        """Detect future information leakage in feature names"""
        suspicious_features = []
        
        # Keywords that might indicate future information
        future_keywords = [
            'future', 'next', 'forward', 'ahead', 'tomorrow', 'lag_-',
            'pred', 'forecast', 'projected', 'expected', 'target',
            'outcome', 'result', 'label', 'y_', 'response'
        ]
        
        # Statistical patterns that might indicate target leakage
        statistical_keywords = [
            'mean_target', 'target_mean', 'target_encoded', 'label_encoded',
            'outcome_encoded', 'response_encoded'
        ]
        
        for col in X.columns:
            col_lower = col.lower()
            
            # Check for obvious future information
            if any(keyword in col_lower for keyword in future_keywords):
                suspicious_features.append({
                    'feature': col,
                    'reason': 'future_keyword',
                    'risk_level': 'HIGH'
                })
            
            # Check for statistical leakage patterns
            elif any(keyword in col_lower for keyword in statistical_keywords):
                suspicious_features.append({
                    'feature': col,
                    'reason': 'statistical_leakage_pattern',
                    'risk_level': 'MEDIUM'
                })
        
        return {
            'detected': len(suspicious_features) > 0,
            'suspicious_features': [f['feature'] for f in suspicious_features],
            'detailed_analysis': suspicious_features
        }
    
    def _statistical_leakage_test(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Statistical test for data leakage using mutual information"""
        try:
            if not self.sklearn_available:
                return self._statistical_leakage_test_simplified(X, y)
            
            # Use mutual information to detect suspiciously high relationships
            numeric_X = X.select_dtypes(include=[np.number])
            if len(numeric_X.columns) == 0:
                return {'error': 'No numeric features for statistical analysis'}
            
            mi_scores = mutual_info_classif(numeric_X, y, random_state=42)
            
            suspicious_features = []
            feature_scores = []
            
            for i, score in enumerate(mi_scores):
                feature_name = numeric_X.columns[i]
                feature_scores.append({
                    'feature': feature_name,
                    'mutual_info_score': float(score)
                })
                
                if score > 0.8:  # Suspiciously high mutual information
                    suspicious_features.append(feature_name)
            
            return {
                'mutual_info_scores': [float(s) for s in mi_scores],
                'feature_scores': feature_scores,
                'suspicious_features': suspicious_features,
                'max_mi_score': float(mi_scores.max()) if len(mi_scores) > 0 else 0,
                'method': 'sklearn_mutual_info'
            }
            
        except Exception as e:
            error_msg = f"Statistical leakage test failed: {str(e)}"
                self.logger.warning(f"⚠️ {error_msg}", component="LeakageDetector")
            
            # Fallback to simplified method
            return self._statistical_leakage_test_simplified(X, y)
    
    def _statistical_leakage_test_simplified(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Simplified statistical leakage test without sklearn"""
        try:
            suspicious_features = []
            feature_scores = []
            
            numeric_X = X.select_dtypes(include=[np.number])
            
            for col in numeric_X.columns:
                try:
                    # Calculate correlation instead of mutual information
                    corr = abs(numeric_X[col].corr(y))
                    
                    if not np.isnan(corr):
                        feature_scores.append({
                            'feature': col,
                            'correlation_score': float(corr)
                        })
                        
                        if corr > 0.9:  # Very high correlation
                            suspicious_features.append(col)
                            
                except:
                    continue
            
            max_score = max([s['correlation_score'] for s in feature_scores]) if feature_scores else 0
            
            return {
                'feature_scores': feature_scores,
                'suspicious_features': suspicious_features,
                'max_correlation_score': float(max_score),
                'method': 'simplified_correlation'
            }
            
        except Exception as e:
            return {
                'error': f'Simplified statistical leakage test failed: {str(e)}',
                'method': 'simplified_failed'
            }
    
    def _analyze_feature_naming_patterns(self, X: pd.DataFrame) -> Dict:
        """Analyze feature naming patterns for potential leakage indicators"""
        try:
            suspicious_patterns = []
            warning_patterns = []
            
            # High-risk patterns
            high_risk_patterns = [
                r'.*target.*', r'.*label.*', r'.*outcome.*', r'.*response.*',
                r'.*future.*', r'.*next.*', r'.*forward.*', r'.*ahead.*',
                r'.*pred.*', r'.*forecast.*', r'.*expected.*'
            ]
            
            # Medium-risk patterns
            medium_risk_patterns = [
                r'.*encoded.*target.*', r'.*mean.*target.*', r'.*target.*mean.*',
                r'.*leak.*', r'.*cheat.*', r'.*solution.*'
            ]
            
            import re
            
            for col in X.columns:
                col_lower = col.lower()
                
                # Check high-risk patterns
                for pattern in high_risk_patterns:
                    if re.match(pattern, col_lower):
                        suspicious_patterns.append({
                            'feature': col,
                            'pattern': pattern,
                            'risk_level': 'HIGH'
                        })
                        break
                
                # Check medium-risk patterns
                for pattern in medium_risk_patterns:
                    if re.match(pattern, col_lower):
                        warning_patterns.append({
                            'feature': col,
                            'pattern': pattern,
                            'risk_level': 'MEDIUM'
                        })
                        break
            
            all_suspicious = suspicious_patterns + warning_patterns
            
            return {
                'high_risk_patterns': suspicious_patterns,
                'medium_risk_patterns': warning_patterns,
                'suspicious_features': [p['feature'] for p in all_suspicious],
                'pattern_analysis_complete': True
            }
            
        except Exception as e:
            return {
                'error': f'Feature naming pattern analysis failed: {str(e)}',
                'pattern_analysis_complete': False
            }
    
    def _compute_leakage_score(self, leakage_results: Dict) -> float:
        """Compute overall leakage score"""
        try:
            score = 0.0
            
            # High correlation penalty (40% weight)
            if 'suspicious_correlations' in leakage_results:
                high_corr_count = len([c for c in leakage_results['suspicious_correlations'] 
                                     if c.get('correlation', 0) > 0.98])
                score += min(high_corr_count * 0.2, 0.4)
            
            # Temporal leakage penalty (25% weight)
            if leakage_results.get('temporal_leakage', False):
                score += 0.25
            
            # Future feature penalty (20% weight)
            if leakage_results.get('future_feature_leakage', False):
                future_count = len(leakage_results.get('future_features', []))
                score += min(future_count * 0.05, 0.2)
            
            # Statistical leakage penalty (15% weight)
            stat_leak = leakage_results.get('statistical_leakage', {})
            if stat_leak.get('method') == 'sklearn_mutual_info':
                max_mi = stat_leak.get('max_mi_score', 0)
                if max_mi > 0.8:
                    score += min((max_mi - 0.8) * 0.75, 0.15)
            elif stat_leak.get('method') == 'simplified_correlation':
                max_corr = stat_leak.get('max_correlation_score', 0)
                if max_corr > 0.9:
                    score += min((max_corr - 0.9) * 1.5, 0.15)
            
            # Naming pattern penalty (remaining weight)
            naming_patterns = leakage_results.get('naming_patterns', {})
            high_risk_count = len(naming_patterns.get('high_risk_patterns', []))
            medium_risk_count = len(naming_patterns.get('medium_risk_patterns', []))
            
            pattern_score = min(high_risk_count * 0.05 + medium_risk_count * 0.02, 0.1)
            score += pattern_score
            
            return min(score, 1.0)
            
        except Exception as e:
                self.logger.warning(f"⚠️ Leakage score computation failed: {str(e)}", component="LeakageDetector")
            return 0.0
