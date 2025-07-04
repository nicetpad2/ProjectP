#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TIME SERIES VALIDATOR MODULE
Enterprise-grade time series integrity validation

Validation Methods:
- Temporal Ordering Validation
- Time Gap Analysis
- Seasonality Detection
- Trend Analysis
- Time Series Specific Leakage Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from datetime import datetime, timedelta

# Import advanced logging system
try:
    from core.advanced_terminal_logger import get_terminal_logger, LogLevel
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    import logging


class TimeSeriesValidator:
    """📅 Enterprise Time Series Integrity Validation System"""
    
    def __init__(self, config: Dict = None, logger=None):
        self.config = config or {}
        
        # Initialize logging
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
        else:
            self.logger = logger or logging.getLogger(__name__)
    
    def update_config(self, new_config: Dict):
        """Update validator configuration"""
        self.config.update(new_config)
    
    def validate_timeseries_integrity(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict[str, Any]:
        """
        Comprehensive time series integrity validation
        
        Args:
            X: Feature matrix
            y: Target vector
            datetime_col: Name of datetime column
            
        Returns:
            Time series validation results
        """
        try:
            if datetime_col not in X.columns:
                return {
                    'status': 'ERROR',
                    'error': f'Datetime column "{datetime_col}" not found in data'
                }
            
            validation_results = {
                'status': 'ANALYZING',
                'temporal_issues': False,
                'integrity_score': 0.0,
                'ordering_analysis': {},
                'gap_analysis': {},
                'seasonality_analysis': {},
                'trend_analysis': {},
                'leakage_analysis': {},
                'recommendations': []
            }
            
            # 1. Temporal Ordering Validation
            ordering_analysis = self._validate_temporal_ordering(X, datetime_col)
            validation_results['ordering_analysis'] = ordering_analysis
            
            # 2. Time Gap Analysis
            gap_analysis = self._analyze_time_gaps(X, datetime_col)
            validation_results['gap_analysis'] = gap_analysis
            
            # 3. Seasonality Detection
            seasonality_analysis = self._detect_seasonality(X, y, datetime_col)
            validation_results['seasonality_analysis'] = seasonality_analysis
            
            # 4. Trend Analysis
            trend_analysis = self._analyze_trends(X, y, datetime_col)
            validation_results['trend_analysis'] = trend_analysis
            
            # 5. Time Series Specific Leakage Detection
            leakage_analysis = self._detect_timeseries_leakage(X, y, datetime_col)
            validation_results['leakage_analysis'] = leakage_analysis
            
            # Overall integrity assessment
            integrity_assessment = self._compute_integrity_assessment(validation_results)
            validation_results['integrity_score'] = integrity_assessment['score']
            validation_results['temporal_issues'] = integrity_assessment['issues_detected']
            validation_results['status'] = 'ISSUES_DETECTED' if integrity_assessment['issues_detected'] else 'VALID'
            
            # Generate recommendations
            recommendations = self._generate_timeseries_recommendations(validation_results)
            validation_results['recommendations'] = recommendations
            
            return validation_results
            
        except Exception as e:
            error_msg = f"❌ Time series validation failed: {str(e)}"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(error_msg, "TimeSeriesValidator")
            else:
                self.logger.error(error_msg)
            return {'status': 'ERROR', 'error': str(e)}
    
    def _validate_temporal_ordering(self, X: pd.DataFrame, datetime_col: str) -> Dict:
        """Validate temporal ordering of the dataset"""
        try:
            datetime_series = pd.to_datetime(X[datetime_col])
            
            ordering_results = {
                'is_sorted': False,
                'sort_violations': 0,
                'duplicate_timestamps': 0,
                'ordering_score': 0.0,
                'first_timestamp': None,
                'last_timestamp': None,
                'total_timespan': None
            }
            
            # Check if data is sorted
            is_sorted = datetime_series.is_monotonic_increasing
            ordering_results['is_sorted'] = is_sorted
            
            # Count sort violations
            if not is_sorted:
                sort_violations = 0
                for i in range(1, len(datetime_series)):
                    if datetime_series.iloc[i] < datetime_series.iloc[i-1]:
                        sort_violations += 1
                ordering_results['sort_violations'] = sort_violations
            
            # Check for duplicate timestamps
            duplicate_count = datetime_series.duplicated().sum()
            ordering_results['duplicate_timestamps'] = int(duplicate_count)
            
            # Time span analysis
            if len(datetime_series) > 0:
                first_ts = datetime_series.min()
                last_ts = datetime_series.max()
                
                ordering_results['first_timestamp'] = first_ts.isoformat() if pd.notna(first_ts) else None
                ordering_results['last_timestamp'] = last_ts.isoformat() if pd.notna(last_ts) else None
                
                if pd.notna(first_ts) and pd.notna(last_ts):
                    timespan = last_ts - first_ts
                    ordering_results['total_timespan'] = str(timespan)
            
            # Compute ordering score
            total_records = len(datetime_series)
            if total_records > 0:
                violation_ratio = ordering_results['sort_violations'] / total_records
                duplicate_ratio = ordering_results['duplicate_timestamps'] / total_records
                ordering_score = max(0.0, 1.0 - (violation_ratio + duplicate_ratio))
                ordering_results['ordering_score'] = float(ordering_score)
            
            return ordering_results
            
        except Exception as e:
            return {'error': f'Temporal ordering validation failed: {str(e)}'}
    
    def _analyze_time_gaps(self, X: pd.DataFrame, datetime_col: str) -> Dict:
        """Analyze time gaps in the dataset"""
        try:
            datetime_series = pd.to_datetime(X[datetime_col]).sort_values()
            
            gap_results = {
                'time_differences': [],
                'gap_statistics': {},
                'large_gaps': [],
                'gap_regularity_score': 0.0,
                'expected_frequency': None
            }
            
            if len(datetime_series) < 2:
                return {'error': 'Insufficient data for gap analysis'}
            
            # Calculate time differences
            time_diffs = datetime_series.diff().dropna()
            
            # Convert to seconds for analysis
            diff_seconds = time_diffs.dt.total_seconds()
            
            gap_results['gap_statistics'] = {
                'mean_gap_seconds': float(diff_seconds.mean()),
                'median_gap_seconds': float(diff_seconds.median()),
                'std_gap_seconds': float(diff_seconds.std()),
                'min_gap_seconds': float(diff_seconds.min()),
                'max_gap_seconds': float(diff_seconds.max()),
                'total_gaps': len(diff_seconds)
            }
            
            # Detect large gaps (outliers)
            if diff_seconds.std() > 0:
                # Use median + 3*MAD as threshold for large gaps
                median_gap = diff_seconds.median()
                mad = np.median(np.abs(diff_seconds - median_gap))
                
                if mad > 0:
                    threshold = median_gap + 3 * mad * 1.4826  # 1.4826 is scaling factor for MAD
                    large_gap_indices = diff_seconds[diff_seconds > threshold].index
                    
                    large_gaps = []
                    for idx in large_gap_indices:
                        gap_info = {
                            'index': int(idx),
                            'gap_seconds': float(diff_seconds.loc[idx]),
                            'gap_duration': str(timedelta(seconds=diff_seconds.loc[idx])),
                            'before_timestamp': datetime_series.loc[idx-1].isoformat(),
                            'after_timestamp': datetime_series.loc[idx].isoformat()
                        }
                        large_gaps.append(gap_info)
                    
                    gap_results['large_gaps'] = large_gaps
            
            # Assess gap regularity
            if len(diff_seconds) > 0:
                cv = diff_seconds.std() / diff_seconds.mean() if diff_seconds.mean() > 0 else float('inf')
                regularity_score = max(0.0, 1.0 - min(cv / 2, 1.0))  # Lower CV = higher regularity
                gap_results['gap_regularity_score'] = float(regularity_score)
            
            # Estimate expected frequency
            if len(diff_seconds) > 0:
                median_seconds = diff_seconds.median()
                
                if median_seconds < 3600:  # Less than 1 hour
                    if median_seconds < 300:  # Less than 5 minutes
                        freq_estimate = "High frequency (< 5 min)"
                    else:
                        freq_estimate = "Minute/Hourly frequency"
                elif median_seconds < 86400:  # Less than 1 day
                    freq_estimate = "Hourly frequency"
                elif median_seconds < 604800:  # Less than 1 week
                    freq_estimate = "Daily frequency"
                else:
                    freq_estimate = "Weekly or lower frequency"
                
                gap_results['expected_frequency'] = freq_estimate
            
            return gap_results
            
        except Exception as e:
            return {'error': f'Time gap analysis failed: {str(e)}'}
    
    def _detect_seasonality(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict:
        """Detect seasonal patterns in the data"""
        try:
            datetime_series = pd.to_datetime(X[datetime_col])
            
            seasonality_results = {
                'seasonal_patterns': {},
                'detected_seasons': [],
                'seasonality_strength': 0.0,
                'dominant_period': None
            }
            
            # Create time-based features
            df_temp = pd.DataFrame({
                'datetime': datetime_series,
                'target': y
            }).sort_values('datetime')
            
            df_temp['hour'] = df_temp['datetime'].dt.hour
            df_temp['day_of_week'] = df_temp['datetime'].dt.dayofweek
            df_temp['day_of_month'] = df_temp['datetime'].dt.day
            df_temp['month'] = df_temp['datetime'].dt.month
            df_temp['quarter'] = df_temp['datetime'].dt.quarter
            
            # Analyze patterns for different time periods
            seasonal_patterns = {}
            
            # Hourly pattern
            if df_temp['hour'].nunique() > 1:
                hourly_pattern = df_temp.groupby('hour')['target'].mean()
                hourly_std = hourly_pattern.std()
                seasonal_patterns['hourly'] = {
                    'variation': float(hourly_std),
                    'pattern_strength': float(hourly_std / max(hourly_pattern.mean(), 1e-8))
                }
            
            # Daily pattern
            if df_temp['day_of_week'].nunique() > 1:
                daily_pattern = df_temp.groupby('day_of_week')['target'].mean()
                daily_std = daily_pattern.std()
                seasonal_patterns['daily'] = {
                    'variation': float(daily_std),
                    'pattern_strength': float(daily_std / max(daily_pattern.mean(), 1e-8))
                }
            
            # Monthly pattern
            if df_temp['month'].nunique() > 1:
                monthly_pattern = df_temp.groupby('month')['target'].mean()
                monthly_std = monthly_pattern.std()
                seasonal_patterns['monthly'] = {
                    'variation': float(monthly_std),
                    'pattern_strength': float(monthly_std / max(monthly_pattern.mean(), 1e-8))
                }
            
            # Quarterly pattern
            if df_temp['quarter'].nunique() > 1:
                quarterly_pattern = df_temp.groupby('quarter')['target'].mean()
                quarterly_std = quarterly_pattern.std()
                seasonal_patterns['quarterly'] = {
                    'variation': float(quarterly_std),
                    'pattern_strength': float(quarterly_std / max(quarterly_pattern.mean(), 1e-8))
                }
            
            seasonality_results['seasonal_patterns'] = seasonal_patterns
            
            # Determine detected seasons
            detected_seasons = []
            threshold = 0.1  # Pattern strength threshold
            
            for period, pattern in seasonal_patterns.items():
                if pattern['pattern_strength'] > threshold:
                    detected_seasons.append({
                        'period': period,
                        'strength': pattern['pattern_strength'],
                        'significance': 'HIGH' if pattern['pattern_strength'] > threshold * 2 else 'MEDIUM'
                    })
            
            seasonality_results['detected_seasons'] = detected_seasons
            
            # Overall seasonality strength
            if seasonal_patterns:
                all_strengths = [p['pattern_strength'] for p in seasonal_patterns.values()]
                seasonality_results['seasonality_strength'] = float(max(all_strengths))
                
                # Dominant period
                max_strength_period = max(seasonal_patterns.keys(), 
                                        key=lambda k: seasonal_patterns[k]['pattern_strength'])
                seasonality_results['dominant_period'] = max_strength_period
            
            return seasonality_results
            
        except Exception as e:
            return {'error': f'Seasonality detection failed: {str(e)}'}
    
    def _analyze_trends(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict:
        """Analyze trends in the target variable over time"""
        try:
            datetime_series = pd.to_datetime(X[datetime_col])
            
            trend_results = {
                'trend_direction': 'NONE',
                'trend_strength': 0.0,
                'trend_significance': 0.0,
                'trend_analysis': {},
                'change_points': []
            }
            
            # Create time-sorted data
            df_temp = pd.DataFrame({
                'datetime': datetime_series,
                'target': y
            }).sort_values('datetime').reset_index(drop=True)
            
            if len(df_temp) < 3:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Convert datetime to numeric for trend analysis
            df_temp['time_numeric'] = (df_temp['datetime'] - df_temp['datetime'].min()).dt.total_seconds()
            
            # Simple linear trend analysis
            if df_temp['time_numeric'].std() > 0:
                correlation = np.corrcoef(df_temp['time_numeric'], df_temp['target'])[0, 1]
                
                if not np.isnan(correlation):
                    trend_results['trend_strength'] = float(abs(correlation))
                    trend_results['trend_significance'] = float(correlation)
                    
                    if correlation > 0.1:
                        trend_results['trend_direction'] = 'INCREASING'
                    elif correlation < -0.1:
                        trend_results['trend_direction'] = 'DECREASING'
                    else:
                        trend_results['trend_direction'] = 'STABLE'
            
            # Moving average trend analysis
            if len(df_temp) > 10:
                window_size = min(len(df_temp) // 4, 50)
                df_temp['ma'] = df_temp['target'].rolling(window=window_size, center=True).mean()
                
                # Trend in moving average
                ma_clean = df_temp['ma'].dropna()
                if len(ma_clean) > 2:
                    ma_trend = np.polyfit(range(len(ma_clean)), ma_clean, 1)[0]
                    trend_results['trend_analysis']['moving_average_trend'] = float(ma_trend)
            
            # Simple change point detection (basic version)
            if len(df_temp) > 20:
                change_points = self._detect_simple_change_points(df_temp['target'])
                trend_results['change_points'] = change_points
            
            return trend_results
            
        except Exception as e:
            return {'error': f'Trend analysis failed: {str(e)}'}
    
    def _detect_simple_change_points(self, series: pd.Series) -> List[Dict]:
        """Simple change point detection using moving statistics"""
        try:
            change_points = []
            
            window_size = max(len(series) // 10, 5)
            
            # Calculate rolling mean and std
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()
            
            # Find points where rolling statistics change significantly
            mean_diff = rolling_mean.diff().abs()
            std_diff = rolling_std.diff().abs()
            
            # Thresholds for significant change
            mean_threshold = mean_diff.quantile(0.95) if len(mean_diff.dropna()) > 0 else 0
            std_threshold = std_diff.quantile(0.95) if len(std_diff.dropna()) > 0 else 0
            
            for i in range(window_size, len(series) - window_size):
                if (mean_diff.iloc[i] > mean_threshold or std_diff.iloc[i] > std_threshold):
                    change_points.append({
                        'index': int(i),
                        'mean_change': float(mean_diff.iloc[i]) if not pd.isna(mean_diff.iloc[i]) else 0,
                        'std_change': float(std_diff.iloc[i]) if not pd.isna(std_diff.iloc[i]) else 0,
                        'significance': 'HIGH' if (mean_diff.iloc[i] > mean_threshold * 1.5 or 
                                                 std_diff.iloc[i] > std_threshold * 1.5) else 'MEDIUM'
                    })
            
            return change_points
            
        except Exception as e:
            return []
    
    def _detect_timeseries_leakage(self, X: pd.DataFrame, y: pd.Series, datetime_col: str) -> Dict:
        """Detect time series specific data leakage"""
        try:
            leakage_results = {
                'future_looking_features': [],
                'temporal_leakage_score': 0.0,
                'suspicious_patterns': [],
                'leakage_detected': False
            }
            
            datetime_series = pd.to_datetime(X[datetime_col])
            
            # 1. Check for features that might contain future information
            future_keywords = [
                'future', 'next', 'forward', 'ahead', 'tomorrow',
                'lag_-', 'lead_', 'shift_-', 'predict', 'forecast'
            ]
            
            future_features = []
            for col in X.columns:
                if col != datetime_col:
                    col_lower = col.lower()
                    for keyword in future_keywords:
                        if keyword in col_lower:
                            future_features.append({
                                'feature': col,
                                'suspicious_keyword': keyword,
                                'risk_level': 'HIGH'
                            })
                            break
            
            leakage_results['future_looking_features'] = future_features
            
            # 2. Check for suspiciously perfect correlations with target
            numeric_features = X.select_dtypes(include=[np.number]).columns
            
            suspicious_patterns = []
            for col in numeric_features:
                if col != datetime_col:
                    try:
                        correlation = abs(X[col].corr(y))
                        if not np.isnan(correlation) and correlation > 0.95:
                            suspicious_patterns.append({
                                'feature': col,
                                'correlation': float(correlation),
                                'pattern_type': 'high_correlation',
                                'risk_level': 'HIGH' if correlation > 0.98 else 'MEDIUM'
                            })
                    except:
                        continue
            
            leakage_results['suspicious_patterns'] = suspicious_patterns
            
            # 3. Temporal ordering checks
            if not datetime_series.is_monotonic_increasing:
                leakage_results['suspicious_patterns'].append({
                    'feature': datetime_col,
                    'pattern_type': 'temporal_disorder',
                    'risk_level': 'HIGH',
                    'description': 'Data is not properly time-ordered'
                })
            
            # Compute temporal leakage score
            score = 0.0
            
            # Future features penalty
            score += min(len(future_features) * 0.3, 0.6)
            
            # High correlation penalty
            high_corr_count = len([p for p in suspicious_patterns 
                                 if p.get('pattern_type') == 'high_correlation' and p.get('correlation', 0) > 0.98])
            score += min(high_corr_count * 0.2, 0.4)
            
            # Temporal disorder penalty
            temporal_disorder = any(p.get('pattern_type') == 'temporal_disorder' for p in suspicious_patterns)
            if temporal_disorder:
                score += 0.3
            
            leakage_results['temporal_leakage_score'] = float(min(score, 1.0))
            leakage_results['leakage_detected'] = score > 0.3
            
            return leakage_results
            
        except Exception as e:
            return {'error': f'Time series leakage detection failed: {str(e)}'}
    
    def _compute_integrity_assessment(self, validation_results: Dict) -> Dict:
        """Compute overall time series integrity assessment"""
        try:
            assessment = {
                'score': 0.0,
                'issues_detected': False,
                'integrity_level': 'UNKNOWN',
                'issue_summary': []
            }
            
            integrity_components = []
            issues = []
            
            # Temporal ordering component (30% weight)
            ordering_data = validation_results.get('ordering_analysis', {})
            ordering_score = ordering_data.get('ordering_score', 0.5)
            integrity_components.append(('ordering', ordering_score, 0.3))
            
            if ordering_score < 0.8:
                issues.append("Temporal ordering issues detected")
            
            # Gap analysis component (20% weight)
            gap_data = validation_results.get('gap_analysis', {})
            gap_regularity = gap_data.get('gap_regularity_score', 0.5)
            large_gaps_count = len(gap_data.get('large_gaps', []))
            gap_score = gap_regularity * (1.0 - min(large_gaps_count / 10, 0.5))
            integrity_components.append(('gaps', gap_score, 0.2))
            
            if gap_score < 0.7:
                issues.append("Irregular time gaps detected")
            
            # Leakage analysis component (30% weight)
            leakage_data = validation_results.get('leakage_analysis', {})
            leakage_score_raw = leakage_data.get('temporal_leakage_score', 0.0)
            leakage_score = 1.0 - leakage_score_raw  # Invert since lower leakage is better
            integrity_components.append(('leakage', leakage_score, 0.3))
            
            if leakage_data.get('leakage_detected', False):
                issues.append("Temporal leakage detected")
            
            # Trend/seasonality component (20% weight)
            trend_data = validation_results.get('trend_analysis', {})
            seasonality_data = validation_results.get('seasonality_analysis', {})
            
            # This is more informational, so give a moderate score unless there are issues
            pattern_score = 0.7  # Default reasonable score
            integrity_components.append(('patterns', pattern_score, 0.2))
            
            # Calculate weighted score
            total_score = 0.0
            total_weight = 0.0
            
            for component_name, score, weight in integrity_components:
                total_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                assessment['score'] = float(total_score / total_weight)
            
            # Determine issues and integrity level
            assessment['issues_detected'] = len(issues) > 0
            assessment['issue_summary'] = issues
            
            if assessment['issues_detected']:
                assessment['integrity_level'] = 'ISSUES_DETECTED'
            elif assessment['score'] >= 0.9:
                assessment['integrity_level'] = 'EXCELLENT'
            elif assessment['score'] >= 0.8:
                assessment['integrity_level'] = 'GOOD'
            elif assessment['score'] >= 0.6:
                assessment['integrity_level'] = 'ACCEPTABLE'
            else:
                assessment['integrity_level'] = 'POOR'
            
            return assessment
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.warning(f"⚠️ Integrity assessment computation failed: {str(e)}", "TimeSeriesValidator")
            else:
                self.logger.warning(f"⚠️ Integrity assessment computation failed: {str(e)}")
            
            return {
                'score': 0.5,
                'issues_detected': True,
                'integrity_level': 'UNKNOWN',
                'issue_summary': [f"Assessment failed: {str(e)}"]
            }
    
    def _generate_timeseries_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate time series improvement recommendations"""
        try:
            recommendations = []
            
            # Temporal ordering recommendations
            ordering_data = validation_results.get('ordering_analysis', {})
            if not ordering_data.get('is_sorted', True):
                recommendations.append("Sort data by timestamp to ensure proper temporal ordering")
            
            if ordering_data.get('duplicate_timestamps', 0) > 0:
                recommendations.append("Handle duplicate timestamps: aggregate or remove duplicates")
            
            # Gap analysis recommendations
            gap_data = validation_results.get('gap_analysis', {})
            if len(gap_data.get('large_gaps', [])) > 0:
                recommendations.append("Address large time gaps: consider interpolation or gap handling strategies")
            
            # Leakage recommendations
            leakage_data = validation_results.get('leakage_analysis', {})
            if len(leakage_data.get('future_looking_features', [])) > 0:
                recommendations.append("Remove or fix future-looking features to prevent temporal leakage")
            
            if len(leakage_data.get('suspicious_patterns', [])) > 0:
                recommendations.append("Investigate suspicious correlation patterns for potential leakage")
            
            # Seasonality recommendations
            seasonality_data = validation_results.get('seasonality_analysis', {})
            if len(seasonality_data.get('detected_seasons', [])) > 0:
                recommendations.append("Consider incorporating seasonal features or seasonal decomposition")
            
            # Trend recommendations
            trend_data = validation_results.get('trend_analysis', {})
            if trend_data.get('trend_direction') != 'STABLE':
                recommendations.append("Account for trend in modeling: consider detrending or trend-aware methods")
            
            return recommendations
            
        except Exception as e:
            return [f"Could not generate time series recommendations: {str(e)}"]
