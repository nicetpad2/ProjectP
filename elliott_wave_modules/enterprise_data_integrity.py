#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛡️ ENTERPRISE DATA INTEGRITY SYSTEM
Advanced data validation and integrity checks for NICEGOLD ProjectP-1
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus


# Suppress warnings
warnings.filterwarnings('ignore')

class EnterpriseDataIntegritySystem:
    """
    🛡️ Enterprise Data Integrity System
    
    Features:
    - Data quality validation
    - Integrity checks  
    - Anomaly detection
    - Data cleaning
    - Quality scoring
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_unified_logger()
        self.validation_results = {}
        self.quality_score = 0.0
        
    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        🔍 Comprehensive data integrity validation
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        
        self.logger.info("🛡️ Starting Enterprise Data Integrity Validation")
        
        results = {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'validation_passed': True,
            'issues_found': [],
            'quality_score': 0.0,
            'recommendations': []
        }
        
        try:
            # Check for missing values
            missing_check = self._check_missing_values(data)
            results['missing_values'] = missing_check
            
            # Check for duplicates
            duplicate_check = self._check_duplicates(data)
            results['duplicates'] = duplicate_check
            
            # Check data types
            type_check = self._check_data_types(data)
            results['data_types'] = type_check
            
            # Check for outliers
            outlier_check = self._check_outliers(data)
            results['outliers'] = outlier_check
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(results)
            results['quality_score'] = quality_score
            
            # Generate recommendations
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations
            
            self.validation_results = results
            self.quality_score = quality_score
            
            self.logger.info(f"✅ Data integrity validation completed. Quality Score: {quality_score:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Data integrity validation failed: {e}")
            results['validation_passed'] = False
            results['error'] = str(e)
            return results
    
    def _check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values"""
        missing_info = {}
        
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            
            missing_info[col] = {
                'count': missing_count,
                'percentage': missing_pct,
                'critical': missing_pct > 10.0  # Critical if > 10%
            }
        
        return missing_info
    
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate records"""
        duplicate_count = data.duplicated().sum()
        duplicate_pct = (duplicate_count / len(data)) * 100
        
        return {
            'count': duplicate_count,
            'percentage': duplicate_pct,
            'critical': duplicate_pct > 5.0  # Critical if > 5%
        }
    
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data types consistency"""
        type_info = {}
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            type_info[col] = {
                'type': dtype,
                'consistent': True,  # Assume consistent for now
                'recommended_type': self._get_recommended_type(col, data[col])
            }
        
        return type_info
    
    def _check_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for outliers in numerical columns"""
        outlier_info = {}
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(data)) * 100
            
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'critical': outlier_pct > 15.0  # Critical if > 15%
            }
        
        return outlier_info
    
    def _get_recommended_type(self, col_name: str, series: pd.Series) -> str:
        """Get recommended data type for a column"""
        # Simple heuristics for type recommendation
        if 'timestamp' in col_name.lower() or 'date' in col_name.lower():
            return 'datetime64'
        elif 'price' in col_name.lower() or 'volume' in col_name.lower():
            return 'float64'
        elif 'count' in col_name.lower() or 'id' in col_name.lower():
            return 'int64'
        else:
            return str(series.dtype)
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Penalize for missing values
        if 'missing_values' in results:
            for col_info in results['missing_values'].values():
                if col_info['critical']:
                    score -= 10.0
                elif col_info['percentage'] > 0:
                    score -= 2.0
        
        # Penalize for duplicates
        if 'duplicates' in results:
            if results['duplicates']['critical']:
                score -= 15.0
            elif results['duplicates']['percentage'] > 0:
                score -= 5.0
        
        # Penalize for outliers
        if 'outliers' in results:
            for col_info in results['outliers'].values():
                if col_info['critical']:
                    score -= 5.0
                elif col_info['percentage'] > 0:
                    score -= 1.0
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Missing values recommendations
        if 'missing_values' in results:
            critical_missing = [col for col, info in results['missing_values'].items() 
                              if info['critical']]
            if critical_missing:
                recommendations.append(f"🚨 Critical: Address missing values in columns: {', '.join(critical_missing)}")
        
        # Duplicates recommendations
        if 'duplicates' in results and results['duplicates']['critical']:
            recommendations.append("🚨 Critical: Remove duplicate records")
        
        # Outliers recommendations
        if 'outliers' in results:
            critical_outliers = [col for col, info in results['outliers'].items() 
                               if info['critical']]
            if critical_outliers:
                recommendations.append(f"⚠️ Review outliers in columns: {', '.join(critical_outliers)}")
        
        # General recommendations
        if results['quality_score'] < 80:
            recommendations.append("📊 Consider comprehensive data cleaning")
        
        return recommendations
    
    def get_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        if not self.validation_results:
            return "❌ No validation results available"
        
        report = []
        report.append("🛡️ ENTERPRISE DATA INTEGRITY REPORT")
        report.append("=" * 50)
        
        results = self.validation_results
        
        report.append(f"📊 Total Records: {results['total_records']:,}")
        report.append(f"📈 Total Columns: {results['total_columns']}")
        report.append(f"⭐ Quality Score: {results['quality_score']:.2f}/100")
        report.append("")
        
        # Missing values summary
        if 'missing_values' in results:
            critical_missing = sum(1 for info in results['missing_values'].values() if info['critical'])
            if critical_missing > 0:
                report.append(f"🚨 Critical Missing Values: {critical_missing} columns")
        
        # Duplicates summary
        if 'duplicates' in results and results['duplicates']['count'] > 0:
            report.append(f"🔄 Duplicate Records: {results['duplicates']['count']:,} ({results['duplicates']['percentage']:.1f}%)")
        
        # Recommendations
        if results['recommendations']:
            report.append("")
            report.append("📋 RECOMMENDATIONS:")
            for rec in results['recommendations']:
                report.append(f"   {rec}")
        
        return "\n".join(report)

# Factory function
def create_data_integrity_system(logger: Optional[logging.Logger] = None) -> EnterpriseDataIntegritySystem:
    """Factory function to create data integrity system"""
    return EnterpriseDataIntegritySystem(logger)

# Utility functions
def validate_dataframe(data: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Quick validation of a DataFrame"""
    integrity_system = create_data_integrity_system(logger)
    return integrity_system.validate_data_integrity(data)

def get_data_quality_score(data: pd.DataFrame) -> float:
    """Get quick data quality score"""
    integrity_system = create_data_integrity_system()
    results = integrity_system.validate_data_integrity(data)
    return results.get('quality_score', 0.0)

# Export classes and functions
__all__ = [
    'EnterpriseDataIntegritySystem',
    'create_data_integrity_system',
    'validate_dataframe',
    'get_data_quality_score'
]
