#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CORE ML PROTECTION SYSTEM
Main orchestrator for Enterprise ML Protection with modular architecture

Core Features:
- Centralized protection configuration
- Module coordination and orchestration
- Enterprise-grade protection reporting
- Advanced logging integration
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
    from core.real_time_progress_manager import get_progress_manager, ProgressType
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    import logging

# Import Protection Modules
from .overfitting_detector import OverfittingDetector
from .leakage_detector import DataLeakageDetector
from .noise_analyzer import NoiseQualityAnalyzer
from .feature_analyzer import FeatureStabilityAnalyzer
from .timeseries_validator import TimeSeriesValidator


class EnterpriseMLProtectionSystem:
    """🛡️ Core Enterprise ML Protection System - Modular Architecture"""
    
    def __init__(self, config: Dict = None, logger=None):
        # Initialize advanced logging system
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger = get_terminal_logger()
            self.progress_manager = get_progress_manager()
        else:
            self.logger = logger or logging.getLogger(__name__)
            self.progress_manager = None
        
        self.config = config or {}
        
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
        
        # Initialize protection modules
        self._initialize_protection_modules()
        
        if ADVANCED_LOGGING_AVAILABLE:
            self.logger.info("🛡️ Enterprise ML Protection System initialized", "MLProtection_Core")
        else:
            self.logger.info("🛡️ Enterprise ML Protection System initialized")
    
    def _initialize_protection_modules(self):
        """Initialize all protection modules with shared configuration"""
        try:
            self.overfitting_detector = OverfittingDetector(
                config=self.protection_config, 
                logger=self.logger
            )
            self.leakage_detector = DataLeakageDetector(
                config=self.protection_config, 
                logger=self.logger
            )
            self.noise_analyzer = NoiseQualityAnalyzer(
                config=self.protection_config, 
                logger=self.logger
            )
            self.feature_analyzer = FeatureStabilityAnalyzer(
                config=self.protection_config, 
                logger=self.logger
            )
            self.timeseries_validator = TimeSeriesValidator(
                config=self.protection_config, 
                logger=self.logger
            )
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.info("✅ All protection modules initialized successfully", "MLProtection_Core")
            else:
                self.logger.info("✅ All protection modules initialized successfully")
                
        except Exception as e:
            error_msg = f"❌ Failed to initialize protection modules: {str(e)}"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(error_msg, "MLProtection_Core")
            else:
                self.logger.error(error_msg)
            raise
    
    def update_protection_config(self, new_config: Dict) -> bool:
        """Update protection configuration for all modules"""
        try:
            self.protection_config.update(new_config)
            
            # Update configuration for all modules
            for module in [self.overfitting_detector, self.leakage_detector, 
                          self.noise_analyzer, self.feature_analyzer, self.timeseries_validator]:
                if hasattr(module, 'update_config'):
                    module.update_config(new_config)
            
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.info("🔧 Protection configuration updated", "MLProtection_Core")
            else:
                self.logger.info("🔧 Protection configuration updated")
            return True
            
        except Exception as e:
            error_msg = f"❌ Failed to update protection configuration: {str(e)}"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(error_msg, "MLProtection_Core")
            else:
                self.logger.error(error_msg)
            return False
    
    def get_protection_config(self) -> Dict:
        """Get current protection configuration"""
        return self.protection_config.copy()
    
    def comprehensive_protection_analysis(self, X: np.ndarray, y: np.ndarray, 
                                        datetime_col: str = None, 
                                        model: Any = None,
                                        process_id: str = None) -> Dict[str, Any]:
        """
        🛡️ Comprehensive Enterprise ML Protection Analysis
        
        Args:
            X: Feature matrix
            y: Target vector  
            datetime_col: Name of datetime column for temporal analysis
            model: ML model for advanced analysis (optional)
            process_id: Process identifier for tracking
            
        Returns:
            Comprehensive protection analysis results
        """
        try:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.info("🛡️ Starting comprehensive ML protection analysis", "MLProtection_Analysis")
                # Create progress tracking
                task_id = self.progress_manager.create_progress(
                    "Enterprise ML Protection Analysis", 
                    6,  # total steps
                    ProgressType.ANALYSIS
                )
                return self._run_comprehensive_analysis(X, y, datetime_col, model, process_id, task_id)
            else:
                self.logger.info("🛡️ Starting comprehensive ML protection analysis")
                return self._run_comprehensive_analysis(X, y, datetime_col, model, process_id, None)
                
        except Exception as e:
            error_msg = f"❌ Comprehensive protection analysis failed: {str(e)}"
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(error_msg, "MLProtection_Analysis")
            else:
                self.logger.error(error_msg)
            
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'enterprise_ready': False,
                'critical_issues': [f"Analysis failed: {str(e)}"]
            }
    
    def _run_comprehensive_analysis(self, X, y, datetime_col, model, process_id, task_id):
        """Execute comprehensive analysis with progress tracking"""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': X.shape,
            'protection_config': self.protection_config,
            'modules_results': {},
            'overall_assessment': {},
            'enterprise_ready': False,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Step 1: Data Leakage Detection
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.update_progress(task_id, 1, "Analyzing data leakage risks...")
            
            leakage_results = self.leakage_detector.detect_data_leakage(X, y, datetime_col)
            analysis_results['modules_results']['leakage_detection'] = leakage_results
            
            # Step 2: Overfitting Detection
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.update_progress(task_id, 1, "Detecting overfitting patterns...")
                
            overfitting_results = self.overfitting_detector.detect_overfitting(X, y, model, process_id)
            analysis_results['modules_results']['overfitting_detection'] = overfitting_results
            
            # Step 3: Noise and Quality Analysis
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.update_progress(task_id, 1, "Analyzing noise and data quality...")
                
            noise_results = self.noise_analyzer.analyze_noise_and_quality(X, y)
            analysis_results['modules_results']['noise_analysis'] = noise_results
            
            # Step 4: Feature Stability Analysis
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.update_progress(task_id, 1, "Analyzing feature stability...")
                
            stability_results = self.feature_analyzer.analyze_feature_stability(X, y, datetime_col)
            analysis_results['modules_results']['feature_stability'] = stability_results
            
            # Step 5: Time Series Validation (if applicable)
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.update_progress(task_id, 1, "Validating time series integrity...")
                
            if datetime_col:
                timeseries_results = self.timeseries_validator.validate_timeseries_integrity(X, y, datetime_col)
                analysis_results['modules_results']['timeseries_validation'] = timeseries_results
            
            # Step 6: Overall Assessment
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.update_progress(task_id, 1, "Computing overall assessment...")
                
            overall_assessment = self._compute_overall_assessment(analysis_results['modules_results'])
            analysis_results['overall_assessment'] = overall_assessment
            analysis_results['enterprise_ready'] = overall_assessment['enterprise_ready']
            analysis_results['critical_issues'] = overall_assessment['critical_issues']
            analysis_results['warnings'] = overall_assessment['warnings']
            analysis_results['recommendations'] = overall_assessment['recommendations']
            
            # Complete progress
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.complete_progress(task_id, "✅ Analysis completed successfully")
            
            # Store results
            self.protection_results = analysis_results
            
            return analysis_results
            
        except Exception as e:
            if task_id and ADVANCED_LOGGING_AVAILABLE:
                self.progress_manager.fail_progress(task_id, f"Analysis failed: {str(e)}")
            raise
    
    def _compute_overall_assessment(self, modules_results: Dict) -> Dict:
        """Compute overall enterprise readiness assessment"""
        try:
            assessment = {
                'enterprise_ready': True,
                'overall_score': 0.0,
                'component_scores': {},
                'critical_issues': [],
                'warnings': [],
                'recommendations': [],
                'readiness_level': 'UNKNOWN'
            }
            
            total_score = 0.0
            component_count = 0
            
            # Analyze each component
            for component, results in modules_results.items():
                if isinstance(results, dict) and 'status' in results:
                    component_score = 0.0
                    
                    # Score based on component status and results
                    if component == 'leakage_detection':
                        if results.get('leakage_detected', True):
                            assessment['critical_issues'].append("Data leakage detected")
                            assessment['enterprise_ready'] = False
                            component_score = 0.0
                        else:
                            component_score = 1.0 - results.get('leakage_score', 0.0)
                    
                    elif component == 'overfitting_detection':
                        if results.get('overfitting_detected', True):
                            assessment['critical_issues'].append("Overfitting detected")
                            assessment['enterprise_ready'] = False
                            component_score = 0.0
                        else:
                            component_score = 1.0 - results.get('overfitting_score', 0.0)
                    
                    elif component == 'noise_analysis':
                        noise_level = results.get('noise_level', 1.0)
                        if noise_level > self.protection_config['noise_threshold']:
                            assessment['warnings'].append(f"High noise level: {noise_level:.3f}")
                        component_score = max(0.0, 1.0 - noise_level)
                    
                    elif component == 'feature_stability':
                        stability_score = results.get('stability_score', 0.0)
                        if stability_score < 0.7:
                            assessment['warnings'].append(f"Low feature stability: {stability_score:.3f}")
                        component_score = stability_score
                    
                    elif component == 'timeseries_validation':
                        if results.get('temporal_issues', True):
                            assessment['warnings'].append("Time series integrity issues detected")
                        component_score = results.get('integrity_score', 0.5)
                    
                    assessment['component_scores'][component] = component_score
                    total_score += component_score
                    component_count += 1
            
            # Calculate overall score
            if component_count > 0:
                assessment['overall_score'] = total_score / component_count
            
            # Determine readiness level
            if assessment['enterprise_ready'] and len(assessment['critical_issues']) == 0:
                if assessment['overall_score'] >= 0.9:
                    assessment['readiness_level'] = 'ENTERPRISE_READY'
                elif assessment['overall_score'] >= 0.8:
                    assessment['readiness_level'] = 'PRODUCTION_READY'
                elif assessment['overall_score'] >= 0.7:
                    assessment['readiness_level'] = 'DEVELOPMENT_READY'
                else:
                    assessment['readiness_level'] = 'NEEDS_IMPROVEMENT'
            else:
                assessment['readiness_level'] = 'NOT_READY'
                assessment['enterprise_ready'] = False
            
            # Generate recommendations
            if assessment['overall_score'] < 0.8:
                assessment['recommendations'].append("Improve overall data quality and model validation")
            
            if len(assessment['critical_issues']) > 0:
                assessment['recommendations'].append("Address all critical issues before production deployment")
            
            return assessment
            
        except Exception as e:
            if ADVANCED_LOGGING_AVAILABLE:
                self.logger.error(f"❌ Overall assessment computation failed: {str(e)}", "MLProtection_Assessment")
            else:
                self.logger.error(f"❌ Overall assessment computation failed: {str(e)}")
            
            return {
                'enterprise_ready': False,
                'overall_score': 0.0,
                'component_scores': {},
                'critical_issues': [f"Assessment failed: {str(e)}"],
                'warnings': [],
                'recommendations': ["Fix assessment computation issues"],
                'readiness_level': 'ERROR'
            }
    
    def get_protection_summary(self) -> Dict:
        """Get a summary of the latest protection analysis"""
        if not self.protection_results:
            return {
                'status': 'NO_ANALYSIS',
                'message': 'No protection analysis has been performed yet'
            }
        
        return {
            'status': 'AVAILABLE',
            'timestamp': self.protection_results.get('timestamp'),
            'enterprise_ready': self.protection_results.get('enterprise_ready', False),
            'overall_score': self.protection_results.get('overall_assessment', {}).get('overall_score', 0.0),
            'readiness_level': self.protection_results.get('overall_assessment', {}).get('readiness_level', 'UNKNOWN'),
            'critical_issues_count': len(self.protection_results.get('critical_issues', [])),
            'warnings_count': len(self.protection_results.get('warnings', []))
        }
