#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML PROTECTION MODULES - Modular Architecture

Enterprise-level ML Protection System broken into specialized modules:
- Core Protection System
- Overfitting Detection  
- Data Leakage Prevention
- Noise Detection & Quality Analysis
- Feature Stability Analysis
- Time Series Validation
"""

from .core_protection import EnterpriseMLProtectionSystem
from .overfitting_detector import OverfittingDetector
from .leakage_detector import DataLeakageDetector
from .noise_analyzer import NoiseQualityAnalyzer
from .feature_analyzer import FeatureStabilityAnalyzer
from .timeseries_validator import TimeSeriesValidator

__all__ = [
    'EnterpriseMLProtectionSystem',
    'OverfittingDetector',
    'DataLeakageDetector', 
    'NoiseQualityAnalyzer',
    'FeatureStabilityAnalyzer',
    'TimeSeriesValidator'
]

__version__ = "1.0.0"
__author__ = "NICEGOLD Enterprise ML Team"
