#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTERPRISE ML PROTECTION SYSTEM - MODULAR IMPORT REDIRECT
Seamless import redirect to the new modular ML protection system

This file maintains backward compatibility while directing imports
to the new modular architecture in the ml_protection package.
"""

# Import all components from the new modular system
from .ml_protection import (
    EnterpriseMLProtectionSystem,
    OverfittingDetector,
    DataLeakageDetector,
    NoiseQualityAnalyzer,
    FeatureStabilityAnalyzer,
    TimeSeriesValidator
)

# Maintain backward compatibility
__all__ = [
    'EnterpriseMLProtectionSystem',
    'OverfittingDetector',
    'DataLeakageDetector',
    'NoiseQualityAnalyzer',
    'FeatureStabilityAnalyzer',
    'TimeSeriesValidator'
]

# Version and metadata
__version__ = "2.0.0-modular"
__author__ = "NICEGOLD Enterprise ML Team"
__description__ = "Enterprise ML Protection System - Modular Architecture"

# Optional: Show a one-time migration notice
import warnings
import os

# Only show warning in development mode or if explicitly requested
if os.environ.get('SHOW_MIGRATION_WARNINGS', '').lower() == 'true':
    warnings.warn(
        "✅ Successfully using modular ML Protection System. "
        "The system has been upgraded to a modular architecture for better performance and maintainability.",
        FutureWarning,
        stacklevel=2
    )
