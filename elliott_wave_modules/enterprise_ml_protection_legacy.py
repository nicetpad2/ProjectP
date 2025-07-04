#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTERPRISE ML PROTECTION SYSTEM - MODULAR VERSION
Enterprise-level ML Protection with Modular Architecture

⚠️ DEPRECATED: This file is now replaced by the modular ml_protection package.
Use: from elliott_wave_modules.ml_protection import EnterpriseMLProtectionSystem

Core Features (Now Modularized):
- Core Protection System (core_protection.py)
- Overfitting Detection (overfitting_detector.py)
- Data Leakage Prevention (leakage_detector.py)
- Noise Detection & Quality Analysis (noise_analyzer.py)
- Feature Stability Analysis (feature_analyzer.py)
- Time Series Validation (timeseries_validator.py)

🏢 Enterprise Standards:
- Zero Tolerance for Data Leakage
- Statistical Significance Testing
- Cross-Validation with Time Awareness
- Feature Stability Analysis
- Model Performance Degradation Detection
"""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np

# Import the new modular system
try:
    from .ml_protection import (
        EnterpriseMLProtectionSystem,
        OverfittingDetector,
        DataLeakageDetector,
        NoiseQualityAnalyzer,
        FeatureStabilityAnalyzer,
        TimeSeriesValidator
    )
    
    # For backward compatibility, expose the main class
    __all__ = [
        'EnterpriseMLProtectionSystem',
        'OverfittingDetector',
        'DataLeakageDetector',
        'NoiseQualityAnalyzer',
        'FeatureStabilityAnalyzer',
        'TimeSeriesValidator'
    ]
    
    # Deprecation warning
    warnings.warn(
        "enterprise_ml_protection.py is deprecated. Use 'from elliott_wave_modules.ml_protection import EnterpriseMLProtectionSystem' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
except ImportError as e:
    # Fallback for compatibility
    warnings.warn(
        f"Could not import modular ML protection system: {str(e)}. "
        "Please ensure the ml_protection package is properly installed.",
        ImportWarning,
        stacklevel=2
    )
    
    # Provide a basic fallback class
    class EnterpriseMLProtectionSystem:
        """Fallback ML Protection System"""
        
        def __init__(self, config: Dict = None, logger=None):
            self.config = config or {}
            self.logger = logger
            warnings.warn(
                "Using fallback ML Protection System. Full functionality requires modular components.",
                UserWarning,
                stacklevel=2
            )
        
        def comprehensive_protection_analysis(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
            """Fallback protection analysis"""
            return {
                'error': 'Modular ML protection system not available',
                'fallback_used': True,
                'enterprise_ready': False,
                'critical_issues': ['Modular protection system not available'],
                'recommendations': ['Install the complete ml_protection package']
            }
    
    __all__ = ['EnterpriseMLProtectionSystem']


# Version information
__version__ = "2.0.0-modular"
__author__ = "NICEGOLD Enterprise ML Team"
__description__ = "Enterprise ML Protection System - Modular Architecture"

# Migration guide
MIGRATION_GUIDE = """
MIGRATION GUIDE: enterprise_ml_protection.py → ml_protection package

OLD USAGE:
from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem

NEW USAGE:
from elliott_wave_modules.ml_protection import EnterpriseMLProtectionSystem

BENEFITS OF MODULAR VERSION:
✅ Better code organization and maintainability
✅ Faster loading and reduced memory usage
✅ Individual module testing and development
✅ Easier to extend with new protection methods
✅ Better separation of concerns
✅ Improved error handling and logging

INDIVIDUAL MODULES AVAILABLE:
- Core Protection: EnterpriseMLProtectionSystem
- Overfitting Detection: OverfittingDetector  
- Data Leakage Detection: DataLeakageDetector
- Noise Analysis: NoiseQualityAnalyzer
- Feature Stability: FeatureStabilityAnalyzer
- Time Series Validation: TimeSeriesValidator

EXAMPLE USAGE:
```python
from elliott_wave_modules.ml_protection import (
    EnterpriseMLProtectionSystem,
    OverfittingDetector,
    DataLeakageDetector
)

# Full protection system
protection_system = EnterpriseMLProtectionSystem(config=my_config)
results = protection_system.comprehensive_protection_analysis(X, y)

# Individual modules
overfitting_detector = OverfittingDetector(config=my_config)
overfitting_results = overfitting_detector.detect_overfitting(X, y)
```
"""

def print_migration_guide():
    """Print migration guide for users"""
    print(MIGRATION_GUIDE)


if __name__ == "__main__":
    print_migration_guide()
