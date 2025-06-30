#!/usr/bin/env python3
"""
Core System Package
🏢 NICEGOLD Enterprise Core Components
"""

__version__ = "2.0 DIVINE EDITION"
__author__ = "NICEGOLD Enterprise"

# Import with error handling
try:
    from .compliance import EnterpriseComplianceValidator
    from .config import load_enterprise_config, EnterpriseConfig
    from .logger import setup_enterprise_logger, EnterpriseLogger
    from .menu_system import MenuSystem
    
    __all__ = [
        'EnterpriseComplianceValidator',
        'load_enterprise_config',
        'EnterpriseConfig', 
        'setup_enterprise_logger',
        'EnterpriseLogger',
        'MenuSystem'
    ]
    
except ImportError as e:
    print(f"⚠️ Warning: Some core modules could not be imported: {e}")
    __all__ = []
