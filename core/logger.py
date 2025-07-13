#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[DEPRECATED] CORE LOGGER - COMPATIBILITY WRAPPER
This module is deprecated. Use core.unified_enterprise_logger directly.

This compatibility wrapper forwards all calls to the unified enterprise logger.
"""

import warnings
from core.unified_enterprise_logger import get_unified_logger, UnifiedEnterpriseLogger

# Issue deprecation warning
warnings.warn(
    "core.logger is deprecated. Use core.unified_enterprise_logger instead.",
    DeprecationWarning,
    stacklevel=2
)

# Export the unified logger functions for backward compatibility
def get_logger(name: str = "NICEGOLD_PROJECT"):
    """Deprecated: Use get_unified_logger instead"""
    warnings.warn(
        "get_logger is deprecated. Use get_unified_logger instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_unified_logger(name)

def setup_enterprise_logger(name: str = "NICEGOLD_PROJECT", level: str = "INFO"):
    """Deprecated: Use get_unified_logger instead"""
    warnings.warn(
        "setup_enterprise_logger is deprecated. Use get_unified_logger instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_unified_logger(name)

# Legacy class alias
class EnterpriseLogger:
    """Deprecated: Use UnifiedEnterpriseLogger instead"""
    
    def __init__(self, name: str = "NICEGOLD_PROJECT"):
        warnings.warn(
            "EnterpriseLogger is deprecated. Use UnifiedEnterpriseLogger instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.logger = get_unified_logger(name)
    
    def __getattr__(self, name):
        return getattr(self.logger, name)

# Export all the main functions from unified enterprise logger
__all__ = ['get_logger', 'get_unified_logger', 'UnifiedEnterpriseLogger', 'setup_enterprise_logger', 'EnterpriseLogger']

# Re-export from unified logger for compatibility
from core.unified_enterprise_logger import * 