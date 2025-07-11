"""core/logger.py
Legacy compatibility module mapping to the Unified Enterprise Logger.
This exists to satisfy older imports (`import core.logger`) that may persist in
third-party or legacy code after the logging system unification.
All helper functions simply forward to the UnifiedEnterpriseLogger instance.
"""
from core.unified_enterprise_logger import get_unified_logger as _get_unified_logger, UnifiedEnterpriseLogger

# Obtain a singleton unified logger instance for legacy use
_logger = _get_unified_logger("LegacyCompatibilityLogger")

# Expose standard logging functions expected by old code
info = _logger.info
warning = _logger.warning
error = _logger.error
debug = _logger.debug
critical = _logger.critical
success = getattr(_logger, "success", _logger.info)

# Provide a convenience function to retrieve the underlying logger
def get_logger():
    """Return the underlying UnifiedEnterpriseLogger instance (legacy call)."""
    return _logger

# Legacy alias expected by some call sites
get_unified_logger = get_logger 

def setup_enterprise_logger(name: str = "LegacyEnterpriseLogger"):
    """Legacy function for setting up enterprise logger (compatibility)."""
    return _get_unified_logger(name)

# Legacy class alias for backward compatibility
EnterpriseLogger = UnifiedEnterpriseLogger 