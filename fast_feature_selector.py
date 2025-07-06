"""
Enterprise-compliant Fast Feature Selector (DEPRECATED)
Redirects to RealProfitFeatureSelector - NO FAST MODE
Author: AI Assistant
Date: 2024
"""

from .real_profit_feature_selector import RealProfitFeatureSelector
import logging
import warnings

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FastFeatureSelector(RealProfitFeatureSelector):
    """
    DEPRECATED: Fast Feature Selector
    
    This class is DEPRECATED and redirects to RealProfitFeatureSelector.
    NO fast mode is allowed in enterprise environment.
    Only real, production-grade feature selection with AUC >= 70%.
    """
    
    def __init__(self, data=None, target_col='target', **kwargs):
        """Initialize with enterprise compliance (NO fast mode)"""
        warnings.warn(
            "FastFeatureSelector is DEPRECATED. Use RealProfitFeatureSelector directly. "
            "Fast mode is NOT allowed in enterprise environment.",
            DeprecationWarning,
            stacklevel=2
        )
        
        logger.warning("FastFeatureSelector is DEPRECATED - Redirecting to RealProfitFeatureSelector")
        
        # Force enterprise mode - NO fast mode allowed
        kwargs['enterprise_mode'] = True
        kwargs['fast_mode'] = False
        
        # Initialize parent with enterprise settings
        super().__init__(
            data=data,
            target_col=target_col,
            **kwargs
        )
        
        logger.info("FastFeatureSelector redirected to RealProfitFeatureSelector - Enterprise compliance guaranteed")
    
    def select_features(self, *args, **kwargs):
        """
        Enterprise feature selection - NO fast mode allowed
        """
        logger.warning("FastFeatureSelector.select_features() called - Redirecting to enterprise mode")
        
        # Force enterprise mode
        kwargs['enterprise_mode'] = True
        kwargs['fast_mode'] = False
        
        # Call parent's enterprise method
        return super().select_features(*args, **kwargs)

# For backward compatibility
def create_fast_selector(*args, **kwargs):
    """Factory function - DEPRECATED, redirects to enterprise selector"""
    warnings.warn(
        "create_fast_selector is DEPRECATED. Use RealProfitFeatureSelector directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return FastFeatureSelector(*args, **kwargs)

# Export
__all__ = ['FastFeatureSelector', 'create_fast_selector']
