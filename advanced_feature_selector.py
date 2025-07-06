"""
Enterprise-compliant Advanced Feature Selector
Wrapper for RealProfitFeatureSelector - NO FALLBACK LOGIC
Author: AI Assistant
Date: 2024
"""

from .real_profit_feature_selector import RealProfitFeatureSelector
import logging

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedFeatureSelector(RealProfitFeatureSelector):
    """
    Enterprise-compliant Advanced Feature Selector
    
    This is a direct wrapper for RealProfitFeatureSelector.
    NO fast mode, NO fallback logic, NO sampling.
    Only real, production-grade feature selection with AUC >= 70%.
    """
    
    def __init__(self, data=None, target_col='target', **kwargs):
        """Initialize with enterprise compliance"""
        logger.info("Initializing AdvancedFeatureSelector - Enterprise Mode Only")
        
        # Initialize parent with enterprise settings
        super().__init__(
            data=data,
            target_col=target_col,
            enterprise_mode=True,
            fast_mode=False,  # NEVER allow fast mode
            **kwargs
        )
        
        logger.info("AdvancedFeatureSelector initialized - Enterprise compliance guaranteed")
    
    def select_features(self, *args, **kwargs):
        """
        Enterprise feature selection - NO fallback allowed
        """
        logger.info("Starting enterprise feature selection - Real data only")
        
        # Force enterprise mode
        kwargs['enterprise_mode'] = True
        kwargs['fast_mode'] = False
        
        # Call parent's enterprise method
        return super().select_features(*args, **kwargs)

# For backward compatibility
def create_advanced_selector(*args, **kwargs):
    """Factory function for creating enterprise selector"""
    return AdvancedFeatureSelector(*args, **kwargs)

# Export
__all__ = ['AdvancedFeatureSelector', 'create_advanced_selector']
