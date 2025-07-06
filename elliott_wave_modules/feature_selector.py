"""
Enterprise-compliant Elliott Wave Feature Selector
Wrapper for RealProfitFeatureSelector - NO FALLBACK LOGIC
Author: AI Assistant
Date: 2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from real_profit_feature_selector import RealProfitFeatureSelector
import logging

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureSelector(RealProfitFeatureSelector):
    """
    Enterprise-compliant Elliott Wave Feature Selector
    
    This is a direct wrapper for RealProfitFeatureSelector.
    NO fast mode, NO fallback logic, NO sampling.
    Only real, production-grade feature selection with AUC >= 70%.
    """
    
    def __init__(self, data=None, target_col='target', **kwargs):
        """Initialize with enterprise compliance"""
        logger.info("Initializing Elliott Wave FeatureSelector - Enterprise Mode Only")
        
        # Initialize parent with enterprise settings
        super().__init__(
            data=data,
            target_col=target_col,
            enterprise_mode=True,
            fast_mode=False,  # NEVER allow fast mode
            **kwargs
        )
        
        logger.info("Elliott Wave FeatureSelector initialized - Enterprise compliance guaranteed")
    
    def select_features(self, *args, **kwargs):
        """
        Enterprise feature selection - NO fallback allowed
        """
        logger.info("Starting Elliott Wave enterprise feature selection - Real data only")
        
        # Force enterprise mode
        kwargs['enterprise_mode'] = True
        kwargs['fast_mode'] = False
        
        # Call parent's enterprise method
        return super().select_features(*args, **kwargs)

# For backward compatibility
def create_feature_selector(*args, **kwargs):
    """Factory function for creating enterprise selector"""
    return FeatureSelector(*args, **kwargs)

# Export
__all__ = ['FeatureSelector', 'create_feature_selector']
