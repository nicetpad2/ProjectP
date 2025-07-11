#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 NICEGOLD PROJECTP - MENU 1 ELLIOTT WAVE WRAPPER
Legacy wrapper for compatibility with existing system.
Redirects to the enhanced menu implementation.
"""

from typing import Dict, Any, Optional
from .enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
from core.unified_enterprise_logger import get_unified_logger


def run_menu_1_elliott_wave(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    🌊 Legacy wrapper function for Menu 1 Elliott Wave Pipeline
    
    This function maintains compatibility with older system components
    while redirecting to the enhanced implementation.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dict containing pipeline results
    """
    logger = get_unified_logger()
    
    try:
        logger.info("🌊 Starting Elliott Wave Full Pipeline (Legacy Wrapper)")
        
        # Create enhanced menu instance
        enhanced_menu = EnhancedMenu1ElliottWave(config=config)
        
        # Run the enhanced pipeline
        results = enhanced_menu.run()
        
        logger.info("✅ Elliott Wave Pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"❌ Elliott Wave Pipeline failed: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "message": "Legacy wrapper failed to execute enhanced pipeline"
        }


def get_menu_1_info() -> Dict[str, Any]:
    """
    Get information about Menu 1 Elliott Wave capabilities
    
    Returns:
        Dict containing menu information
    """
    return {
        "name": "Elliott Wave Full Pipeline",
        "description": "Complete AI-powered Elliott Wave trading system",
        "features": [
            "Real market data processing",
            "Elliott Wave pattern recognition",
            "CNN-LSTM deep learning",
            "DQN reinforcement learning",
            "SHAP + Optuna feature selection",
            "Enterprise performance analytics"
        ],
        "requirements": {
            "min_auc": 0.70,
            "data_policy": "real_data_only",
            "enterprise_compliance": True
        },
        "wrapper_version": "2.0",
        "enhanced_implementation": "EnhancedMenu1ElliottWave"
    }


# Legacy compatibility exports
__all__ = [
    'run_menu_1_elliott_wave',
    'get_menu_1_info'
]
