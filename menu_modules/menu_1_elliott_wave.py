#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌊 MENU 1 ELLIOTT WAVE - REDIRECT TO REAL ENTERPRISE MENU
Legacy compatibility wrapper that redirects to the real enterprise menu system

This file maintains backward compatibility while directing to the new
Real Enterprise Menu 1 system with complete AI functionality.
"""

import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Suppress deprecation warnings for clean output
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import the real enterprise menu system
try:
    from .real_enterprise_menu_1 import RealEnterpriseMenu1 as RealMenu1
    REAL_MENU_AVAILABLE = True
except ImportError:
    REAL_MENU_AVAILABLE = False
    RealMenu1 = None
    print("⚠️ Real Enterprise Menu 1 not available")

# Legacy compatibility function
def run_menu_1_elliott_wave(config=None):
    """
    Legacy function that redirects to Real Enterprise Menu 1
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Menu 1 execution results
    """
    if not REAL_MENU_AVAILABLE:
        print("❌ Real Enterprise Menu 1 not available")
        return {"status": "error", "message": "Real menu not available"}
    
    print("🔄 Redirecting to Real Enterprise Menu 1...")
    
    # Initialize real menu
    if RealMenu1 is not None:
        menu = RealMenu1(config or {})
        # Run the real pipeline (correct method name)
        return menu.run()
    else:
        return {"status": "error", "message": "RealMenu1 class not available"}

# Backward compatibility class
class Menu1ElliottWave:
    """Legacy class that wraps Real Enterprise Menu 1"""
    
    def __init__(self, config=None):
        self.config = config or {}
        if REAL_MENU_AVAILABLE:
            self.real_menu = RealMenu1(self.config)
        else:
            self.real_menu = None
    
    def run_full_pipeline(self):
        """Legacy method that calls real pipeline"""
        if not self.real_menu:
            print("❌ Real Enterprise Menu 1 not available")
            return {"status": "error", "message": "Real menu not available"}
        
        # Use correct method name
        return self.real_menu.run()
    
    def display_beautiful_menu(self):
        """Legacy method for menu display"""
        print("🌊 Elliott Wave Full Pipeline (Real Enterprise)")
        print("   🧠 Real AI Processing with CNN-LSTM + DQN")
        print("   📊 Real Market Data Analysis")
        print("   🎨 Beautiful Progress Tracking")

# Export for backward compatibility
__all__ = [
    'run_menu_1_elliott_wave',
    'Menu1ElliottWave'
]

if __name__ == "__main__":
    # Test execution
    print("🧪 Testing Menu 1 Elliott Wave...")
    result = run_menu_1_elliott_wave()
    print(f"✅ Test completed: {result}") 