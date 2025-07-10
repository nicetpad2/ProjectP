#!/usr/bin/env python3
"""
Menu Modules Package
🎛️ NICEGOLD Enterprise Menu System
"""

__version__ = "2.0 DIVINE EDITION"
__author__ = "NICEGOLD Enterprise"

# Import with error handling
try:
    from .menu_1_elliott_wave import run_menu_1_elliott_wave
    # from .enhanced_menu_1_elliott_wave import EnhancedMenu1ElliottWave
    
    __all__ = [
        'run_menu_1_elliott_wave',
        # 'EnhancedMenu1ElliottWave'
    ]
    
except ImportError as e:
    print(f"⚠️ Warning: Some menu modules could not be imported: {e}")
    __all__ = []
