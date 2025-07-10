#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - UNIFIED PRODUCTION ENTRY POINT
Single, clean entry point for the entire enterprise system.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings

def setup_environment():
    """Sets up the environment for the application."""
    # Force UTF-8 encoding for all I/O to prevent UnicodeEncodeError on Windows
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    # Add project root to path for consistent imports
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Force CPU-only operation for stability
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Suppress TensorFlow and other warnings for a cleaner enterprise output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings('ignore')

def main():
    """
    Main entry point for the NICEGOLD Enterprise ProjectP.
    Initializes the master menu system and starts the application.
    """
    try:
        setup_environment()
        
        # We import here to ensure the environment is set up first.
        from core.unified_master_menu_system import UnifiedMasterMenuSystem
        
        master_menu = UnifiedMasterMenuSystem()
        if master_menu.initialize_components():
            master_menu.start()
        else:
            print("\n❌ CRITICAL: Failed to initialize core components. System cannot start.")
            sys.exit(1)
            
        return True
    
    except ImportError as e:
        print(f"\n❌ CRITICAL IMPORT ERROR: {e}")
        print("Please ensure all dependencies are installed correctly using 'pip install -r requirements.txt'")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An unexpected critical error occurred: {e}")
        # Use traceback for detailed error info in development
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if main():
        print("\n✅ System exited gracefully.")
    else:
        print("\n❌ System exited with errors.")
