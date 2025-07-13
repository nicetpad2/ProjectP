#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - UNIFIED PRODUCTION ENTRY POINT
Single, clean entry point for the entire enterprise system.
"""

import os
import sys
# Real Enterprise Menu System
try:
    from menu_modules.real_enterprise_menu_1 import RealEnterpriseMenu1
    REAL_MENU1_AVAILABLE = True
    print("✅ Real Enterprise Menu 1 Elliott Wave system loaded!")
except ImportError as e:
    REAL_MENU1_AVAILABLE = False
    print(f"⚠️ Real Enterprise Menu 1 not available: {e}")

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

    # 🚀 ENTERPRISE TENSORFLOW CONFIGURATION
    try:
        # Apply comprehensive TensorFlow optimizations
        from core.tensorflow_config import configure_enterprise_tensorflow
        config_result = configure_enterprise_tensorflow()
        if config_result.get('enterprise_ready'):
            print("🎉 Enterprise TensorFlow Configuration Applied Successfully!")
        else:
            print("⚠️ Partial TensorFlow configuration applied")
    except ImportError:
        # Fallback to basic configuration if tensorflow_config is not available
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['PYTHONWARNINGS'] = 'ignore'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        warnings.filterwarnings('ignore')
        print("ℹ️ Basic TensorFlow configuration applied")

    # Switch Windows terminal to UTF-8 code page to avoid UnicodeEncodeError
    if os.name == 'nt':
        try:
            os.system('chcp 65001 > NUL')
            # Force the Windows console to use UTF-8 through WinAPI in case `chcp` is ineffective
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleOutputCP(65001)
                kernel32.SetConsoleCP(65001)
            except Exception:
                # Silently ignore if WinAPI is unavailable (e.g. Wine, restricted env)
                pass

            # Explicitly reconfigure Python stdout/stderr to UTF-8 to prevent Rich/Colorama
            # from raising UnicodeEncodeError when printing emojis or non-ASCII symbols.
            import io
            if hasattr(sys.stdout, "reconfigure"):
                try:
                    sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
                    sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                try:
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
                except Exception:
                    pass
        except Exception:
            pass

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
