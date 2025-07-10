#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - UNIFIED PRODUCTION ENTRY POINT
Single, clean entry point for the entire enterprise system.
"""

# Environment setup must be first
import os
import sys
import warnings
from pathlib import Path

def setup_environment():
    """Sets up the environment variables and warnings for production."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    warnings.filterwarnings('ignore')
    # Add project root to the Python path
    sys.path.append(str(Path(__file__).parent))

def safe_print(message):
    """A print function that is safe from BrokenPipeError."""
    try:
        print(message)
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        # Fallback to stderr if stdout is broken (e.g., in some CI/CD runners)
        try:
            sys.stderr.write(f"{message}\n")
            sys.stderr.flush()
        except Exception:
            pass # Fails silently if stderr is also broken

def main():
    """The single main entry point for the NICEGOLD Enterprise system."""
    setup_environment()
    safe_print("="*80)
    safe_print("🏢 NICEGOLD ENTERPRISE PROJECTP - UNIFIED PRODUCTION SYSTEM")
    safe_print("✨ Zero Duplication Edition with Complete Integration")
    safe_print("="*80)
    
    try:
        from core.unified_master_menu_system import UnifiedMasterMenuSystem
        
        safe_print("🚀 Initializing Unified Master System...")
        unified_system = UnifiedMasterMenuSystem()
        unified_system.start()
        
    except ImportError as e:
        safe_print(f"❌ CRITICAL ERROR: Failed to import UnifiedMasterMenuSystem: {e}")
        safe_print("   Please ensure all core modules are correctly installed and accessible.")
        return False
    except Exception as e:
        safe_print(f"❌ An unexpected error occurred in the Unified Master System: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    try:
        if main():
            safe_print("\n✅ NICEGOLD Enterprise ProjectP - Session Complete.")
        else:
            safe_print("\n⚠️ NICEGOLD Enterprise ProjectP - Session finished with critical errors.")
            
    except KeyboardInterrupt:
        safe_print("\n👋 System interruption by user. Shutting down.")
    except Exception as e:
        safe_print(f"\n❌ A critical error occurred during system startup: {e}")
        import traceback
        traceback.print_exc()
    finally:
        safe_print("🎛️ Thank you for using the NICEGOLD Enterprise ProjectP Unified System!")
