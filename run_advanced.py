#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE - QUICK LAUNCHER
Advanced Version Launcher Script
"""

import sys
import os
from pathlib import Path

def main():
    """Launch NICEGOLD ProjectP Advanced"""
    
    print("🚀 Starting NICEGOLD Enterprise ProjectP - Advanced Edition...")
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Import and run the advanced version
        from ProjectP_Advanced import main as run_advanced
        
        print("✅ Advanced version loaded successfully")
        print("🌊 Launching Elliott Wave System...")
        
        # Run the application
        return run_advanced()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Trying fallback to original version...")
        
        try:
            # Fallback to original ProjectP
            import ProjectP
            print("✅ Original version loaded")
            return 0
        except ImportError:
            print("❌ Failed to load any version")
            return 1
    
    except Exception as e:
        print(f"❌ Launch error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
