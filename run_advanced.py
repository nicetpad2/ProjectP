#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE - REDIRECTOR TO MAIN ENTRY POINT
This script redirects to ProjectP.py (the only authorized main entry point)

⚠️ This file does not run the system directly
All execution must go through ProjectP.py
"""

import sys
from pathlib import Path


def main():
    """Redirect to ProjectP.py main entry point"""
    
    print("🚀 NICEGOLD Enterprise - Entry Point Redirector")
    print("="*60)
    print("⚠️  This script redirects to the main entry point")
    print("🎯 Main Entry Point: ProjectP.py")
    print("="*60)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        print("🔄 Redirecting to ProjectP.py...")
        
        # Import and run the main ProjectP
        import ProjectP
        print("✅ ProjectP.py loaded successfully")
        print("🌊 Starting NICEGOLD Enterprise System...")
        
        # Run the main application
        ProjectP.main()
        return 0
        
    except ImportError as e:
        print(f"❌ Failed to import ProjectP.py: {e}")
        print("💡 Please ensure ProjectP.py exists in the project directory")
        return 1
    
    except Exception as e:
        print(f"❌ Execution error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
