#!/usr/bin/env python3
"""
🚀 NICEGOLD ENTERPRISE - STARTUP REDIRECTOR
This script ensures users always use the correct entry point

⚠️ IMPORTANT: This script redirects to ProjectP.py
Only ProjectP.py is authorized as the main entry point
"""

import sys
from pathlib import Path


def main():
    """Redirect to the only authorized main entry point"""
    
    print("🏢 NICEGOLD ENTERPRISE PROJECTP")
    print("="*50)
    print("🎯 Entry Point Policy Enforcer")
    print("="*50)
    print()
    print("⚠️  IMPORTANT NOTICE:")
    print("   This system has ONLY ONE authorized main entry point")
    print()
    print("✅ CORRECT USAGE:")
    print("   python ProjectP.py")
    print()
    print("❌ DO NOT USE:")
    print("   - python ProjectP_Advanced.py")
    print("   - python run_advanced.py")
    print("   - Any other main files")
    print()
    print("🔄 Redirecting to ProjectP.py in 3 seconds...")
    
    # Wait for user to read the message
    import time
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...", end='\r')
        time.sleep(1)
    
    print("\n")
    print("🚀 Starting NICEGOLD Enterprise via ProjectP.py...")
    print("="*50)
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Import and run the main ProjectP
        import ProjectP
        ProjectP.main()
        return 0
        
    except ImportError as e:
        print(f"❌ Failed to import ProjectP.py: {e}")
        print("💡 Please ensure ProjectP.py exists in the project directory")
        return 1
    
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
        return 0
    
    except SystemExit:
        return 0
    
    except:  # noqa: E722
        print("❌ Unexpected execution error occurred")
        print("💡 Please check ProjectP.py and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
