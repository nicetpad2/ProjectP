#!/usr/bin/env python3
"""
🎬 NICEGOLD ENTERPRISE - DEMO SCRIPT
Quick demonstration of advanced logging features
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_advanced_logging():
    """Demonstrate advanced logging features"""
    
    print("🎬 NICEGOLD Enterprise - Advanced Logging Demo")
    print("="*60)
    
    try:
        # Import advanced logger
        from core.advanced_logger import get_advanced_logger
        
        # Create logger
        logger = get_advanced_logger("DEMO")
        
        print("\n1. 🚀 Basic Logging Messages:")
        logger.info("System starting up...")
        logger.success("Database connection established")
        logger.warning("High memory usage detected")
        logger.debug("Debug information logged")
        
        print("\n2. 📊 Process Tracking Demo:")
        logger.start_process_tracking("demo_process", "Demo Process", 5)
        
        for i in range(1, 6):
            logger.update_process_progress("demo_process", i, f"Processing step {i}")
            time.sleep(0.5)  # Simulate work
        
        logger.complete_process("demo_process", True)
        
        print("\n3. 🛡️ Error Handling Demo:")
        try:
            # Simulate an error
            raise ValueError("This is a demo error")
        except Exception as e:
            logger.error("Demo error occurred", exception=e)
        
        print("\n4. 📈 Performance Summary:")
        logger.display_performance_summary()
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Run demo"""
    demo_advanced_logging()

if __name__ == "__main__":
    main()
