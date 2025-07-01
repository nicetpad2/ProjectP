#!/usr/bin/env python3
"""
Test the Unicode-safe logging system
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logger import EnterpriseLogger


def test_logging():
    """Test the logging system with various message types"""
    logger = EnterpriseLogger("TEST")
    
    print("Testing Unicode-safe logging system...")
    
    # Test different log levels
    logger.info("This is an info message 🚀")
    logger.warning("This is a warning message ⚠️")
    logger.error("This is an error message ❌")
    logger.success("This is a success message ✅")
    logger.debug("This is a debug message 🔍")
    
    # Test message with multiple emojis
    logger.info("Multi-emoji test: 🚀 🎯 📊 🧠 🤖")
    
    print("Logging test completed successfully!")

if __name__ == "__main__":
    test_logging()
