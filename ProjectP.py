#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP
ระบบ AI-Powered Algorithmic Trading System ระดับ Enterprise

📊 Main Entry Point - สำหรับรันระบบหลัก
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# Enterprise Compliance Check
from core.compliance import EnterpriseComplianceValidator
from core.menu_system import MenuSystem
from core.logger import setup_enterprise_logger
from core.config import load_enterprise_config

def main():
    """จุดเริ่มต้นของระบบ NICEGOLD Enterprise"""
    
    # Setup Enterprise Logger
    logger = setup_enterprise_logger()
    logger.info("🚀 NICEGOLD Enterprise ProjectP Starting...")
    
    # Load Enterprise Configuration
    config = load_enterprise_config()
    
    # Validate Enterprise Compliance
    validator = EnterpriseComplianceValidator()
    if not validator.validate_enterprise_compliance():
        logger.error("❌ Enterprise Compliance Validation Failed!")
        sys.exit(1)
    
    # Initialize Menu System
    menu_system = MenuSystem(config=config, logger=logger)
    
    try:
        # Start Main Menu Loop
        menu_system.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")
        
    except Exception as e:
        logger.error(f"💥 System error: {str(e)}")
        sys.exit(1)
        
    finally:
        logger.info("✅ NICEGOLD Enterprise ProjectP Shutdown Complete")

if __name__ == "__main__":
    main()
