#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP
ระบบ AI-Powered Algorithmic Trading System ระดับ Enterprise

📊 Main Entry Point - สำหรับรันระบบหลัก
⚠️ THIS IS THE ONLY AUTHORIZED MAIN ENTRY POINT
🚫 DO NOT create alternative main files - use this file only
"""

# 🛠️ CUDA FIX: Apply immediate CUDA fixes before any imports
import os
import sys
import warnings

# Force CPU-only operation to prevent CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Suppress all CUDA-related warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Enterprise Compliance Check
from core.compliance import EnterpriseComplianceValidator
from core.menu_system import MenuSystem
from core.logger import setup_enterprise_logger
from core.config import load_enterprise_config


def main():
    """จุดเริ่มต้นของระบบ NICEGOLD Enterprise - ONLY AUTHORIZED ENTRY POINT"""
    
    # Setup Enterprise Logger
    logger = setup_enterprise_logger()
    logger.info("🚀 NICEGOLD Enterprise ProjectP Starting...")
    logger.info("📌 Using ONLY authorized entry point: ProjectP.py")
    
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
