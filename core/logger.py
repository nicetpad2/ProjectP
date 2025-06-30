#!/usr/bin/env python3
"""
📊 NICEGOLD ENTERPRISE LOGGER
ระบบการบันทึกข้อมูลระดับ Enterprise
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Optional
import os

def setup_enterprise_logger(log_level: str = "INFO") -> logging.Logger:
    """ตั้งค่า Logger ระดับ Enterprise"""
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("NICEGOLD_Enterprise")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(
        f"logs/nicegold_enterprise_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

class EnterpriseLogger:
    """Enterprise Logger Class"""
    
    def __init__(self, name: str = "NICEGOLD"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """ตั้งค่า Logger"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str):
        """Log Info Message"""
        self.logger.info(f"ℹ️  {message}")
    
    def warning(self, message: str):
        """Log Warning Message"""
        self.logger.warning(f"⚠️  {message}")
    
    def error(self, message: str):
        """Log Error Message"""
        self.logger.error(f"❌ {message}")
    
    def success(self, message: str):
        """Log Success Message"""
        self.logger.info(f"✅ {message}")
    
    def debug(self, message: str):
        """Log Debug Message"""
        self.logger.debug(f"🔍 {message}")
