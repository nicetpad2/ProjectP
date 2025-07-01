#!/usr/bin/env python3
"""
📊 NICEGOLD ENTERPRISE LOGGER
ระบบการบันทึกข้อมูลระดับ Enterprise
"""

import logging
import sys
import platform
from datetime import datetime
from typing import Dict, Optional
import os
import io

def setup_enterprise_logger(log_level: str = "INFO") -> logging.Logger:
    """ตั้งค่า Logger ระดับ Enterprise พร้อม Unicode support"""
    
    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("NICEGOLD_Enterprise")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter with safe characters for Windows
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler with UTF-8 encoding support
    try:
        # Try to create UTF-8 console handler
        console_handler = logging.StreamHandler(
            io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        )
    except (AttributeError, OSError):
        # Fallback for systems that don't support UTF-8 console
        console_handler = logging.StreamHandler(sys.stdout)
    
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with UTF-8 encoding
    try:
        file_handler = logging.FileHandler(
            f"logs/nicegold_enterprise_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
    except (OSError, UnicodeError):
        # Fallback without explicit encoding
        file_handler = logging.FileHandler(
            f"logs/nicegold_enterprise_{datetime.now().strftime('%Y%m%d')}.log"
        )
    
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def safe_log_message(message: str) -> str:
    """สร้างข้อความ log ที่ปลอดภัยสำหรับทุกระบบ"""
    # Replace emoji with safe text representations
    emoji_replacements = {
        '🚀': '[ROCKET]',
        '✅': '[CHECK]',
        '❌': '[X]',
        '⚠️': '[WARNING]',
        'ℹ️': '[INFO]',
        '🔍': '[SEARCH]',
        '💥': '[EXPLOSION]',
        '🛑': '[STOP]',
        '🌊': '[WAVE]',
        '📊': '[CHART]',
        '🎯': '[TARGET]',
        '🧠': '[BRAIN]',
        '🤖': '[ROBOT]',
        '🏆': '[TROPHY]',
        '⚡': '[LIGHTNING]',
        '🔗': '[LINK]',
        '📈': '[CHART_UP]',
        '🎉': '[PARTY]',
        '📁': '[FOLDER]',
        '🎛️': '[CONTROL]',
        '🏢': '[BUILDING]'
    }
    
    safe_message = message
    for emoji, replacement in emoji_replacements.items():
        safe_message = safe_message.replace(emoji, replacement)
    
    return safe_message

class EnterpriseLogger:
    """Enterprise Logger Class with Unicode safety"""
    
    def __init__(self, name: str = "NICEGOLD"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """ตั้งค่า Logger"""
        if not self.logger.handlers:
            try:
                # Try UTF-8 handler first
                handler = logging.StreamHandler(
                    io.TextIOWrapper(
                        sys.stdout.buffer, 
                        encoding='utf-8', 
                        errors='replace'
                    )
                )
            except (AttributeError, OSError):
                # Fallback to standard handler
                handler = logging.StreamHandler()
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str):
        """Log Info Message"""
        safe_message = safe_log_message(f"[INFO] {message}")
        self.logger.info(safe_message)
    
    def warning(self, message: str):
        """Log Warning Message"""
        safe_message = safe_log_message(f"[WARNING] {message}")
        self.logger.warning(safe_message)
    
    def error(self, message: str):
        """Log Error Message"""
        safe_message = safe_log_message(f"[X] {message}")
        self.logger.error(safe_message)
    
    def success(self, message: str):
        """Log Success Message"""
        safe_message = safe_log_message(f"[CHECK] {message}")
        self.logger.info(safe_message)
    
    def debug(self, message: str):
        """Log Debug Message"""
        safe_message = safe_log_message(f"[SEARCH] {message}")
        self.logger.debug(safe_message)
