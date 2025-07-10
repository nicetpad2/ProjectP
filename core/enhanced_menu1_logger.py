#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 ENHANCED LOGGING SYSTEM FOR MENU 1 [DEPRECATED]
Advanced logging system with timestamped files and beautiful progress tracking

⚠️ DEPRECATED: This logger is now deprecated. Please use the unified_enterprise_logger instead.
    Import from unified_enterprise_logger directly or use logger_compatibility for transition.

Example:
    from core.unified_enterprise_logger import get_unified_logger
    logger = get_unified_logger("MENU1")

OR for compatibility:
    from core.logger_compatibility import get_enterprise_menu1_logger
    logger = get_enterprise_menu1_logger()

This file now simply forwards all calls to the unified_enterprise_logger.
"""

import os
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Issue a deprecation warning
warnings.warn(
    "The enhanced_menu1_logger is deprecated. Use unified_enterprise_logger instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the unified logger
from core.unified_enterprise_logger import (
    get_unified_logger as _get_unified_logger,
    CompatibilityLogger
)

# Forward all calls to the unified logger
def get_enhanced_menu1_logger(component: str = "MENU1"):
    """Get enhanced menu1 logger (forwarded to unified logger)"""
    return _get_unified_logger(component)

class EnhancedMenu1Logger:
    """Legacy EnhancedMenu1Logger class that forwards to unified logger"""
    
    def __init__(self, log_name: str = "Menu1_Elliott_Wave"):
        self._logger = _get_unified_logger(log_name)
        
    def __getattr__(self, name):
        return getattr(self._logger, name)
        
        # Create timestamped log file
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"{log_name}_{timestamp}.log"
        
        # Create "latest" symlink
        self.latest_link = self.logs_dir / f"{log_name}_latest.log"
        if self.latest_link.exists():
            self.latest_link.unlink()
        try:
            self.latest_link.symlink_to(self.log_file.name)
        except OSError:
            # Fallback for systems that don't support symlinks
            pass
        
        # Setup file logger
        self.setup_file_logger()
        
        # Initialize session
        self.log_session_start()
    
    def setup_file_logger(self):
        """Setup file logging with detailed formatting"""
        self.file_logger = get_unified_logger()
        self.file_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.file_logger.addHandler(file_handler)
        self.file_logger.propagate = False
    
    def log_session_start(self):
        """Log session start information"""
        session_info = [
            "=" * 80,
            f"🏢 NICEGOLD ENTERPRISE - MENU 1 ELLIOTT WAVE SESSION",
            f"📅 Session Start: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"📝 Log File: {self.log_file}",
            f"🔗 Latest Link: {self.latest_link}",
            "=" * 80
        ]
        
        for line in session_info:
            print(line)
            self.file_logger.info(line)
    
    def log_step_start(self, step_num: int, step_name: str, description: str):
        """Log step start with beautiful formatting"""
        step_header = [
            "",
            f"┌{'─' * 75}┐",
            f"│ Step {step_num:2d}/10: {step_name:<30} │",
            f"│ {description:<73} │",
            f"└{'─' * 75}┘"
        ]
        
        for line in step_header:
            print(line)
            self.file_logger.info(line)
    
    def log_progress(self, message: str, level: str = "INFO"):
        """Log progress message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"   [{timestamp}] {message}"
        
        print(formatted_msg)
        
        if level == "ERROR":
            self.file_logger.error(message)
        elif level == "WARNING":
            self.file_logger.warning(message)
        else:
            self.file_logger.info(message)
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log error with detailed information"""
        error_header = f"❌ ERROR: {error_msg}"
        print(error_header)
        self.file_logger.error(error_header)
        
        if exception:
            import traceback
            error_details = traceback.format_exc()
            print(f"   Exception Details: {str(exception)}")
            self.file_logger.error(f"Exception Details: {str(exception)}")
            self.file_logger.error(f"Full Traceback:\n{error_details}")
    
    def log_success(self, message: str):
        """Log success message"""
        success_msg = f"✅ {message}"
        print(success_msg)
        self.file_logger.info(success_msg)
    
    def log_warning(self, message: str):
        """Log warning message"""
        warning_msg = f"⚠️ {message}"
        print(warning_msg)
        self.file_logger.warning(warning_msg)
    
    def info(self, message: str):
        """Log info message - for compatibility with standard logger interface"""
        info_msg = f"ℹ️ {message}"
        print(info_msg)
        self.file_logger.info(message)
    
    def debug(self, message: str):
        """Log debug message - for compatibility with standard logger interface"""
        debug_msg = f"🔍 {message}"
        print(debug_msg)
        self.file_logger.debug(message)
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message - for compatibility with standard logger interface"""
        self.error(message, exception)
    
    def warning(self, message: str):
        """Log warning message - for compatibility with standard logger interface"""
        self.warning(message)
    
    def success(self, message: str):
        """Log success message - for compatibility with standard logger interface"""
        self.success(message)
    
    def log_session_end(self, success: bool = True):
        """Log session end with summary"""
        session_end = datetime.now()
        duration = session_end - self.session_start
        
        summary = [
            "",
            "=" * 80,
            f"📊 SESSION SUMMARY",
            f"📅 Start Time: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"📅 End Time: {session_end.strftime('%Y-%m-%d %H:%M:%S')}",
            f"⏱️ Duration: {duration}",
            f"🎯 Status: {'✅ SUCCESS' if success else '❌ FAILED'}",
            f"📝 Log File: {self.log_file}",
            "=" * 80
        ]
        
        for line in summary:
            print(line)
            self.file_logger.info(line)
    
    def show_animated_progress(self, duration: float = 1.0, width: int = 20):
        """Show animated progress bar"""
        import time
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus

        
        print("   Progress: [", end="", flush=True)
        step_time = duration / width
        
        for i in range(width):
            time.sleep(step_time)
            if i < width - 1:
                print("█", end="", flush=True)
            else:
                print("█] 100%", flush=True)
        
    def get_log_file_path(self) -> str:
        """Get the current log file path"""
        return str(self.log_file)
    
    def get_latest_log_path(self) -> str:
        """Get the latest log file path"""
        return str(self.latest_link)

# Demo function to test the logging system
def demo_enhanced_logging():
    """Demo the enhanced logging system"""
    logger = get_unified_logger()
    
    try:
        # Simulate pipeline steps
        steps = [
            (1, "🌊 Data Loading", "Loading real market data from datacsv/"),
            (2, "🔧 Data Preprocessing", "Cleaning and validating market data"),
            (3, "🌊 Elliott Wave Detection", "Detecting wave patterns"),
            (4, "🛠️ Feature Engineering", "Creating technical features"),
            (5, "🎯 Feature Selection", "SHAP + Optuna optimization")
        ]
        
        for step_num, step_name, description in steps:
            logger.log_step_start(step_num, step_name, description)
            logger.show_animated_progress(duration=0.5)
            logger.log_progress(f"{step_name} completed successfully!")
            time.sleep(0.3)
        
        logger.success("All steps completed successfully!")
        logger.log_session_end(success=True)
        
        print(f"\n📝 Log saved to: {logger.get_log_file_path()}")
        print(f"🔗 Latest log: {logger.get_latest_log_path()}")
        
    except Exception as e:
        logger.error("Demo failed", e)
        logger.log_session_end(success=False)

if __name__ == "__main__":
    demo_enhanced_logging()
