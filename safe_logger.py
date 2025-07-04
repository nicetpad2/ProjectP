#!/usr/bin/env python3
"""
🛡️ Safe Logger System - NICEGOLD Enterprise
Handles logging errors and file handle issues safely
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional

class SafeLogger:
    """Safe logger that handles file I/O errors gracefully"""
    
    def __init__(self, name: str = "NICEGOLD_ENTERPRISE"):
        self.name = name
        self.logger = None
        self._initialize_safe_logger()
    
    def _initialize_safe_logger(self):
        """Initialize logger with safe error handling"""
        try:
            # Create logger
            self.logger = logging.getLogger(self.name)
            self.logger.handlers.clear()  # Clear existing handlers
            self.logger.setLevel(logging.INFO)
            
            # Console handler with safe formatting
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Simple safe formatter
            formatter = logging.Formatter(
                fmt='%(levelname)s [%(asctime)s.%(msecs)03d] %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            
            # Add handler
            self.logger.addHandler(console_handler)
            
            # Prevent propagation to root logger
            self.logger.propagate = False
            
        except Exception as e:
            # Fallback to print if logger fails
            print(f"⚠️ Logger initialization failed: {e}")
            self.logger = None
    
    def info(self, message: str, extra_info: str = ""):
        """Safe info logging"""
        self._safe_log("INFO", message, extra_info)
    
    def warning(self, message: str, extra_info: str = ""):
        """Safe warning logging"""
        self._safe_log("WARNING", message, extra_info)
    
    def error(self, message: str, extra_info: str = ""):
        """Safe error logging"""
        self._safe_log("ERROR", message, extra_info)
    
    def success(self, message: str, extra_info: str = ""):
        """Safe success logging"""
        self._safe_log("SUCCESS", message, extra_info)
    
    def _safe_log(self, level: str, message: str, extra_info: str = ""):
        """Safe logging with fallback to print"""
        try:
            if self.logger and hasattr(self.logger, 'handlers') and self.logger.handlers:
                # Check if handlers are still valid
                valid_handlers = []
                for handler in self.logger.handlers:
                    try:
                        # Test if handler is still writable
                        if hasattr(handler, 'stream') and hasattr(handler.stream, 'write'):
                            if not handler.stream.closed:
                                valid_handlers.append(handler)
                    except (ValueError, AttributeError):
                        continue
                
                # Update logger with valid handlers only
                self.logger.handlers = valid_handlers
                
                if valid_handlers:
                    # Log through valid handlers
                    full_message = f"{message}"
                    if extra_info:
                        full_message += f" | {extra_info}"
                    
                    if level == "ERROR":
                        self.logger.error(full_message)
                    elif level == "WARNING":
                        self.logger.warning(full_message)
                    elif level == "SUCCESS":
                        self.logger.info(f"✅ {full_message}")
                    else:
                        self.logger.info(full_message)
                    return
            
            # Fallback to safe print
            self._safe_print(level, message, extra_info)
            
        except Exception as e:
            # Ultimate fallback
            self._safe_print("ERROR", f"Logging failed: {e}", "")
            self._safe_print(level, message, extra_info)
    
    def _safe_print(self, level: str, message: str, extra_info: str = ""):
        """Ultimate safe fallback using print"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            prefix = {
                "INFO": "ℹ️",
                "WARNING": "⚠️", 
                "ERROR": "❌",
                "SUCCESS": "✅"
            }.get(level, "📝")
            
            full_message = f"{prefix} {level} [{timestamp}] {message}"
            if extra_info:
                full_message += f" | {extra_info}"
            
            print(full_message, flush=True)
        except Exception:
            # Final fallback - basic print
            print(f"{level}: {message}")

def get_safe_logger(name: str = "NICEGOLD_ENTERPRISE") -> SafeLogger:
    """Get a safe logger instance"""
    return SafeLogger(name)

# Suppress optuna logging to prevent file handle issues
def suppress_optuna_logging():
    """Suppress optuna's internal logging to prevent errors"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Disable optuna's own loggers
        for logger_name in ['optuna', 'optuna.study', 'optuna.trial']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            
    except Exception:
        pass

# Initialize safe logging for the entire system
def initialize_safe_system_logging():
    """Initialize safe logging for the entire system"""
    try:
        # Suppress optuna logging first
        suppress_optuna_logging()
        
        # Configure root logger safely
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)  # Only warnings and errors
        
        return get_safe_logger()
        
    except Exception as e:
        print(f"⚠️ Safe logging initialization failed: {e}")
        return get_safe_logger()
