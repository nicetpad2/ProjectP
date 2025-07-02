#!/usr/bin/env python3
"""
🎨 SIMPLE BEAUTIFUL PROGRESS - NO RICH DEPENDENCIES
ระบบ Progress Tracking ที่สวยงามแต่ไม่ใช้ Rich complex components
เพื่อหลีกเลี่ยงปัญหา Text และ import errors
"""

import time
from datetime import datetime
from typing import Optional, Dict, Any
import logging


class SimpleBeautifulLogger:
    """Logger ที่สวยงามแต่ไม่ซับซ้อน - Fixed version to avoid closed file streams"""
    
    def __init__(self, name: str):
        self.name = name
        # Use a safer approach to avoid closed file stream issues
        self.logger = None
        try:
            # Try to get logger, but don't rely on it completely
            self.logger = logging.getLogger(name)
            # Test if logger is working
            self.logger.info("Test message")
        except (ValueError, OSError) as e:
            # If logger has issues, disable it
            print(f"⚠️ Logger issue detected: {e}. Using console-only mode.")
            self.logger = None
        
        self.current_step = None
        self.step_start_time = None
    
    def _safe_log(self, level: str, message: str):
        """Safely log message, fallback to console if logger fails"""
        try:
            if self.logger:
                if level == 'info':
                    self.logger.info(message)
                elif level == 'warning':
                    self.logger.warning(message)
                elif level == 'error':
                    self.logger.error(message)
        except (ValueError, OSError, AttributeError):
            # If logging fails, just continue - console output is already printed
            pass
    
    def start_step(self, step_num: int, step_name: str, description: str):
        """เริ่ม step ใหม่"""
        self.current_step = step_num
        self.step_start_time = time.time()
        
        print("╭" + "─" * 90 + "╮")
        print(f"│ ⚡ Starting Step {step_num} " + " " * (90 - len(f"⚡ Starting Step {step_num} ") - 1) + "│")
        print(f"│ 🚀 STEP {step_num}: {step_name.upper()}" + " " * (90 - len(f"🚀 STEP {step_num}: {step_name.upper()}") - 1) + "│")
        print(f"│ {description}" + " " * (90 - len(description) - 1) + "│")
        print("╰" + "─" * 90 + "╯")
        
        self._safe_log('info', f"🚀 Starting Step {step_num}: {step_name}")
    
    def complete_step(self, step_num: int, message: str):
        """完成 step"""
        duration = time.time() - self.step_start_time if self.step_start_time else 0
        
        print("╭" + "─" * 90 + "╮")
        print(f"│ 🎉 Step Completed " + " " * (90 - len("🎉 Step Completed ") - 1) + "│")
        print(f"│ ✅ STEP {step_num} COMPLETED" + " " * (90 - len(f"✅ STEP {step_num} COMPLETED") - 1) + "│")
        print(f"│ {message}" + " " * (90 - len(message) - 1) + "│")
        print(f"│ ⏱️ Duration: {duration:.2f}s" + " " * (90 - len(f"⏱️ Duration: {duration:.2f}s") - 1) + "│")
        print("╰" + "─" * 90 + "╯")
        
        self._safe_log('info', f"✅ Step {step_num} completed in {duration:.2f}s")
    
    def fail_step(self, step_num: int, message: str):
        """step ล้มเหลว"""
        duration = time.time() - self.step_start_time if self.step_start_time else 0
        
        print("╭" + "─" * 90 + "╮")
        print(f"│ 💥 Step Failed " + " " * (90 - len("💥 Step Failed ") - 1) + "│")
        print(f"│ ❌ STEP {step_num} FAILED" + " " * (90 - len(f"❌ STEP {step_num} FAILED") - 1) + "│")
        print(f"│ {message}" + " " * (90 - len(message) - 1) + "│")
        print(f"│ ⏱️ Duration: {duration:.2f}s" + " " * (90 - len(f"⏱️ Duration: {duration:.2f}s") - 1) + "│")
        print("╰" + "─" * 90 + "╯")
        
        self._safe_log('error', f"❌ Step {step_num} failed after {duration:.2f}s")
    
    def log_info(self, message: str):
        """Log info message"""
        print(f"ℹ️ {message}")
        self._safe_log('info', message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        print(f"⚠️ {message}")
        self._safe_log('warning', message)
    
    def log_error(self, message: str):
        """Log error message"""
        print(f"❌ {message}")
        self._safe_log('error', message)
    
    def log_success(self, message: str):
        """Log success message"""
        print(f"✅ {message}")
        self._safe_log('info', message)


class SimpleProgressTracker:
    """Simple Progress Tracker ที่ไม่ใช้ Rich"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.pipeline_start_time = None
        self.steps_completed = 0
        self.total_steps = 10  # Elliott Wave pipeline steps
    
    def start_pipeline(self):
        """เริ่ม pipeline"""
        self.pipeline_start_time = time.time()
        
        print("="*80)
        print("🚀 ELLIOTT WAVE PIPELINE STARTING")
        print("="*80)
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        self.logger.info("🚀 Elliott Wave Pipeline started")
    
    def update_progress(self, step: int, description: str):
        """อัพเดต progress"""
        progress = (step / self.total_steps) * 100
        
        # Simple progress bar
        bar_length = 50
        filled_length = int(bar_length * step // self.total_steps)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        print(f"\n📊 Progress: [{bar}] {progress:.1f}% - Step {step}/{self.total_steps}")
        print(f"🔄 {description}")
        
        self.logger.info(f"Progress: {progress:.1f}% - {description}")
    
    def complete_pipeline(self):
        """สิ้นสุด pipeline"""
        duration = time.time() - self.pipeline_start_time if self.pipeline_start_time else 0
        
        print("\n" + "="*80)
        print("🎉 ELLIOTT WAVE PIPELINE COMPLETED")
        print("="*80)
        print(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Total Duration: {duration:.2f}s")
        print("="*80)
        
        self.logger.info(f"🎉 Elliott Wave Pipeline completed in {duration:.2f}s")


def setup_simple_beautiful_logging(name: str, log_file: Optional[str] = None) -> SimpleBeautifulLogger:
    """Setup simple beautiful logging"""
    logger = SimpleBeautifulLogger(name)
    
    if log_file:
        # Setup file handler if needed
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)
    
    return logger


def create_simple_progress_tracker(logger: Optional[logging.Logger] = None) -> SimpleProgressTracker:
    """Create simple progress tracker"""
    return SimpleProgressTracker(logger)


# Aliases for backward compatibility
BeautifulProgressTracker = SimpleProgressTracker
setup_beautiful_logging = setup_simple_beautiful_logging


class PrintBasedBeautifulLogger:
    """Print-based beautiful logger that avoids all file stream conflicts"""
    
    def __init__(self, name="BeautifulLogger"):
        self.name = name
        self.start_time = None
        self.current_step = None
    
    def start_step(self, step_num, step_name, description=""):
        """Start a new step with beautiful formatting"""
        self.current_step = step_num
        self.start_time = time.time()
        
        print("╭──────────────────────────────────────────────────────────────────────────────────────────╮")
        print(f"│ ⚡ Starting Step {step_num:<77} │")
        print(f"│ 🚀 STEP {step_num}: {step_name.upper():<73} │")
        if description:
            print(f"│ {description:<88} │")
        print("╰──────────────────────────────────────────────────────────────────────────────────────────╯")
    
    def complete_step(self, step_num, message=""):
        """Complete a step with timing"""
        if self.start_time:
            duration = time.time() - self.start_time
        else:
            duration = 0.0
        
        print("╭──────────────────────────────────────────────────────────────────────────────────────────╮")
        print("│ 🎉 Step Completed                                                                         │")
        print(f"│ ✅ STEP {step_num} COMPLETED{' '*71} │")
        if message:
            print(f"│ {message:<88} │")
        print(f"│ ⏱️ Duration: {duration:.2f}s{' '*74} │")
        print("╰──────────────────────────────────────────────────────────────────────────────────────────╯")
    
    def log_info(self, message):
        """Log info message"""
        print(f"ℹ️ {message}")
    
    def log_warning(self, message):
        """Log warning message"""
        print(f"⚠️ {message}")
    
    def log_error(self, message):
        """Log error message"""
        print(f"❌ {message}")


def setup_print_based_beautiful_logging(name="BeautifulLogger"):
    """Setup print-based beautiful logging that won't conflict with file streams"""
    return PrintBasedBeautifulLogger(name)


# Additional aliases
setup_robust_beautiful_logging = setup_print_based_beautiful_logging
