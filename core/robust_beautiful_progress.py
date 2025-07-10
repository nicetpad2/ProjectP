#!/usr/bin/env python3
"""
🎨 ROBUST BEAUTIFUL PROGRESS LOGGER
Completely avoids standard logging to prevent closed file stream errors
"""

import time
from typing import Optional
from core.unified_enterprise_logger import get_unified_logger, ElliottWaveStep, Menu1Step, LogLevel, ProcessStatus


class RobustBeautifulLogger:
    """
    Ultra-robust logger that never fails due to closed streams
    Uses only console output with beautiful formatting
    """
    
    def __init__(self, name: str = "RobustLogger"):
        self.name = name
        self.start_time = None
        self.step_start_time = None
        self.current_step = None
    
    def _create_box(self, title: str, content: str = "", emoji: str = "🎯") -> str:
        """Create a beautiful box for important messages"""
        width = 94
        top = "╭" + "─" * (width - 2) + "╮"
        bottom = "╰" + "─" * (width - 2) + "╯"
        
        # Title line
        title_line = f"│ {emoji} {title}"
        title_line += " " * (width - len(title_line) - 1) + "│"
        
        lines = [top, title_line]
        
        if content:
            content_lines = content.split('\n')
            for line in content_lines:
                content_line = f"│ {line}"
                content_line += " " * (width - len(content_line) - 1) + "│"
                lines.append(content_line)
        
        lines.append(bottom)
        return '\n'.join(lines)
    
    def start_step(self, step_num: int, step_name: str, description: str = ""):
        """Start a new step with beautiful formatting"""
        self.current_step = step_num
        self.step_start_time = time.time()
        
        title = f"⚡ Starting Step {step_num}"
        content = f"🚀 STEP {step_num}: {step_name.upper()}"
        if description:
            content += f"\n{description}"
        
        box = self._create_box(title, content)
        print(box)
    
    def complete_step(self, step_num: int, success_message: str = ""):
        """Complete a step with beautiful formatting"""
        duration = time.time() - self.step_start_time if self.step_start_time else 0
        
        title = "🎉 Step Completed"
        content = f"✅ STEP {step_num} COMPLETED"
        if success_message:
            content += f"\n{success_message}"
        content += f"\n⏱️ Duration: {duration:.2f}s"
        
        box = self._create_box(title, content)
        print(box)
    
    def log_progress(self, message: str, emoji: str = "⚡"):
        """Log progress with emoji"""
        print(f"{emoji} {message}")
    
    def log_info(self, message: str):
        """Log info message"""
        print(f"ℹ️ {message}")
    
    def log_success(self, message: str):
        """Log success message"""
        print(f"✅ {message}")
    
    def log_warning(self, message: str):
        """Log warning message"""
        print(f"⚠️ {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        print(f"❌ {message}")
    
    def log_status(self, status: str, message: str):
        """Log status with custom prefix"""
        print(f"{status} {message}")
    
    def step_start(self, step_num: int, step_name: str, description: str = ""):
        """Alias for start_step to match expected interface"""
        self.start_step(step_num, step_name, description)
    
    def step_complete(self, step_num: int, step_name: str, duration: float, details: dict = None):
        """Complete step with detailed information"""
        content = f"✅ {step_name.upper()} COMPLETED"
        if details:
            for key, value in details.items():
                content += f"\n📊 {key}: {value}"
        content += f"\n⏱️ Duration: {duration:.2f}s"
        
        box = self._create_box("🎉 Step Completed", content)
        print(box)
    
    def step_error(self, step_num: int, step_name: str, error_msg: str):
        """Log step error"""
        content = f"❌ {step_name.upper()} FAILED"
        content += f"\n💥 Error: {error_msg}"
        
        box = self._create_box("⚠️ Step Failed", content, "❌")
        print(box)
    
    def info(self, message: str, details: dict = None):
        """Enhanced info logging with optional details"""
        print(f"ℹ️ {message}")
        if details:
            for key, value in details.items():
                print(f"   📊 {key}: {value}")
    
    def success(self, message: str, details: dict = None):
        """Enhanced success logging with optional details"""
        print(f"✅ {message}")
        if details:
            for key, value in details.items():
                print(f"   🎯 {key}: {value}")
    
    def error(self, message: str, details: dict = None):
        """Enhanced error logging with optional details"""
        print(f"❌ {message}")
        if details:
            for key, value in details.items():
                print(f"   ⚠️ {key}: {value}")
    
    def critical(self, message: str, details: dict = None):
        """Enhanced critical logging with optional details"""
        content = f"🚨 CRITICAL ERROR: {message}"
        if details:
            for key, value in details.items():
                content += f"\n💥 {key}: {value}"
        
        box = self._create_box("🚨 CRITICAL ERROR", content, "🚨")
        print(box)

def setup_robust_beautiful_logging(name: str = "RobustLogger") -> RobustBeautifulLogger:
    """
    Setup robust beautiful logging that never fails
    """
    return RobustBeautifulLogger(name)

# For backward compatibility
setup_simple_beautiful_logging = setup_robust_beautiful_logging
SimpleBeautifulLogger = RobustBeautifulLogger
