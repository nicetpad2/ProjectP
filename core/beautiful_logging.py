#!/usr/bin/env python3
"""
📝 BEAUTIFUL LOGGING SYSTEM
ระบบ Logging แบบสวยงามและรายละเอียดสำหรับ Elliott Wave System

Features:
- Colorful and formatted log output
- Rich console integration
- Structured error reporting
- Performance metrics logging
- Real-time log streaming
- Enterprise-grade error tracking
"""

import logging
import sys
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Try to import rich components
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.syntax import Syntax
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("💡 Installing rich for beautiful logging...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.syntax import Syntax
    from rich import box
    from rich.align import Align


class LogLevel(Enum):
    """ระดับของ Log"""
    DEBUG = "🔍 DEBUG"
    INFO = "ℹ️ INFO"
    SUCCESS = "✅ SUCCESS"
    WARNING = "⚠️ WARNING"
    ERROR = "❌ ERROR"
    CRITICAL = "💥 CRITICAL"


@dataclass
class LogEntry:
    """รายการ Log"""
    timestamp: str
    level: LogLevel
    step_id: Optional[int]
    step_name: str
    message: str
    details: Dict[str, Any]
    duration: Optional[float] = None
    error_traceback: Optional[str] = None


class BeautifulLogger:
    """Logger สวยงามสำหรับ Elliott Wave System"""
    
    def __init__(self, name: str = "ElliottWave", log_file: Optional[str] = None):
        self.name = name
        self.console = Console()
        self.log_entries: List[LogEntry] = []
        self.step_timers: Dict[int, float] = {}
        self.current_step_id: Optional[int] = None
        self.current_step_name: str = ""
        
        # Setup file logging
        self.log_file = log_file or f"logs/elliott_wave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._setup_file_logging()
        
        # Setup rich logging handler
        self._setup_rich_logging()
        
        # Performance tracking
        self.start_time = time.time()
        self.performance_metrics: Dict[str, Any] = {}
    
    def _setup_file_logging(self):
        """ตั้งค่า File Logging"""
        log_path = Path(self.log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        # Standard file logger
        self.file_logger = logging.getLogger(f"{self.name}_file")
        self.file_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
    
    def _setup_rich_logging(self):
        """ตั้งค่า Rich Logging"""
        # Rich console logger
        self.rich_logger = logging.getLogger(f"{self.name}_rich")
        self.rich_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        for handler in self.rich_logger.handlers[:]:
            self.rich_logger.removeHandler(handler)
        
        # Rich handler
        rich_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True
        )
        self.rich_logger.addHandler(rich_handler)
    
    def start_step(self, step_id: int, step_name: str, description: str = ""):
        """เริ่มขั้นตอนใหม่"""
        self.current_step_id = step_id
        self.current_step_name = step_name
        self.step_timers[step_id] = time.time()
        
        # Beautiful step header
        step_text = Text()
        step_text.append(f"🚀 STEP {step_id}: {step_name.upper()}", style="bold cyan")
        if description:
            step_text.append(f"\n{description}", style="italic white")
        
        panel = Panel(
            step_text,
            title=f"⚡ Starting Step {step_id}",
            border_style="bright_cyan",
            box=box.ROUNDED
        )
        
        self.console.print(panel)
        self._log_entry(LogLevel.INFO, f"Started Step {step_id}: {step_name}", {"description": description})
    
    def log_success(self, message: str, details: Dict[str, Any] = None):
        """บันทึก Success Log"""
        self.console.print(f"✅ {message}", style="bold green")
        self._log_entry(LogLevel.SUCCESS, message, details or {})
    
    def log_info(self, message: str, details: Dict[str, Any] = None):
        """บันทึก Info Log"""
        self.console.print(f"ℹ️ {message}", style="blue")
        self._log_entry(LogLevel.INFO, message, details or {})
    
    def log_warning(self, message: str, details: Dict[str, Any] = None):
        """บันทึก Warning Log"""
        self.console.print(f"⚠️ {message}", style="bold yellow")
        self._log_entry(LogLevel.WARNING, message, details or {})
    
    def log_error(self, message: str, error: Exception = None, details: Dict[str, Any] = None):
        """บันทึก Error Log"""
        error_text = Text(f"❌ {message}", style="bold red")
        self.console.print(error_text)
        
        error_details = details or {}
        error_traceback = None
        
        if error:
            error_details.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
            error_traceback = traceback.format_exc()
            
            # Display beautiful error panel
            self._display_error_panel(message, error, error_traceback)
        
        self._log_entry(LogLevel.ERROR, message, error_details, error_traceback=error_traceback)
    
    def log_critical(self, message: str, error: Exception = None, details: Dict[str, Any] = None):
        """บันทึก Critical Log"""
        critical_text = Text(f"💥 CRITICAL: {message}", style="bold red on white")
        self.console.print(critical_text)
        
        error_details = details or {}
        error_traceback = None
        
        if error:
            error_details.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
            error_traceback = traceback.format_exc()
        
        self._log_entry(LogLevel.CRITICAL, message, error_details, error_traceback=error_traceback)
    
    def _display_error_panel(self, message: str, error: Exception, error_traceback: str):
        """แสดง Error Panel สวยงาม"""
        error_content = Text()
        error_content.append(f"Error Type: {type(error).__name__}\n", style="bold red")
        error_content.append(f"Message: {str(error)}\n", style="red")
        
        if self.current_step_id:
            error_content.append(f"Step: {self.current_step_id} - {self.current_step_name}\n", style="yellow")
        
        # Truncated traceback for display
        if error_traceback:
            lines = error_traceback.split('\n')
            if len(lines) > 10:
                traceback_preview = '\n'.join(lines[-10:])
                error_content.append(f"\nRecent Traceback:\n{traceback_preview}", style="dim red")
            else:
                error_content.append(f"\nTraceback:\n{error_traceback}", style="dim red")
        
        error_panel = Panel(
            error_content,
            title="💥 ERROR DETAILS",
            border_style="bright_red",
            box=box.DOUBLE
        )
        
        self.console.print(error_panel)
    
    def complete_step(self, success: bool = True, summary: str = ""):
        """เสร็จสิ้นขั้นตอน"""
        if self.current_step_id is None:
            return
        
        # Calculate duration
        duration = time.time() - self.step_timers.get(self.current_step_id, time.time())
        
        if success:
            completion_text = Text(f"✅ STEP {self.current_step_id} COMPLETED", style="bold green")
            if summary:
                completion_text.append(f"\n{summary}", style="italic green")
            completion_text.append(f"\n⏱️ Duration: {duration:.2f}s", style="dim white")
            
            panel = Panel(
                completion_text,
                title="🎉 Step Completed",
                border_style="bright_green",
                box=box.ROUNDED
            )
        else:
            completion_text = Text(f"❌ STEP {self.current_step_id} FAILED", style="bold red")
            if summary:
                completion_text.append(f"\n{summary}", style="italic red")
            completion_text.append(f"\n⏱️ Duration: {duration:.2f}s", style="dim white")
            
            panel = Panel(
                completion_text,
                title="💥 Step Failed",
                border_style="bright_red",
                box=box.ROUNDED
            )
        
        self.console.print(panel)
        
        # Log completion
        status = "completed" if success else "failed"
        self._log_entry(
            LogLevel.SUCCESS if success else LogLevel.ERROR,
            f"Step {self.current_step_id} {status}",
            {"summary": summary, "duration": duration},
            duration=duration
        )
        
        # Clear current step
        self.current_step_id = None
        self.current_step_name = ""
    
    def log_performance(self, metric_name: str, value: Any, unit: str = ""):
        """บันทึก Performance Metrics"""
        self.performance_metrics[metric_name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        }
        
        self.console.print(f"📊 {metric_name}: {value} {unit}", style="bold blue")
        self._log_entry(LogLevel.INFO, f"Performance metric: {metric_name}", {"value": value, "unit": unit})
    
    def display_performance_summary(self):
        """แสดงสรุป Performance"""
        if not self.performance_metrics:
            return
        
        table = Table(
            title="📊 Performance Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Metric", style="bold white")
        table.add_column("Value", style="green")
        table.add_column("Unit", style="dim white")
        
        for metric_name, metric_data in self.performance_metrics.items():
            table.add_row(
                metric_name,
                str(metric_data["value"]),
                metric_data["unit"]
            )
        
        self.console.print(table)
    
    def _log_entry(self, level: LogLevel, message: str, details: Dict[str, Any], 
                   duration: Optional[float] = None, error_traceback: Optional[str] = None):
        """บันทึก Log Entry"""
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            step_id=self.current_step_id,
            step_name=self.current_step_name,
            message=message,
            details=details,
            duration=duration,
            error_traceback=error_traceback
        )
        
        self.log_entries.append(entry)
        
        # Log to file
        log_message = f"Step {entry.step_id or 'N/A'} | {entry.message}"
        if details:
            log_message += f" | Details: {json.dumps(details, default=str)}"
        
        if level == LogLevel.DEBUG:
            self.file_logger.debug(log_message)
        elif level == LogLevel.INFO or level == LogLevel.SUCCESS:
            self.file_logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.file_logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.file_logger.error(log_message)
            if error_traceback:
                self.file_logger.error(f"Traceback: {error_traceback}")
        elif level == LogLevel.CRITICAL:
            self.file_logger.critical(log_message)
            if error_traceback:
                self.file_logger.critical(f"Traceback: {error_traceback}")
    
    def save_log_summary(self, output_path: Optional[str] = None):
        """บันทึกสรุป Log"""
        if not output_path:
            output_path = f"logs/elliott_wave_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_path = Path(output_path)
        log_path.parent.mkdir(exist_ok=True)
        
        summary = {
            "session_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration": time.time() - self.start_time,
                "total_entries": len(self.log_entries)
            },
            "performance_metrics": self.performance_metrics,
            "log_entries": [asdict(entry) for entry in self.log_entries],
            "step_summary": self._get_step_summary()
        }
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.console.print(f"💾 Log summary saved to: {log_path}", style="green")
    
    def _get_step_summary(self) -> Dict[str, Any]:
        """ได้สรุปขั้นตอน"""
        steps = {}
        for entry in self.log_entries:
            if entry.step_id:
                if entry.step_id not in steps:
                    steps[entry.step_id] = {
                        "step_name": entry.step_name,
                        "logs": [],
                        "errors": 0,
                        "warnings": 0,
                        "success": 0,
                        "total_duration": 0
                    }
                
                steps[entry.step_id]["logs"].append({
                    "level": entry.level.value,
                    "message": entry.message,
                    "timestamp": entry.timestamp
                })
                
                if entry.level == LogLevel.ERROR or entry.level == LogLevel.CRITICAL:
                    steps[entry.step_id]["errors"] += 1
                elif entry.level == LogLevel.WARNING:
                    steps[entry.step_id]["warnings"] += 1
                elif entry.level == LogLevel.SUCCESS:
                    steps[entry.step_id]["success"] += 1
                
                if entry.duration:
                    steps[entry.step_id]["total_duration"] += entry.duration
        
        return steps
    
    def display_final_summary(self):
        """แสดงสรุปสุดท้าย"""
        total_time = time.time() - self.start_time
        
        # Count log levels
        level_counts = {}
        for entry in self.log_entries:
            level_name = entry.level.name
            level_counts[level_name] = level_counts.get(level_name, 0) + 1
        
        # Create summary table
        summary_table = Table(
            title="📝 Logging Session Summary",
            box=box.DOUBLE,
            show_header=True,
            header_style="bold cyan"
        )
        
        summary_table.add_column("Metric", style="bold white")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Duration", f"{total_time:.2f}s")
        summary_table.add_row("Total Log Entries", str(len(self.log_entries)))
        summary_table.add_row("Log File", str(self.log_file))
        
        for level, count in level_counts.items():
            emoji = {
                "SUCCESS": "✅",
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "CRITICAL": "💥",
                "DEBUG": "🔍"
            }.get(level, "📝")
            summary_table.add_row(f"{emoji} {level}", str(count))
        
        self.console.print(summary_table)


# Global logger instance
_beautiful_logger: Optional[BeautifulLogger] = None


def get_beautiful_logger() -> BeautifulLogger:
    """ได้ Beautiful Logger แบบ Singleton"""
    global _beautiful_logger
    if _beautiful_logger is None:
        _beautiful_logger = BeautifulLogger()
    return _beautiful_logger


def setup_beautiful_logging(name: str = "ElliottWave", log_file: Optional[str] = None) -> BeautifulLogger:
    """ตั้งค่า Beautiful Logging"""
    global _beautiful_logger
    _beautiful_logger = BeautifulLogger(name, log_file)
    return _beautiful_logger


# Convenience functions
def log_success(message: str, details: Dict[str, Any] = None):
    """Log Success Message"""
    logger = get_beautiful_logger()
    logger.log_success(message, details)


def log_info(message: str, details: Dict[str, Any] = None):
    """Log Info Message"""
    logger = get_beautiful_logger()
    logger.log_info(message, details)


def log_warning(message: str, details: Dict[str, Any] = None):
    """Log Warning Message"""
    logger = get_beautiful_logger()
    logger.log_warning(message, details)


def log_error(message: str, error: Exception = None, details: Dict[str, Any] = None):
    """Log Error Message"""
    logger = get_beautiful_logger()
    logger.log_error(message, error, details)


def start_step(step_id: int, step_name: str, description: str = ""):
    """Start Pipeline Step"""
    logger = get_beautiful_logger()
    logger.start_step(step_id, step_name, description)


def complete_step(success: bool = True, summary: str = ""):
    """Complete Pipeline Step"""
    logger = get_beautiful_logger()
    logger.complete_step(success, summary)


# Demo
if __name__ == "__main__":
    # Demo the beautiful logger
    logger = setup_beautiful_logging("Demo")
    
    logger.start_step(1, "Data Loading", "Loading market data from CSV files")
    logger.log_info("Scanning for CSV files...")
    logger.log_success("Found 2 CSV files", {"files": ["M1.csv", "M15.csv"]})
    logger.log_performance("Data Loading Speed", 1234.56, "rows/sec")
    
    try:
        # Simulate an error
        raise ValueError("Demo error for testing")
    except Exception as e:
        logger.log_error("Failed to process data", e)
    
    logger.log_warning("Data quality issue detected")
    logger.complete_step(False, "Step failed due to data quality issues")
    
    logger.display_performance_summary()
    logger.display_final_summary()
    logger.save_log_summary()
