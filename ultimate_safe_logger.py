#!/usr/bin/env python3
"""
🔒 ULTIMATE SAFE LOGGER - NICEGOLD ENTERPRISE
Logger ที่ปลอดภัยจาก I/O errors อย่างสมบูรณ์แบบ
"""

import logging
import sys
import io
from datetime import datetime
from contextlib import contextmanager

class UltimateSafeLogger:
    """Logger ที่ปลอดภัยอย่างสมบูรณ์แบบ"""
    
    def __init__(self, name="NICEGOLD"):
        self.name = name
        self.buffer = io.StringIO()
        
    def _safe_write(self, message, level="INFO"):
        """เขียน log อย่างปลอดภัย 100%"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_msg = f"[{timestamp}] {level}: {message}"
            
            # เขียนไปยัง buffer
            self.buffer.write(formatted_msg + "\n")
            
            # พยายามแสดงผลหน้าจอ
            try:
                print(formatted_msg)
                sys.stdout.flush()
            except:
                pass
                
        except Exception:
            # หากทุกอย่างล้มเหลว ใช้ print ธรรมดา
            try:
                print(f"[LOG] {message}")
            except:
                pass
    
    def info(self, message):
        """Log level INFO"""
        self._safe_write(message, "INFO")
        
    def error(self, message):
        """Log level ERROR"""
        self._safe_write(message, "ERROR")
        
    def warning(self, message):
        """Log level WARNING"""
        self._safe_write(message, "WARNING")
        
    def debug(self, message):
        """Log level DEBUG"""
        self._safe_write(message, "DEBUG")
        
    def get_logs(self):
        """ดึง logs ทั้งหมด"""
        try:
            return self.buffer.getvalue()
        except:
            return "Logs unavailable"

# Global instance
ultimate_logger = UltimateSafeLogger("NICEGOLD_ULTIMATE")

def get_ultimate_logger(name="NICEGOLD"):
    """ได้ logger ที่ปลอดภัยอย่างสมบูรณ์แบบ"""
    return UltimateSafeLogger(name)

@contextmanager
def safe_logging_context():
    """Context สำหรับ logging ที่ปลอดภัย"""
    logger = get_ultimate_logger()
    try:
        yield logger
    except Exception as e:
        try:
            print(f"[SAFE_LOG_ERROR] {e}")
        except:
            pass
