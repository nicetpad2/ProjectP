#!/usr/bin/env python3
"""
🎯 MENU 1 ENTERPRISE LOGGER DEMO
ตัวอย่างการใช้ระบบ Enterprise Logger สำหรับ Menu 1
รันเพื่อดูระบบ logging ที่สวยงามและครบถ้วน
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import logging system
from core.menu1_logger import (
    start_menu1_session,
    log_step,
    log_error,
    log_warning,
    log_success,
    complete_menu1_session,
    ProcessStatus
)


def demo_menu1_logging():
    """
    สาธิตระบบ logging สำหรับ Menu 1
    """
    print("🌟 NICEGOLD ENTERPRISE - Menu 1 Logger Demo")
    print("="*80)
    
    # เริ่ม Menu 1 Session
    logger = start_menu1_session("demo_20250701")
    
    # จำลองการทำงานของ Menu 1 Pipeline
    steps = [
        ("Data Loading & Validation", "Loading XAUUSD market data"),
        ("Elliott Wave Pattern Detection", "Analyzing wave patterns"),
        ("Advanced Feature Engineering", "Creating 50+ indicators"),
        ("SHAP + Optuna Feature Selection", "Optimizing features"),
        ("CNN-LSTM Model Training", "Training deep learning model"),
        ("DQN Agent Training", "Training RL agent"),
        ("System Integration", "Integrating all components"),
        ("Performance Analysis", "Analyzing final metrics"),
        ("Quality Validation", "Validating enterprise requirements"),
        ("Results Export", "Exporting comprehensive reports")
    ]
    
    # รัน pipeline steps พร้อม progress tracking
    success_count = 0
    for i, (step_name, details) in enumerate(steps, 1):
        try:
            # เริ่ม step
            log_step(i, step_name, ProcessStatus.RUNNING,
                     details, progress=(i-1)*10)
            
            # จำลองการประมวลผล
            time.sleep(1)  # แค่เพื่อ demo
            
            # จำลองผลลัพธ์
            if i == 3:  # Feature Engineering - แสดง warning
                log_warning("Some features have high correlation", step_name)
                time.sleep(0.5)
            elif i == 4:  # Feature Selection - แสดงผลสำเร็จ
                metrics = {
                    "selected_features": 25,
                    "auc_score": 0.732,
                    "optimization_trials": 150
                }
                log_success("Feature selection completed with AUC=0.732",
                           step_name, metrics)
                success_count += 1
            else:
                log_success(f"{step_name} completed successfully", step_name)
                success_count += 1
                
            # อัปเดต progress
            log_step(i, step_name, ProcessStatus.SUCCESS,
                     f"Completed: {details}", progress=i*10)
            
        except Exception as e:
            log_error(f"Step {i} failed: {step_name}", e, step_name)
    
    # จำลองผลลัพธ์สุดท้าย
    final_results = {
        "pipeline_status": "completed",
        "auc_score": 0.732,
        "total_features": 25,
        "sharpe_ratio": 1.45,
        "max_drawdown": 0.08,
        "win_rate": 67.3,
        "enterprise_compliant": True,
        "execution_time": "00:12:34"
    }
    
    # จบ session
    complete_menu1_session(final_results)
    
    print("\n🎉 Demo completed! Check logs/menu1/sessions/ for detailed logs.")

if __name__ == "__main__":
    demo_menu1_logging()
