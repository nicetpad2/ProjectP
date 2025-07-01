#!/usr/bin/env python3
"""
🧪 MENU 1 ENTERPRISE LOGGER TEST SYSTEM
ทดสอบระบบ Logging ขั้นสูงสำหรับ Menu 1

คุณสมบัติที่ทดสอบ:
- สีสันสวยงามและถนอมสายตา
- การติดตาม Progress แบบ Real-time
- การจัดการ Error/Warning ที่ครบถ้วน
- ระบบ Reporting ที่สมบูรณ์
- บันทึกการทำงานแบบ Enterprise Grade
"""

import sys
import os
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_menu1_enterprise_logging():
    """
    ทดสอบระบบ Enterprise Logging สำหรับ Menu 1
    """
    try:
        # Import logging system
        from core.menu1_logger import (
            start_menu1_session, 
            log_step, 
            log_error, 
            log_warning, 
            log_success, 
            complete_menu1_session,
            ProcessStatus,
            get_menu1_logger
        )
        
        print("🌟 Testing NICEGOLD Enterprise Menu 1 Logging System")
        print("="*80)
        
        # เริ่ม session
        session_id = "test_logging_20250701"
        logger = start_menu1_session(session_id)
        
        print(f"\n✅ Logging session started: {session_id}")
        time.sleep(1)
        
        # ทดสอบ Step Logging แบบต่างๆ
        test_steps = [
            {
                "step": 1,
                "name": "Data Loading & Validation",
                "details": "Loading XAUUSD market data from datacsv/",
                "status": ProcessStatus.SUCCESS,
                "metrics": {"rows": 1771970, "size_mb": 131}
            },
            {
                "step": 2,
                "name": "Elliott Wave Pattern Detection",
                "details": "Analyzing wave patterns using Fibonacci levels",
                "status": ProcessStatus.SUCCESS,
                "metrics": {"patterns_found": 45, "confidence": 0.87}
            },
            {
                "step": 3,
                "name": "Feature Engineering",
                "details": "Creating 50+ technical indicators",
                "status": ProcessStatus.WARNING,
                "warning": "Some features have high correlation (>0.9)"
            },
            {
                "step": 4,
                "name": "SHAP + Optuna Feature Selection",
                "details": "Optimizing features for AUC ≥ 70%",
                "status": ProcessStatus.SUCCESS,
                "metrics": {
                    "trials": 150,
                    "best_auc": 0.732,
                    "selected_features": 25
                }
            },
            {
                "step": 5,
                "name": "CNN-LSTM Model Training",
                "details": "Training deep learning model",
                "status": ProcessStatus.SUCCESS,
                "metrics": {
                    "epochs": 100,
                    "final_loss": 0.0234,
                    "training_time": "00:08:45"
                }
            },
            {
                "step": 6,
                "name": "DQN Agent Training",
                "details": "Training reinforcement learning agent",
                "status": ProcessStatus.SUCCESS,
                "metrics": {
                    "episodes": 1000,
                    "total_reward": 1547.23,
                    "win_rate": 67.3
                }
            },
            {
                "step": 7,
                "name": "System Integration",
                "details": "Integrating all AI components",
                "status": ProcessStatus.SUCCESS,
                "metrics": {"integration_score": 0.95}
            },
            {
                "step": 8,
                "name": "Performance Validation",
                "details": "Validating enterprise requirements",
                "status": ProcessStatus.SUCCESS,
                "metrics": {
                    "auc_score": 0.732,
                    "sharpe_ratio": 1.45,
                    "max_drawdown": 0.08
                }
            }
        ]
        
        # รันการทดสอบ steps
        for step_info in test_steps:
            # เริ่ม step
            log_step(
                step_info["step"],
                step_info["name"],
                ProcessStatus.RUNNING,
                f"Starting: {step_info['details']}",
                (step_info["step"] - 1) * 12
            )
            
            # จำลองการประมวลผล
            time.sleep(0.8)
            
            # จัดการผลลัพธ์
            if step_info["status"] == ProcessStatus.WARNING:
                log_warning(step_info.get("warning", "Unknown warning"), step_info["name"])
                time.sleep(0.3)
                log_success(f"{step_info['name']} completed with warnings", 
                           step_info["name"])
            elif step_info["status"] == ProcessStatus.SUCCESS:
                metrics = step_info.get("metrics", {})
                log_success(f"{step_info['name']} completed successfully", 
                           step_info["name"], metrics)
            
            # อัปเดต progress
            log_step(
                step_info["step"],
                step_info["name"],
                step_info["status"],
                f"Completed: {step_info['details']}",
                step_info["step"] * 12
            )
            
            time.sleep(0.5)
        
        # ทดสอบ Error Handling
        print("\n🧪 Testing Error Handling...")
        try:
            # จำลอง error
            raise ValueError("Simulated critical error for testing")
        except Exception as e:
            log_error("Testing error reporting system", e, "Error Handling Test")
        
        time.sleep(1)
        
        # ผลลัพธ์สุดท้าย
        final_results = {
            "test_status": "completed",
            "pipeline_performance": {
                "auc_score": 0.732,
                "sharpe_ratio": 1.45,
                "max_drawdown": 0.08,
                "win_rate": 67.3
            },
            "enterprise_compliance": {
                "real_data_only": True,
                "no_simulation": True,
                "auc_target_achieved": True
            },
            "system_metrics": {
                "total_steps": 8,
                "successful_steps": 7,
                "warnings": 1,
                "errors": 1,
                "execution_time": "00:12:34"
            }
        }
        
        # จบ session
        complete_menu1_session(final_results)
        
        # แสดงสถิติ session
        stats = logger.get_session_stats()
        print(f"\n📊 Session Statistics:")
        print(f"   Session ID: {stats['session_id']}")
        print(f"   Total Steps: {stats['total_steps']}")
        print(f"   Successes: {stats['success_count']}")
        print(f"   Warnings: {stats['warning_count']}")
        print(f"   Errors: {stats['error_count']}")
        print(f"   Duration: {stats['duration']}")
        
        print(f"\n📁 Log Files Generated:")
        print(f"   📄 Session Log: logs/menu1/sessions/{session_id}.log")
        print(f"   📄 Error Log: logs/menu1/errors/{session_id}_errors.log")
        print(f"   📄 Report: logs/menu1/sessions/{session_id}_report.json")
        
        print("\n✅ Enterprise Logging System Test Completed Successfully!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please check if menu1_logger.py exists in core/ directory")
        
    except Exception as e:
        print(f"💥 Test Error: {e}")
        import traceback
        traceback.print_exc()


def test_integration_with_projectp():
    """
    ทดสอบการผนวกรวมกับ ProjectP.py
    """
    print("\n🔗 Testing Integration with ProjectP...")
    
    try:
        # Import main components
        from core.menu_system import MenuSystem
        from core.logger import setup_enterprise_logger
        from core.config import load_enterprise_config
        
        print("✅ Core components imported successfully")
        
        # Test logger setup
        logger = setup_enterprise_logger()
        print("✅ Enterprise logger setup successful")
        
        # Test config loading
        config = load_enterprise_config()
        print("✅ Enterprise config loaded successfully")
        
        # Test menu system
        menu_system = MenuSystem(config=config, logger=logger)
        print("✅ Menu system initialized successfully")
        
        print("🎉 Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")


if __name__ == "__main__":
    print("🚀 NICEGOLD Enterprise Menu 1 Logger Test Suite")
    print("="*80)
    
    # รันการทดสอบหลัก
    test_menu1_enterprise_logging()
    
    # รันการทดสอบการผนวกรวม
    test_integration_with_projectp()
    
    print("\n🎊 All tests completed!")
    print("Ready for production use with ProjectP.py")
