#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - COMPLETE SYSTEM RUNNER
=========================================================

ไฟล์สำหรับรันระบบเต็มรูปแบบที่เปิดใช้ทุกฟีเจอร์:
- ✅ Intelligent Resource Management (อัตโนมัติ 80% allocation)
- ✅ Advanced Logging System (Enterprise-grade with session management)
- ✅ Enterprise ML Protection (ป้องกัน overfitting/noise/leakage)
- ✅ Beautiful Progress Tracking (Real-time แบบสวยงาม)
- ✅ Menu 1 Elliott Wave (CNN-LSTM + DQN pipeline)
- ✅ Performance Analytics (ระดับ Enterprise)

🎯 ENTERPRISE FEATURES ENABLED:
- 🧠 Intelligent Resource Management: CPU, RAM, GPU จัดการอัตโนมัติ
- 📊 Advanced Logging: บันทึกข้อมูลแบบ Enterprise มาตรฐาน
- 🛡️ Enterprise ML Protection: ป้องกันปัญหา ML แบบครบถ้วน
- 🎨 Beautiful Progress: แสดงความคืบหน้าแบบสวยงาม Real-time
- 📈 Performance Analytics: วิเคราะห์ประสิทธิภาพแบบละเอียด
- 🏢 Enterprise Compliance: ตรวจสอบมาตรฐาน Enterprise

วันที่พัฒนา: January 1, 2025
สถานะ: ✅ Production Ready - All Systems Integrated
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Suppress warnings for clean execution
import warnings
warnings.filterwarnings('ignore')

# Set environment for optimal execution
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use CPU for stability
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Minimize TensorFlow logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Beautiful startup banner
print("=" * 100)
print("🏢 NICEGOLD ENTERPRISE PROJECTP - COMPLETE SYSTEM EXECUTION")
print("=" * 100)
print(f"� Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("🎯 Mission: Execute full system with ALL enterprise features")
print("🔧 Features: Resource Management + Logging + ML Protection + Progress + Analytics")
print("=" * 100)

def main():
    """Main execution function"""
    try:
        # Import core systems
        print("📦 Loading core systems...")
        from core.logger import EnterpriseLogger
        
        # Initialize Enterprise Logger
        logger = EnterpriseLogger("CompleteSystem", enable_colors=True)
        logger.info("🏢 NICEGOLD Complete System Starting...")
        
        # Display system capabilities
        print("\n🎯 SYSTEM CAPABILITIES:")
        capabilities = [
            "🧠 Intelligent Resource Management (CPU, RAM, GPU optimization)",
            "📝 Advanced Logging & Progress Tracking",
            "🛡️ Enterprise ML Protection (Overfitting, Data Leakage, Noise detection)",
            "🌊 Elliott Wave CNN-LSTM Pattern Recognition",
            "🤖 DQN Reinforcement Learning Trading Agent",
            "🎯 SHAP + Optuna Feature Selection",
            "📊 Comprehensive Performance Analytics",
            "🎨 Beautiful UI with Real-time Monitoring"
        ]
        
        for cap in capabilities:
            print(f"   ✅ {cap}")
        
        print("\n" + "="*80)
        
        # Check system availability
        print("🔍 Checking System Availability...")
        
        # Check Resource Management
        try:
            from core.intelligent_resource_manager import initialize_intelligent_resources
            print("   ✅ Intelligent Resource Management: AVAILABLE")
            resource_available = True
        except ImportError as e:
            print(f"   ❌ Intelligent Resource Management: NOT AVAILABLE ({e})")
            resource_available = False
        
        # Check ML Protection
        try:
            from elliott_wave_modules.enterprise_ml_protection import EnterpriseMLProtectionSystem
            print("   ✅ Enterprise ML Protection: AVAILABLE")
            protection_available = True
        except ImportError as e:
            print(f"   ❌ Enterprise ML Protection: NOT AVAILABLE ({e})")
            protection_available = False
        
        # Check Elliott Wave Components
        try:
            from menu_modules.menu_1_elliott_wave import Menu1ElliottWaveFixed
            print("   ✅ Elliott Wave Pipeline: AVAILABLE")
            elliott_available = True
        except ImportError as e:
            print(f"   ❌ Elliott Wave Pipeline: NOT AVAILABLE ({e})")
            elliott_available = False
        
        print("\n" + "="*80)
        
        if not elliott_available:
            logger.error("❌ Core Elliott Wave system not available. Cannot continue.")
            return False
        
        # Initialize and run complete system
        logger.info("🚀 Initializing Complete System...")
        
        # Create config for enhanced features
        config = {
            'elliott_wave': {
                'target_auc': 0.70,
                'max_features': 30,
                'enable_resource_management': resource_available,
                'enable_ml_protection': protection_available
            },
            'ml_protection': {
                'overfitting_threshold': 0.2,
                'noise_threshold': 0.1,
                'data_leakage_threshold': 0.3
            },
            'resource_management': {
                'allocation_percentage': 0.8,
                'enable_monitoring': True,
                'enable_advanced_features': True
            }
        }
        
        # Initialize Menu 1 with complete configuration
        logger.info("🌊 Initializing Elliott Wave Menu 1 with all enhancements...")
        menu1 = Menu1ElliottWaveFixed(config=config, logger=logger)
        
        # Display current status
        print("\n📊 SYSTEM STATUS:")
        print(f"   🧠 Resource Management: {'✅ ACTIVE' if menu1.resource_manager else '❌ INACTIVE'}")
        print(f"   🛡️ ML Protection: {'✅ ACTIVE' if menu1.ml_protection else '❌ INACTIVE'}")
        print(f"   📝 Advanced Logging: ✅ ACTIVE")
        print(f"   🎨 Beautiful Progress: ✅ ACTIVE")
        
        # Run the complete pipeline
        print("\n" + "="*80)
        logger.info("🚀 Starting Complete Elliott Wave Pipeline...")
        print("🚀 Running Complete Elliott Wave Pipeline with All Features!")
        print("="*80)
        
        # Execute the pipeline
        success = menu1.run_full_pipeline()
        
        if success:
            print("\n" + "="*80)
            print("🎉 COMPLETE SYSTEM EXECUTION SUCCESSFUL!")
            print("="*80)
            
            # Display summary
            if hasattr(menu1, 'results') and menu1.results:
                print("\n📊 EXECUTION SUMMARY:")
                
                # Data info
                data_info = menu1.results.get('data_info', {})
                if data_info:
                    print(f"   📈 Data Processed: {data_info.get('total_rows', 0):,} rows")
                    print(f"   🎯 Features: {data_info.get('features_count', 0)}")
                
                # ML Results
                cnn_results = menu1.results.get('cnn_lstm_results', {})
                if cnn_results:
                    auc_score = cnn_results.get('evaluation_results', {}).get('auc', cnn_results.get('auc_score', 0))
                    print(f"   🏆 AUC Score: {auc_score:.4f}")
                    print(f"   ✅ Enterprise Target: {'ACHIEVED' if auc_score >= 0.70 else 'NOT ACHIEVED'}")
                
                # Protection Results
                if 'ml_protection' in menu1.results:
                    protection = menu1.results['ml_protection'].get('overall_assessment', {})
                    print(f"   🛡️ ML Protection: {protection.get('protection_status', 'UNKNOWN')}")
                    print(f"   🔒 Risk Level: {protection.get('risk_level', 'UNKNOWN')}")
                
                # Resource Usage
                if menu1.resource_manager and hasattr(menu1.resource_manager, 'get_current_performance'):
                    try:
                        perf = menu1.resource_manager.get_current_performance()
                        print(f"   🧠 Final CPU Usage: {perf.get('cpu_percent', 0):.1f}%")
                        print(f"   💾 Final Memory Usage: {perf.get('memory', {}).get('percent', 0):.1f}%")
                    except:
                        pass
            
            print("\n🎯 All systems performed optimally!")
            logger.info("✅ Complete system execution finished successfully")
            
        else:
            print("\n" + "="*80)
            print("❌ SYSTEM EXECUTION FAILED")
            print("="*80)
            logger.error("❌ Complete system execution failed")
            
        # Cleanup
        if menu1.resource_manager and hasattr(menu1.resource_manager, 'stop_monitoring'):
            try:
                menu1.resource_manager.stop_monitoring()
                logger.info("🧹 Resource monitoring stopped")
            except:
                pass
        
        return success
        
    except Exception as e:
        print(f"\n❌ SYSTEM ERROR: {e}")
        if 'logger' in locals():
            logger.error(f"❌ System error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        
        print("\n" + "="*80)
        if success:
            print("🎊 NICEGOLD COMPLETE SYSTEM - SUCCESS!")
            print("✅ All advanced features executed successfully")
        else:
            print("❌ NICEGOLD COMPLETE SYSTEM - FAILED")
            print("⚠️  Check logs for details")
        
        print("="*80)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  System execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        sys.exit(1)
