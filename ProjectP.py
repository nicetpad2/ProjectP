#!/usr/bin/env python3
"""
🏢 NICEGOLD ENTERPRISE PROJECTP
ระบบ AI-Powered Algorithmic Trading System ระดับ Enterprise

📊 Main Entry Point - สำหรับรันระบบหลัก
⚠️ THIS IS THE ONLY AUTHORIZED MAIN ENTRY POINT
🚫 DO NOT create alternative main files - use this file only
"""

# 🛠️ CUDA FIX: Apply immediate CUDA fixes before any imports
import os
import sys
import warnings

# Force CPU-only operation to prevent CUDA errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Additional CUDA suppression
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=""'

# Suppress all CUDA-related warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*cuDNN.*')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')

# Enterprise Compliance Check
from core.compliance import EnterpriseComplianceValidator
from core.menu_system import MenuSystem
from core.logger import setup_enterprise_logger
from core.config import load_enterprise_config

# Import Intelligent Resource Management
try:
    from core.intelligent_resource_manager import initialize_intelligent_resources
    from core.enhanced_intelligent_resource_manager import initialize_enhanced_intelligent_resources
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Resource Management not available: {e}")
    RESOURCE_MANAGER_AVAILABLE = False


def main():
    """จุดเริ่มต้นของระบบ NICEGOLD Enterprise - ONLY AUTHORIZED ENTRY POINT"""
    
    # Setup Enterprise Logger
    logger = setup_enterprise_logger()
    logger.info("🚀 NICEGOLD Enterprise ProjectP Starting...")
    logger.info("📌 Using ONLY authorized entry point: ProjectP.py")
    
    # Initialize Intelligent Resource Management (80% Allocation Strategy)
    resource_manager = None
    if RESOURCE_MANAGER_AVAILABLE:
        try:
            print("🧠 Initializing Intelligent Resource Management System...")
            logger.info("🧠 Starting intelligent resource detection and allocation...")
            
            # Initialize Enhanced Resource Manager with 80% allocation
            resource_manager = initialize_enhanced_intelligent_resources(
                allocation_percentage=0.8,
                enable_advanced_monitoring=True
            )
            
            print("✅ 🧠 Intelligent Resource Manager: ACTIVE (80% Allocation Strategy)")
            logger.info("✅ Enhanced Intelligent Resource Management System: ACTIVE")
            logger.info("⚡ 80% resource allocation strategy applied successfully")
            
            # Display system summary
            print("📊 System Resource Summary:")
            resource_config = resource_manager.resource_config
            
            # CPU Information
            cpu_config = resource_config.get('cpu', {})
            allocated_threads = cpu_config.get('allocated_threads', 0)
            total_cores = cpu_config.get('total_cores', 0)
            cpu_percentage = cpu_config.get('allocation_percentage', 0)
            
            print(f"   🧮 CPU: {allocated_threads}/{total_cores} cores allocated ({cpu_percentage:.1f}%)")
            
            # Memory Information  
            memory_config = resource_config.get('memory', {})
            allocated_gb = memory_config.get('allocated_gb', 0)
            total_gb = memory_config.get('total_gb', 0)
            memory_percentage = memory_config.get('allocation_percentage', 0)
            
            print(f"   🧠 Memory: {allocated_gb:.1f}/{total_gb:.1f} GB allocated ({memory_percentage:.1f}%)")
            
            # Optimization settings
            optimization = resource_config.get('optimization', {})
            batch_size = optimization.get('batch_size', 32)
            workers = optimization.get('recommended_workers', 4)
            
            print(f"   ⚡ Optimization: Batch Size {batch_size}, Workers {workers}")
            print("")
            
        except Exception as e:
            print(f"⚠️ Resource Manager initialization failed: {e}")
            logger.warning(f"Resource Manager unavailable: {e}")
    else:
        print("⚠️ Intelligent Resource Management not available")
        logger.warning("Operating without intelligent resource management")
    
    # Load Enterprise Configuration
    config = load_enterprise_config()
    
    # Add resource manager to config
    if resource_manager:
        config['resource_manager'] = resource_manager
    
    # Validate Enterprise Compliance
    validator = EnterpriseComplianceValidator()
    if not validator.validate_enterprise_compliance():
        logger.error("❌ Enterprise Compliance Validation Failed!")
        sys.exit(1)
    
    # Initialize Menu System with resource manager
    menu_system = MenuSystem(config=config, logger=logger, resource_manager=resource_manager)
    
    try:
        # Start Main Menu Loop
        menu_system.start()
        
    except KeyboardInterrupt:
        logger.info("🛑 System interrupted by user")
        
    except Exception as e:
        logger.error(f"💥 System error: {str(e)}")
        sys.exit(1)
        
    finally:
        # Cleanup resource manager
        if resource_manager:
            try:
                resource_manager.stop_monitoring()
                print("🧹 Resource monitoring stopped")
                logger.info("🧹 Intelligent resource management cleanup completed")
            except:
                pass
                
        logger.info("✅ NICEGOLD Enterprise ProjectP Shutdown Complete")


if __name__ == "__main__":
    main()
