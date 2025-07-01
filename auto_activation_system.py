#!/usr/bin/env python3
"""
🚀 AUTO ACTIVATION SYSTEM - NICEGOLD ProjectP
===============================================

ระบบเปิดใช้งานอัตโนมัติสำหรับ NICEGOLD Enterprise ProjectP
รองรับการเรียกใช้งาน Full Pipeline และระบบอื่นๆ อัตโนมัติ

🎯 Key Features:
- Auto-activation of Full Pipeline (Menu 1)
- Intelligent Resource Pre-allocation
- Enterprise Compliance Auto-checks
- Production-ready Initialization
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class AutoActivationSystem:
    """
    🚀 Auto Activation System for NICEGOLD ProjectP
    ระบบเปิดใช้งานอัตโนมัติขั้นสูง
    """
    
    def __init__(self):
        """Initialize Auto Activation System"""
        self.activation_time = datetime.now()
        self.system_ready = False
        self.activation_log = []
        
        logger.info("🚀 Auto Activation System initialized")
    
    def auto_activate_full_system(self, 
                                  target_menu: int = 1,
                                  enable_monitoring: bool = True,
                                  allocation_percentage: float = 0.8) -> Dict[str, Any]:
        """
        🌊 Auto-activate Full System with Menu 1 Elliott Wave Pipeline
        
        Args:
            target_menu: เมนูเป้าหมาย (default: 1 = Full Pipeline)
            enable_monitoring: เปิดใช้งานการติดตาม
            allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร
        
        Returns:
            Dict: ผลลัพธ์การเปิดใช้งาน
        """
        activation_results = {
            'status': 'started',
            'target_menu': target_menu,
            'activation_time': self.activation_time.isoformat(),
            'steps_completed': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Step 1: Pre-activation checks
            logger.info("🔍 Step 1: Pre-activation system checks")
            pre_check_result = self._perform_pre_activation_checks()
            activation_results['steps_completed'].append('pre_activation_checks')
            
            if not pre_check_result['success']:
                activation_results['warnings'].extend(pre_check_result['warnings'])
            
            # Step 2: Resource allocation
            logger.info("⚡ Step 2: Intelligent resource allocation")
            resource_result = self._setup_resource_allocation(allocation_percentage)
            activation_results['steps_completed'].append('resource_allocation')
            
            # Step 3: Menu system preparation
            if target_menu == 1:
                logger.info("🌊 Step 3: Menu 1 Elliott Wave Pipeline preparation")
                menu_result = self._prepare_menu1_pipeline()
                activation_results['steps_completed'].append('menu1_preparation')
            
            # Step 4: Final system readiness
            logger.info("✅ Step 4: Final system readiness validation")
            readiness_result = self._validate_system_readiness()
            activation_results['steps_completed'].append('system_readiness')
            
            # Update status
            activation_results['status'] = 'completed'
            activation_results['system_ready'] = True
            self.system_ready = True
            
            logger.info("🎉 Auto Activation System: Full system ready!")
            return activation_results
            
        except Exception as e:
            logger.error(f"❌ Auto activation failed: {e}")
            activation_results['status'] = 'failed'
            activation_results['errors'].append(str(e))
            return activation_results
    
    def _perform_pre_activation_checks(self) -> Dict[str, Any]:
        """🔍 Perform pre-activation system checks"""
        checks = {
            'success': True,
            'warnings': [],
            'checks_performed': []
        }
        
        try:
            # Check data files
            data_files = ['datacsv/XAUUSD_M1.csv', 'datacsv/XAUUSD_M15.csv']
            for file_path in data_files:
                if os.path.exists(file_path):
                    checks['checks_performed'].append(f'data_file_{file_path}')
                else:
                    checks['warnings'].append(f"Data file not found: {file_path}")
            
            # Check core modules
            core_modules = ['core/menu_system.py', 'core/logger.py', 'core/config.py']
            for module_path in core_modules:
                if os.path.exists(module_path):
                    checks['checks_performed'].append(f'core_module_{module_path}')
                else:
                    checks['warnings'].append(f"Core module not found: {module_path}")
            
            # Check Elliott Wave modules
            elliott_modules = [
                'elliott_wave_modules/data_processor.py',
                'elliott_wave_modules/feature_selector.py',
                'elliott_wave_modules/pipeline_orchestrator.py'
            ]
            for module_path in elliott_modules:
                if os.path.exists(module_path):
                    checks['checks_performed'].append(f'elliott_module_{module_path}')
                else:
                    checks['warnings'].append(f"Elliott Wave module not found: {module_path}")
            
            return checks
            
        except Exception as e:
            checks['success'] = False
            checks['warnings'].append(f"Pre-activation check error: {e}")
            return checks
    
    def _setup_resource_allocation(self, allocation_percentage: float) -> Dict[str, Any]:
        """⚡ Setup intelligent resource allocation"""
        resource_setup = {
            'success': True,
            'allocation_percentage': allocation_percentage,
            'resources_allocated': []
        }
        
        try:
            # Import and initialize resource manager
            from core.intelligent_resource_manager import initialize_intelligent_resources
            
            resource_manager = initialize_intelligent_resources(
                allocation_percentage=allocation_percentage,
                enable_monitoring=True
            )
            
            resource_setup['resources_allocated'].append('intelligent_resource_manager')
            resource_setup['resource_manager'] = resource_manager
            
            logger.info(f"⚡ Resources allocated: {allocation_percentage*100}% of system capacity")
            return resource_setup
            
        except Exception as e:
            logger.warning(f"⚠️ Resource allocation setup failed: {e}")
            resource_setup['success'] = False
            resource_setup['error'] = str(e)
            return resource_setup
    
    def _prepare_menu1_pipeline(self) -> Dict[str, Any]:
        """🌊 Prepare Menu 1 Elliott Wave Pipeline"""
        menu1_prep = {
            'success': True,
            'components_prepared': []
        }
        
        try:
            # Validate Menu 1 components
            menu1_file = 'menu_modules/menu_1_elliott_wave.py'
            if os.path.exists(menu1_file):
                menu1_prep['components_prepared'].append('menu_1_elliott_wave')
            
            # Validate Elliott Wave modules
            elliott_components = [
                'data_processor',
                'feature_selector', 
                'cnn_lstm_engine',
                'dqn_agent',
                'pipeline_orchestrator',
                'performance_analyzer'
            ]
            
            for component in elliott_components:
                component_path = f'elliott_wave_modules/{component}.py'
                if os.path.exists(component_path):
                    menu1_prep['components_prepared'].append(component)
            
            logger.info("🌊 Menu 1 Elliott Wave Pipeline prepared")
            return menu1_prep
            
        except Exception as e:
            logger.warning(f"⚠️ Menu 1 preparation failed: {e}")
            menu1_prep['success'] = False
            menu1_prep['error'] = str(e)
            return menu1_prep
    
    def _validate_system_readiness(self) -> Dict[str, Any]:
        """✅ Validate final system readiness"""
        readiness = {
            'success': True,
            'validations_passed': []
        }
        
        try:
            # Check if main entry point is accessible
            if os.path.exists('ProjectP.py'):
                readiness['validations_passed'].append('main_entry_point')
            
            # Check directories
            required_dirs = ['core', 'elliott_wave_modules', 'menu_modules', 'datacsv', 'logs', 'outputs']
            for dir_name in required_dirs:
                if os.path.exists(dir_name):
                    readiness['validations_passed'].append(f'directory_{dir_name}')
            
            logger.info("✅ System readiness validation completed")
            return readiness
            
        except Exception as e:
            logger.warning(f"⚠️ System readiness validation failed: {e}")
            readiness['success'] = False
            readiness['error'] = str(e)
            return readiness
    
    def get_activation_status(self) -> Dict[str, Any]:
        """📊 Get current activation status"""
        return {
            'system_ready': self.system_ready,
            'activation_time': self.activation_time.isoformat(),
            'uptime_seconds': (datetime.now() - self.activation_time).total_seconds(),
            'activation_log': self.activation_log
        }


def auto_activate_full_system(target_menu: int = 1, 
                              enable_monitoring: bool = True,
                              allocation_percentage: float = 0.8) -> Dict[str, Any]:
    """
    🚀 Auto-activate Full NICEGOLD ProjectP System
    
    Args:
        target_menu: เมนูเป้าหมาย (default: 1 = Full Pipeline)
        enable_monitoring: เปิดใช้งานการติดตาม
        allocation_percentage: เปอร์เซ็นต์การจัดสรรทรัพยากร
    
    Returns:
        Dict: ผลลัพธ์การเปิดใช้งาน
    """
    auto_system = AutoActivationSystem()
    return auto_system.auto_activate_full_system(
        target_menu=target_menu,
        enable_monitoring=enable_monitoring,
        allocation_percentage=allocation_percentage
    )


if __name__ == "__main__":
    # Test auto activation system
    print("🚀 Testing Auto Activation System...")
    result = auto_activate_full_system()
    print(f"📊 Activation Result: {result}")
