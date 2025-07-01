#!/usr/bin/env python3
"""
⚙️ NICEGOLD ENTERPRISE CONFIGURATION
ระบบการจัดการค่าตั้งต่างๆ ระดับ Enterprise
"""

import os
import yaml
from typing import Dict, Any, Optional
import logging
from .project_paths import get_project_paths


def load_enterprise_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """โหลดการตั้งค่า Enterprise"""
    
    # Get project paths
    paths = get_project_paths()
    
    # Default configuration path
    if config_path is None:
        config_file = paths.enterprise_config
    else:
        config_file = paths.project_root / config_path
    
    # Default Enterprise Configuration with proper paths
    default_config = {
        "system": {
            "name": "NICEGOLD Enterprise ProjectP",
            "version": "2.0 DIVINE EDITION",
            "environment": "production",
            "debug": False
        },
        "elliott_wave": {
            "enabled": True,
            "cnn_lstm_enabled": True,
            "dqn_enabled": True,
            "target_auc": 0.70,
            "max_features": 30,
            "enterprise_grade": True
        },
        "ml_protection": {
            "anti_overfitting": True,
            "no_data_leakage": True,
            "walk_forward_validation": True,
            "enterprise_compliance": True
        },
        "data": {
            "real_data_only": True,
            "no_mock_data": True,
            "no_simulation": True,
            "datacsv_path": str(paths.datacsv),
            "models_path": str(paths.models)
        },
        "performance": {
            "min_auc": 0.70,
            "min_sharpe_ratio": 1.5,
            "max_drawdown": 0.15,
            "min_win_rate": 0.60
        },
        "paths": {
            "data": str(paths.datacsv),
            "models": str(paths.models),
            "results": str(paths.results),
            "logs": str(paths.logs),
            "temp": str(paths.temp),
            "outputs": str(paths.outputs),
            "reports": str(paths.reports),
            "charts": str(paths.charts),
            "analysis": str(paths.analysis)
        }
    }
    
    # Try to load from file if exists
    if config_file.exists():
        try:
            with open(str(config_file), 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                # Merge with default config
                default_config.update(file_config)
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file: {e}")
            print("Using default configuration...")
    
    # Ensure all directories exist using project paths
    paths.ensure_directory_exists(paths.models)
    paths.ensure_directory_exists(paths.results)
    paths.ensure_directory_exists(paths.logs)
    paths.ensure_directory_exists(paths.temp)
    paths.ensure_directory_exists(paths.outputs)
    
    return default_config


class EnterpriseConfig:
    """Enterprise Configuration Manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = load_enterprise_config(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """ดึงค่าการตั้งค่าตาม key"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """ตั้งค่าตาม key"""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value
    
    def save(self):
        """บันทึกการตั้งค่าลงไฟล์"""
        if self.config_path:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, 
                         default_flow_style=False, 
                         allow_unicode=True)
    
    def is_production(self) -> bool:
        """ตรวจสอบว่าเป็น production environment หรือไม่"""
        return self.get('system.environment') == 'production'
    
    def is_enterprise_grade(self) -> bool:
        """ตรวจสอบว่าเป็น enterprise grade หรือไม่"""
        return self.get('elliott_wave.enterprise_grade', True)
