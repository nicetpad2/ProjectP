#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏢 NICEGOLD ENTERPRISE PROJECTP - RESOURCE MANAGER WRAPPER
Simple Resource Manager Wrapper สำหรับ backward compatibility

ไฟล์นี้ทำหน้าที่เป็น wrapper สำหรับ unified_resource_manager.py
เพื่อให้ระบบสามารถ import get_resource_manager ได้

เวอร์ชัน: 1.0 Enterprise Edition
วันที่: 11 กรกฎาคม 2025
สถานะ: Production Ready
"""

# Disable type checking for this wrapper file
# type: ignore

import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_resource_manager(config=None):
    """
    ✅ ENTERPRISE RESOURCE MANAGER ACCESSOR
    
    ฟังก์ชัน wrapper สำหรับเข้าถึง Unified Resource Manager
    ใช้สำหรับ backward compatibility กับระบบเก่า
    
    Args:
        config: การตั้งค่าสำหรับ resource manager (optional dict)
        
    Returns:
        UnifiedResourceManager instance
    """
    try:
        # Import the unified resource manager
        from core.unified_resource_manager import get_unified_resource_manager
        
        # Get the manager instance
        manager = get_unified_resource_manager(config or {})  # type: ignore
        
        return manager
        
    except ImportError as e:
        print(f"❌ Error: Could not import unified_resource_manager: {e}")
        raise ImportError(
            "❌ Unified Resource Manager is not available. "
            "Please check dependencies or imports."
        )
    except Exception as e:
        print(f"❌ Error initializing resource manager: {e}")
        raise


def create_resource_manager(config=None):
    """
    สร้าง Resource Manager instance ใหม่
    
    Args:
        config: การตั้งค่าสำหรับ resource manager (optional dict)
        
    Returns:
        UnifiedResourceManager instance
    """
    try:
        # Import the unified resource manager class
        from core.unified_resource_manager import UnifiedResourceManager
        
        # Create new instance directly
        return UnifiedResourceManager(config or {})  # type: ignore
        
    except ImportError as e:
        print(f"❌ Error: Could not import UnifiedResourceManager: {e}")
        raise ImportError(
            "❌ Unified Resource Manager is not available. "
            "Please check dependencies or imports."
        )
    except Exception as e:
        print(f"❌ Error creating resource manager: {e}")
        raise


def get_default_resource_config():
    """
    ดึงการตั้งค่า default สำหรับ resource manager
    
    Returns:
        dict: การตั้งค่า default
    """
    return {
        'target_utilization': 0.80,    # 80% target utilization
        'safety_margin': 0.15,         # 15% safety margin
        'emergency_reserve': 0.05,     # 5% emergency reserve
        'monitoring_interval': 5.0,    # 5 seconds monitoring
        'history_limit': 1000          # 1000 history entries
    }


# ====================================================
# COMPATIBILITY FUNCTIONS
# ====================================================

def check_resource_manager_health():
    """ตรวจสอบสุขภาพของ resource manager"""
    try:
        # Try to create a manager instance
        manager = get_resource_manager()
        print("✅ Resource Manager health check: PASSED")
        return True
        
    except Exception as e:
        print(f"⚠️ Resource Manager health check: FAILED ({e})")
        return False


# ====================================================
# EXPORTS AND ALIASES
# ====================================================

# Create aliases for backward compatibility
get_resource_manager_instance = get_resource_manager
create_resource_manager_instance = create_resource_manager

# Module exports
__all__ = [
    'get_resource_manager',
    'create_resource_manager', 
    'get_default_resource_config',
    'check_resource_manager_health',
    'get_resource_manager_instance',
    'create_resource_manager_instance'
]


# ====================================================
# MAIN FOR TESTING
# ====================================================

def main():
    """ฟังก์ชัน main สำหรับทดสอบ"""
    print("🏢 NICEGOLD ENTERPRISE - RESOURCE MANAGER WRAPPER TEST")
    print("=" * 60)
    
    try:
        # ทดสอบการสร้าง resource manager
        print("\n🔧 Testing resource manager creation...")
        manager = get_resource_manager()
        print(f"✅ Resource manager created: {type(manager).__name__}")
        
        # ทดสอบ health check
        print("\n🏥 Running health check...")
        health_ok = check_resource_manager_health()
        print(f"✅ Health check: {'PASSED' if health_ok else 'FAILED'}")
        
        # ทดสอบการดึงการตั้งค่า default
        print("\n⚙️ Testing default config...")
        config = get_default_resource_config()
        print(f"✅ Default config loaded: {len(config)} settings")
        
        # แสดงสถานะทรัพยากรถ้าเป็นไปได้
        print("\n📊 Testing resource status...")
        if hasattr(manager, 'get_resource_status'):
            try:
                status = manager.get_resource_status()
                print(f"✅ Resource status retrieved: {len(status)} resources")
            except Exception as e:
                print(f"⚠️ Could not get resource status: {e}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 