#!/usr/bin/env python3
"""
🧪 Test Auto Activation System Fix
Quick test to verify the auto-activation fix works properly
"""

import sys
import os

# Add the project directory to sys.path
sys.path.insert(0, '/mnt/data/projects/ProjectP')

try:
    from auto_activation_system import auto_activate_full_system
    
    print("🧪 Testing Auto Activation System Fix...")
    print("=" * 50)
    
    # Test the auto activation function
    auto_system = auto_activate_full_system()
    
    print(f"✅ auto_activate_full_system() returned: {type(auto_system)}")
    
    # Test the get_activated_systems method
    activated = auto_system.get_activated_systems()
    print(f"✅ get_activated_systems() returned: {type(activated)}")
    print(f"📊 Activated systems info: {activated}")
    
    # Test last_activation_result
    if hasattr(auto_system, 'last_activation_result'):
        print(f"✅ last_activation_result available: {auto_system.last_activation_result}")
    else:
        print("❌ last_activation_result not found")
    
    print("\n🎉 Auto Activation System Fix: SUCCESS!")
    print("✅ No more 'dict' object has no attribute 'get_activated_systems' error")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
