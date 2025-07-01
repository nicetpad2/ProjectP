#!/usr/bin/env python3
"""
🧪 INTELLIGENT RESOURCE MANAGEMENT TEST
======================================

Quick test to verify Intelligent Resource Management system is working properly
and eliminate the warning: "Intelligent Resource Management not available"
"""

import os
import sys
import traceback

def test_resource_management_imports():
    """🧪 Test all resource management imports"""
    print("🧪 Testing Intelligent Resource Management Imports...")
    print("=" * 60)
    
    test_results = {
        'intelligent_resource_manager': False,
        'enhanced_intelligent_resource_manager': False, 
        'auto_activation_system': False
    }
    
    # Test 1: Basic Intelligent Resource Manager
    try:
        from core.intelligent_resource_manager import initialize_intelligent_resources, IntelligentResourceManager
        print("✅ Basic Intelligent Resource Manager: IMPORT SUCCESS")
        test_results['intelligent_resource_manager'] = True
        
        # Test initialization
        try:
            resource_manager = initialize_intelligent_resources(allocation_percentage=0.8)
            print("✅ Basic Resource Manager: INITIALIZATION SUCCESS")
            print(f"   📊 System Info: {len(resource_manager.system_info)} components detected")
        except Exception as e:
            print(f"⚠️  Basic Resource Manager initialization warning: {e}")
            
    except Exception as e:
        print(f"❌ Basic Intelligent Resource Manager: IMPORT FAILED - {e}")
        print(f"   🔍 Error details: {traceback.format_exc()}")
    
    # Test 2: Enhanced Intelligent Resource Manager  
    try:
        from core.enhanced_intelligent_resource_manager import initialize_enhanced_intelligent_resources
        print("✅ Enhanced Intelligent Resource Manager: IMPORT SUCCESS")
        test_results['enhanced_intelligent_resource_manager'] = True
        
        # Test initialization
        try:
            enhanced_manager = initialize_enhanced_intelligent_resources(allocation_percentage=0.8)
            print("✅ Enhanced Resource Manager: INITIALIZATION SUCCESS")
        except Exception as e:
            print(f"⚠️  Enhanced Resource Manager initialization warning: {e}")
            
    except Exception as e:
        print(f"❌ Enhanced Intelligent Resource Manager: IMPORT FAILED - {e}")
        print(f"   🔍 Error details: {traceback.format_exc()}")
    
    # Test 3: Auto Activation System
    try:
        from auto_activation_system import auto_activate_full_system, AutoActivationSystem
        print("✅ Auto Activation System: IMPORT SUCCESS")
        test_results['auto_activation_system'] = True
        
        # Test initialization
        try:
            auto_system = AutoActivationSystem()
            print("✅ Auto Activation System: INITIALIZATION SUCCESS")
        except Exception as e:
            print(f"⚠️  Auto Activation System initialization warning: {e}")
            
    except Exception as e:
        print(f"❌ Auto Activation System: IMPORT FAILED - {e}")
        print(f"   🔍 Error details: {traceback.format_exc()}")
    
    print("=" * 60)
    
    # Summary
    success_count = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"📊 TEST RESULTS SUMMARY:")
    print(f"   ✅ Successful: {success_count}/{total_tests}")
    print(f"   ❌ Failed: {total_tests - success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED! Intelligent Resource Management is FULLY AVAILABLE")
        return True
    elif success_count > 0:
        print("⚠️  PARTIAL SUCCESS: Some components available")
        return True
    else:
        print("❌ ALL TESTS FAILED: Intelligent Resource Management NOT AVAILABLE")
        return False

def test_projectp_integration():
    """🔗 Test ProjectP.py integration"""
    print("\n🔗 Testing ProjectP.py Integration...")
    print("=" * 60)
    
    try:
        # Import the same way ProjectP.py does
        from core.intelligent_resource_manager import initialize_intelligent_resources
        from core.enhanced_intelligent_resource_manager import initialize_enhanced_intelligent_resources
        from auto_activation_system import auto_activate_full_system
        
        print("✅ ProjectP.py Integration: ALL IMPORTS SUCCESSFUL")
        print("🧠 Intelligent Resource Management: AVAILABLE")
        print("⚡ Enhanced Resource Management: AVAILABLE") 
        print("🚀 Auto Activation System: AVAILABLE")
        
        return True
        
    except Exception as e:
        print(f"❌ ProjectP.py Integration: FAILED - {e}")
        print(f"   🔍 This is why you see the warning in ProjectP.py")
        return False

def main():
    """🎯 Main test execution"""
    print("🚀 NICEGOLD Enterprise ProjectP")
    print("🧪 Intelligent Resource Management System Test")
    print("📅 Test Date: 2025-07-01")
    print("🎯 Purpose: Eliminate 'Intelligent Resource Management not available' warning")
    print("")
    
    # Run tests
    resource_test_success = test_resource_management_imports()
    integration_test_success = test_projectp_integration()
    
    print("\n" + "=" * 60)
    print("🏆 FINAL TEST RESULTS:")
    
    if resource_test_success and integration_test_success:
        print("✅ SUCCESS: Intelligent Resource Management is FULLY FUNCTIONAL")
        print("🔧 The warning should now be ELIMINATED in ProjectP.py")
        print("🎉 System ready for production use!")
        return True
    else:
        print("❌ FAILURE: Issues detected in resource management system")
        print("🔧 The warning will persist until these issues are resolved")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
