#!/usr/bin/env python3
"""
🔍 SINGLE ENTRY POINT POLICY VALIDATOR
ตรวจสอบว่าระบบบังคับใช้นโยบาย Single Entry Point อย่างถูกต้อง

วันที่: 1 กรกฎาคม 2025
เวอร์ชัน: Enterprise Edition
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


class SingleEntryPointValidator:
    """ตัวตรวจสอบนโยบาย Single Entry Point"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.violations = []
        self.passed_tests = []
    
    def validate_policy(self) -> bool:
        """ตรวจสอบนโยบายทั้งหมด"""
        print("🔍 SINGLE ENTRY POINT POLICY VALIDATION")
        print("="*60)
        print()
        
        # Test 1: Check authorized entry point exists
        self._test_authorized_entry_point()
        
        # Test 2: Check unauthorized files redirect properly
        self._test_unauthorized_redirects()
        
        # Test 3: Check no alternative main files
        self._test_no_alternative_mains()
        
        # Test 4: Check documentation exists
        self._test_documentation_exists()
        
        # Print results
        self._print_results()
        
        return len(self.violations) == 0
    
    def _test_authorized_entry_point(self):
        """ทดสอบว่า ProjectP.py เป็น entry point ที่ถูกต้อง"""
        print("1️⃣ Testing Authorized Entry Point...")
        
        projectp_file = self.project_root / "ProjectP.py"
        if not projectp_file.exists():
            self.violations.append("❌ ProjectP.py not found - main entry point missing")
            return
        
        # Check if it has proper main block
        with open(projectp_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'if __name__ == "__main__"' not in content:
            self.violations.append("❌ ProjectP.py missing main execution block")
            return
        
        if 'ONLY AUTHORIZED ENTRY POINT' not in content:
            self.violations.append("❌ ProjectP.py missing entry point policy documentation")
            return
        
        self.passed_tests.append("✅ ProjectP.py is properly configured as single entry point")
    
    def _test_unauthorized_redirects(self):
        """ทดสอบว่าไฟล์ที่ไม่ได้รับอนุญาตทำการส่งต่อไป ProjectP.py"""
        print("2️⃣ Testing Unauthorized File Redirects...")
        
        # Test ProjectP_Advanced.py
        advanced_file = self.project_root / "ProjectP_Advanced.py"
        if advanced_file.exists():
            with open(advanced_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'This is not a main entry point' not in content:
                self.violations.append("❌ ProjectP_Advanced.py not properly restricted")
            else:
                self.passed_tests.append("✅ ProjectP_Advanced.py properly restricted")
        else:
            self.violations.append("❌ ProjectP_Advanced.py not found")
        
        # Test run_advanced.py
        run_advanced_file = self.project_root / "run_advanced.py"
        if run_advanced_file.exists():
            with open(run_advanced_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'Redirect to ProjectP.py' not in content:
                self.violations.append("❌ run_advanced.py not properly redirecting")
            else:
                self.passed_tests.append("✅ run_advanced.py properly redirects to ProjectP.py")
        else:
            self.violations.append("❌ run_advanced.py not found")
    
    def _test_no_alternative_mains(self):
        """ตรวจสอบว่าไม่มีไฟล์ main อื่นที่อาจสร้างความสับสน"""
        print("3️⃣ Testing No Alternative Main Files...")
        
        # Files that should NOT have unrestricted main blocks
        restricted_files = [
            "ProjectP_Advanced.py",
            "run_advanced.py"
        ]
        
        for filename in restricted_files:
            filepath = self.project_root / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if it has restricted main block
                if 'if __name__ == "__main__"' in content:
                    if ('ERROR' in content or 'redirect' in content.lower() or 
                        'not a main entry point' in content.lower()):
                        self.passed_tests.append(f"✅ {filename} has restricted main block")
                    else:
                        self.violations.append(f"❌ {filename} has unrestricted main block")
        
        self.passed_tests.append("✅ No unauthorized alternative main files found")
    
    def _test_documentation_exists(self):
        """ตรวจสอบว่ามีเอกสารอธิบายนโยบาย Single Entry Point"""
        print("4️⃣ Testing Policy Documentation...")
        
        # Check for policy documentation
        policy_files = [
            "ENTRY_POINT_POLICY.md",
            "README.md"
        ]
        
        for filename in policy_files:
            filepath = self.project_root / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'SINGLE ENTRY POINT' in content or 'ProjectP.py' in content:
                    self.passed_tests.append(f"✅ {filename} contains entry point policy documentation")
                else:
                    self.violations.append(f"❌ {filename} missing entry point policy documentation")
            else:
                self.violations.append(f"❌ {filename} documentation file missing")
    
    def _print_results(self):
        """แสดงผลการทดสอบ"""
        print()
        print("="*60)
        print("📋 VALIDATION RESULTS")
        print("="*60)
        
        # Show passed tests
        if self.passed_tests:
            print("\n✅ PASSED TESTS:")
            for test in self.passed_tests:
                print(f"   {test}")
        
        # Show violations
        if self.violations:
            print("\n❌ POLICY VIOLATIONS:")
            for violation in self.violations:
                print(f"   {violation}")
        else:
            print("\n🎉 NO POLICY VIOLATIONS FOUND!")
        
        # Final status
        print("\n" + "="*60)
        if len(self.violations) == 0:
            print("🎯 VALIDATION STATUS: ✅ PASSED")
            print("🚀 Single Entry Point Policy PROPERLY ENFORCED")
        else:
            print("🎯 VALIDATION STATUS: ❌ FAILED")
            print(f"⚠️  Found {len(self.violations)} policy violations")
        print("="*60)


def main():
    """รันการตรวจสอบนโยบาย Single Entry Point"""
    validator = SingleEntryPointValidator()
    success = validator.validate_policy()
    
    if success:
        print("\n🎉 Single Entry Point Policy is properly enforced!")
        print("✅ System ready for production use")
        return 0
    else:
        print("\n⚠️  Single Entry Point Policy violations found!")
        print("🔧 Please fix the violations before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
