#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧹 NICEGOLD PROJECT CLEANUP - ลบระบบซ้ำซ้อน
ลบไฟล์ที่ทำให้โปรเจคงงและรันได้แค่ระบบเดียว
"""

import os
import shutil
from pathlib import Path

# ไฟล์ Feature Selector ที่ซ้ำซ้อน - เก็บแค่ elliott_wave_modules/feature_selector.py
REDUNDANT_FEATURE_SELECTORS = [
    "advanced_feature_selector.py",
    "advanced_feature_selector.py.backup", 
    "fast_feature_selector.py",
    "optimized_enterprise_feature_selector.py",
    "real_profit_feature_selector.py",  # ไฟล์นี้ถูก import ใน elliott_wave_modules/feature_selector.py แล้ว
    "elliott_wave_modules/feature_selector_new.py",
    "test_feature_selector.py"
]

# ไฟล์ทดสอบที่ซ้ำซ้อน
REDUNDANT_TEST_FILES = [
    "check_data_usage.py",
    "check_installation.py", 
    "check_menu1_ai_logger_integration.py",
    "check_menu1_enterprise_integration.py",
    "comprehensive_menu1_demo.py",
    "comprehensive_pipeline_test.py",
    "comprehensive_sampling_fix.py",
    "comprehensive_system_analysis.py",
    "comprehensive_system_analyzer.py",
    "comprehensive_system_check.py",
    "comprehensive_system_validation.py",
    "comprehensive_system_validation_complete.py",
    "comprehensive_system_validation_test.py",
    "corrected_verification_analysis.py",
    "debug_import_only.py",
    "debug_logger_error.py",
    "debug_menu1.py",
    "debug_menu1_import.py",
    "demo_ai_analytics_phase1.py",
    "demo_ai_logger.py",
    "demo_enterprise_menu1_integration.py",
    "demo_enterprise_menu1_logger.py",
    "demo_enterprise_model_management.py",
    "demo_enterprise_model_manager.py",
    "demo_enterprise_resource_control.py",
    "demo_menu1_ai_integration.py",
    "demo_menu1_analytics.py",
    "demo_phase1_analytics_live.py",
    "demo_ultimate_unified_logger.py",
    "detailed_menu1_analysis.py",
    "development_completion_report.py",
    "diagnose_elliott_wave_error.py",
    "final_enterprise_test.py",
    "final_menu1_enterprise_logger_verification.py",
    "final_menu1_validation.py",
    "final_production_cleanup.py",
    "final_resource_compliance_test.py",
    "final_system_validation.py",
    "final_validation_master_menu1.py",
    "final_validation_test.py",
    "quick_analytics_test.py",
    "quick_menu1_status_check.py",
    "quick_menu1_test.py",
    "quick_method_test.py",
    "quick_test.py",
    "simple_enterprise_test.py",
    "simple_menu1_test.py",
    "simple_progress_test.py",
    "simple_resource_test.py",
    "simple_system_test.py",
    "simple_test.py",
    "simple_test_ultimate_logger.py",
    "test_ai_logger.py",
    "test_ai_logger_imports.py",
    "test_ai_logger_with_file_output.py",
    "test_ai_terminal_logger_integration.py",
    "test_comprehensive.py",
    "test_comprehensive_final.py",
    "test_config_to_dict.py",
    "test_csv_data_processing.py",
    "test_data_processor.py",
    "test_elliott_wave_fix.py",
    "test_elliott_wave_fix_complete.py",
    "test_enterprise_menu1.py",
    "test_enterprise_menu1_integration.py",
    "test_enterprise_model_integration.py",
    "test_enterprise_resolution.py",
    "test_enterprise_resource_control.py",
    "test_enterprise_resource_control_corrected.py",
    "test_enterprise_resource_control_final.py",
    "test_enterprise_resource_control_fixed.py",
    "test_full_data_verification.py",
    "test_full_pipeline_complete.py",
    "test_gpu_integration_fixed.py",
    "test_gpu_manager.py",
    "test_import_list.py",
    "test_imports.py",
    "test_individual_imports.py",
    "test_lightweight_resource.py",
    "test_master_menu1.py",
    "test_master_menu1_resources.py",
    "test_master_menu_fix.py",
    "test_menu1.py",
    "test_menu1_simple.py",
    "test_menu1_ultimate_resource_utilization.py",
    "test_menu_1_pipeline.py",
    "test_menu_1_production.py",
    "test_model_manager_import.py",
    "test_phase1_analytics_final.py",
    "test_phase1_analytics_integration.py",
    "test_resource_center.py",
    "test_resource_compliance.py",
    "test_resource_manager.py",
    "test_simple_import.py",
    "test_simple_resource_integration.py",
    "test_system_basic.py",
    "test_ultimate_unified_logger.py",
    "test_unified_integration.py",
    "test_unified_logger.py",
    "test_unified_logger_simple.py",
    "test_validation_method.py"
]

# ไฟล์ Fix และ Tool ที่ซ้ำซ้อน
REDUNDANT_TOOLS = [
    "aggressive_memory_optimizer.py",
    "cleanup_cuda_files.py",
    "controlled_hybrid_menu_1.py",
    "cuda_warnings_suppressor.py",
    "dependency_manager.py",
    "enhanced_hybrid_menu_1.py",
    "enhanced_menu1_with_ai_analytics.py",
    "enhanced_menu_1_fixed.py",
    "enterprise_system_startup.py",
    "fix_dependencies.py",
    "fix_elliott_wave_error.py",
    "fix_import_statements.py",
    "fix_logger_arguments.py",
    "fix_logger_compatibility.py",
    "fix_logger_component_issues.py",
    "fix_numpy_dll_complete.py",
    "fix_pipeline_critical.py",
    "hybrid_menu_1_launcher.py",
    "hybrid_resource_monitor.py",
    "install_all_libraries.py",
    "install_complete_dependencies.py",
    "install_dependencies.py",
    "install_enterprise.py",
    "installation_menu.py",
    "integrate_enterprise_menu1_logger.py",
    "integrate_menu1_ai_analytics.py",
    "integrate_ultimate_logger.py",
    "launch_enhanced_hybrid_menu_1.py",
    "launch_optimized_pipeline.py",
    "launch_optimized_pipeline_complete.py",
    "main_enterprise.py",
    "migrate_to_unified_logger.py",
    "minimal_test.py",
    "monitor_test_progress.py",
    "perfect_system_installer.py",
    "performance_optimizer.py",
    "production_status_validator.py",
    "production_validation.py",
    "project_cleanup_automation.py",
    "repair_syntax.py",
    "robust_hybrid_menu_1.py",
    "syntax_checker.py",
    "ultimate_numpy_fix.py",
    "unified_logging_system_report.py",
    "validate_master_resource_integration.py",
    "verify_analytics_integration.py",
    "verify_complete_system.py",
    "verify_installation.py",
    "verify_paths.py"
]

# รายงานและไฟล์เอกสารที่ซ้ำซ้อน
REDUNDANT_REPORTS = [
    f for f in os.listdir("/content/drive/MyDrive/ProjectP-1") 
    if f.endswith(".md") and f not in [
        "README.md", 
        "README_TH.md", 
        "PROJECT_STRUCTURE.md",
        "AI_CONTEXT_INSTRUCTIONS.md"
    ]
]

# ไฟล์ JSON ผลการทดสอบที่ซ้ำซ้อน
REDUNDANT_JSON_RESULTS = [
    f for f in os.listdir("/content/drive/MyDrive/ProjectP-1") 
    if f.endswith((".json", ".log")) and not f.startswith("requirements")
]

def cleanup_redundant_files():
    """ลบไฟล์ที่ซ้ำซ้อนทั้งหมด"""
    
    project_root = Path("/content/drive/MyDrive/ProjectP-1")
    deleted_count = 0
    errors = []
    
    print("🧹 เริ่มทำความสะอาดโปรเจค...")
    print("=" * 60)
    
    # ลบ Feature Selectors ที่ซ้ำซ้อน
    print("📂 ลบ Feature Selectors ที่ซ้ำซ้อน...")
    for file in REDUNDANT_FEATURE_SELECTORS:
        file_path = project_root / file
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ ลบแล้ว: {file}")
                deleted_count += 1
            except Exception as e:
                errors.append(f"❌ ลบไม่ได้: {file} - {e}")
    
    # ลบไฟล์ทดสอบที่ซ้ำซ้อน
    print(f"\n📂 ลบไฟล์ทดสอบที่ซ้ำซ้อน ({len(REDUNDANT_TEST_FILES)} ไฟล์)...")
    for file in REDUNDANT_TEST_FILES:
        file_path = project_root / file
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ ลบแล้ว: {file}")
                deleted_count += 1
            except Exception as e:
                errors.append(f"❌ ลบไม่ได้: {file} - {e}")
    
    # ลบเครื่องมือที่ซ้ำซ้อน
    print(f"\n📂 ลบเครื่องมือที่ซ้ำซ้อน ({len(REDUNDANT_TOOLS)} ไฟล์)...")
    for file in REDUNDANT_TOOLS:
        file_path = project_root / file
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ ลบแล้ว: {file}")
                deleted_count += 1
            except Exception as e:
                errors.append(f"❌ ลบไม่ได้: {file} - {e}")
    
    # ลบรายงานที่ซ้ำซ้อน
    print(f"\n📂 ลบรายงานที่ซ้ำซ้อน ({len(REDUNDANT_REPORTS)} ไฟล์)...")
    for file in REDUNDANT_REPORTS:
        file_path = project_root / file
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ ลบแล้ว: {file}")
                deleted_count += 1
            except Exception as e:
                errors.append(f"❌ ลบไม่ได้: {file} - {e}")
    
    # ลบผลการทดสอบที่ซ้ำซ้อน
    print(f"\n📂 ลบผลการทดสอบที่ซ้ำซ้อน ({len(REDUNDANT_JSON_RESULTS)} ไฟล์)...")
    for file in REDUNDANT_JSON_RESULTS:
        file_path = project_root / file
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"✅ ลบแล้ว: {file}")
                deleted_count += 1
            except Exception as e:
                errors.append(f"❌ ลบไม่ได้: {file} - {e}")
    
    # ลบ cache และ temporary files
    print("\n📂 ลบ cache และไฟล์ชั่วคราว...")
    cache_dirs = [".mypy_cache", "__pycache__", ".cache", "test_logs", "test_output", "test_temp"]
    for cache_dir in cache_dirs:
        cache_path = project_root / cache_dir
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"✅ ลบแล้ว: {cache_dir}/")
                deleted_count += 1
            except Exception as e:
                errors.append(f"❌ ลบไม่ได้: {cache_dir} - {e}")
    
    # ลบไฟล์ backup ที่ไม่จำเป็น
    print("\n📂 ลบ backup files...")
    backup_path = project_root / "backups"
    if backup_path.exists():
        try:
            shutil.rmtree(backup_path)
            print(f"✅ ลบแล้ว: backups/")
            deleted_count += 1
        except Exception as e:
            errors.append(f"❌ ลบไม่ได้: backups - {e}")
    
    # สรุปผลการทำความสะอาด
    print("\n" + "=" * 60)
    print("🎉 สรุปผลการทำความสะอาด")
    print("=" * 60)
    print(f"✅ ลบไฟล์สำเร็จ: {deleted_count} ไฟล์")
    
    if errors:
        print(f"❌ มีข้อผิดพลาด: {len(errors)} รายการ")
        for error in errors[:5]:  # แสดงแค่ 5 รายการแรก
            print(f"   {error}")
        if len(errors) > 5:
            print(f"   ... และอีก {len(errors) - 5} รายการ")
    
    print("\n🎯 ไฟล์หลักที่เหลือ:")
    core_files = [
        "ProjectP.py",
        "core/unified_enterprise_logger.py", 
        "elliott_wave_modules/feature_selector.py",
        "menu_modules/enhanced_menu_1_elliott_wave.py",
        "datacsv/XAUUSD_M1.csv",
        "requirements.txt"
    ]
    
    for core_file in core_files:
        core_path = project_root / core_file
        if core_path.exists():
            print(f"✅ {core_file}")
        else:
            print(f"❌ {core_file} - ไม่พบไฟล์!")
    
    return deleted_count, errors

if __name__ == "__main__":
    deleted_count, errors = cleanup_redundant_files()
    
    if deleted_count > 0:
        print(f"\n🚀 โปรเจคสะอาดแล้ว! ลบไฟล์ไป {deleted_count} ไฟล์")
        print("📋 ตอนนี้โปรเจคมีแค่ระบบเดียวที่จำเป็น")
        print("🎯 ใช้ python ProjectP.py เพื่อเริ่มระบบ")
    else:
        print("⚠️ ไม่มีไฟล์ใดถูกลบ")
