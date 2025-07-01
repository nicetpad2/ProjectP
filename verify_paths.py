#!/usr/bin/env python3
"""
✅ NICEGOLD PATH VERIFICATION SCRIPT
ตรวจสอบการทำงานของระบบ path ในทุกสภาพแวดล้อม
"""

import sys
import os
import platform
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def test_basic_paths():
    """Test basic path detection"""
    print_header("🔍 BASIC PATH DETECTION")
    
    # Current working directory
    cwd = Path.cwd()
    print_info(f"Current Working Directory: {cwd}")
    
    # Script location
    script_path = Path(__file__).resolve()
    print_info(f"Script Location: {script_path}")
    
    # Project root detection
    project_markers = ['ProjectP.py', 'requirements.txt', 'core', 'config']
    
    for parent in [script_path.parent] + list(script_path.parents):
        markers_found = [marker for marker in project_markers 
                        if (parent / marker).exists()]
        if len(markers_found) >= 3:
            print_success(f"Project Root Found: {parent}")
            print_info(f"Markers found: {', '.join(markers_found)}")
            return parent
    
    print_error("Could not detect project root!")
    return None

def test_project_paths_import(project_root):
    """Test ProjectPaths import"""
    print_header("📦 PROJECT PATHS IMPORT TEST")
    
    if project_root:
        # Add to sys.path
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            print_success(f"Added to sys.path: {project_root_str}")
    
    try:
        from core.project_paths import ProjectPaths, get_project_paths
        print_success("ProjectPaths imported successfully")
        
        # Test instantiation
        paths = get_project_paths()
        print_success("ProjectPaths instantiated successfully")
        
        return paths
    except ImportError as e:
        print_error(f"Failed to import ProjectPaths: {e}")
        return None
    except Exception as e:
        print_error(f"Error creating ProjectPaths: {e}")
        return None

def test_path_utilities():
    """Test path utilities"""
    print_header("🔧 PATH UTILITIES TEST")
    
    try:
        from core.path_utils import (
            ensure_project_in_path,
            get_safe_project_paths,
            resolve_data_path,
            list_data_files
        )
        print_success("Path utilities imported successfully")
        
        # Test project root detection
        project_root = ensure_project_in_path()
        print_success(f"Project root detected: {project_root}")
        
        # Test safe import
        paths = get_safe_project_paths()
        if paths:
            print_success("Safe ProjectPaths import successful")
        else:
            print_warning("Safe ProjectPaths import returned None")
        
        # Test data file resolution
        test_file = "test.csv"
        data_path = resolve_data_path(test_file)
        print_success(f"Data path resolution: {data_path}")
        
        # Test data files listing
        data_files = list_data_files()
        print_success(f"Found {len(data_files)} data files")
        
        return True
    except Exception as e:
        print_error(f"Path utilities test failed: {e}")
        return False

def test_directory_structure(paths):
    """Test directory structure"""
    print_header("📁 DIRECTORY STRUCTURE TEST")
    
    if not paths:
        print_error("No ProjectPaths instance to test")
        return False
    
    # Test all major directories
    directories = {
        'Project Root': paths.project_root,
        'Core': paths.core,
        'Config': paths.config,
        'Data CSV': paths.datacsv,
        'Models': paths.models,
        'Results': paths.results,
        'Logs': paths.logs,
        'Temp': paths.temp,
        'Outputs': paths.outputs,
        'Reports': paths.reports,
        'Charts': paths.charts,
        'Analysis': paths.analysis
    }
    
    all_good = True
    for name, path in directories.items():
        if path.exists():
            print_success(f"{name}: {path}")
        else:
            print_warning(f"{name}: {path} (will be created)")
            all_good = False
    
    return all_good

def test_file_operations(paths):
    """Test file operations"""
    print_header("🗂️  FILE OPERATIONS TEST")
    
    if not paths:
        print_error("No ProjectPaths instance to test")
        return False
    
    try:
        # Test data files
        data_files = paths.list_data_files()
        print_success(f"Data files: {len(data_files)} found")
        for file in data_files[:3]:  # Show first 3
            print_info(f"  - {file}")
        
        # Test path generation
        test_model_path = paths.get_model_file_path("test_model")
        print_success(f"Model path generation: {test_model_path.name}")
        
        test_results_path = paths.get_results_file_path("test_results")
        print_success(f"Results path generation: {test_results_path.name}")
        
        # Test directory creation
        test_dir = paths.temp / "test_verification"
        if paths.ensure_directory_exists(test_dir):
            print_success(f"Directory creation test: {test_dir}")
            # Clean up
            try:
                test_dir.rmdir()
                print_info("Test directory cleaned up")
            except:
                pass
        else:
            print_error("Directory creation test failed")
        
        return True
    except Exception as e:
        print_error(f"File operations test failed: {e}")
        return False

def test_config_integration():
    """Test configuration integration"""
    print_header("⚙️ CONFIGURATION INTEGRATION TEST")
    
    try:
        from core.config import load_enterprise_config
        config = load_enterprise_config()
        print_success("Enterprise config loaded successfully")
        
        # Check if paths are properly set
        paths_config = config.get('paths', {})
        if paths_config:
            print_success("Paths configuration found:")
            for key, value in list(paths_config.items())[:5]:  # Show first 5
                print_info(f"  {key}: {value}")
        else:
            print_warning("No paths configuration found")
        
        return True
    except Exception as e:
        print_error(f"Config integration test failed: {e}")
        return False

def test_cross_platform_compatibility():
    """Test cross-platform compatibility"""
    print_header("🌍 CROSS-PLATFORM COMPATIBILITY TEST")
    
    print_info(f"Platform: {platform.system()}")
    print_info(f"Platform Version: {platform.version()}")
    print_info(f"Architecture: {platform.architecture()[0]}")
    print_info(f"Python Version: {sys.version}")
    print_info(f"Path Separator: '{os.sep}'")
    
    # Test path separators
    test_path = Path("test") / "subdir" / "file.txt"
    print_success(f"Path separator test: {test_path}")
    
    # Test absolute vs relative paths
    relative = Path("./test")
    absolute = relative.resolve()
    print_success(f"Relative to absolute: {relative} → {absolute}")
    
    return True

def main():
    """Main verification function"""
    print_header("🏢 NICEGOLD PROJECT PATH VERIFICATION")
    print_info("Testing path system for cross-platform compatibility...")
    
    # Run all tests
    tests = [
        ("Basic Path Detection", lambda: test_basic_paths()),
        ("Project Paths Import", lambda: test_project_paths_import(test_basic_paths())),
        ("Path Utilities", test_path_utilities),
        ("Cross-Platform Compatibility", test_cross_platform_compatibility),
        ("Configuration Integration", test_config_integration),
    ]
    
    # Get paths for remaining tests
    project_root = test_basic_paths()
    paths = test_project_paths_import(project_root)
    
    # Additional tests that need paths
    if paths:
        tests.extend([
            ("Directory Structure", lambda: test_directory_structure(paths)),
            ("File Operations", lambda: test_file_operations(paths)),
        ])
    
    # Run tests and collect results
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print_header("📋 TEST SUMMARY")
    
    passed = sum(1 for name, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{Colors.BOLD}Overall Result: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print_success("🎉 All tests passed! Path system is working correctly.")
        return 0
    else:
        print_error("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
