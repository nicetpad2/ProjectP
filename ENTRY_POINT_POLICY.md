# 🚀 NICEGOLD ENTERPRISE - ENTRY POINT POLICY

## 📋 Single Entry Point Rule

**CRITICAL**: The NICEGOLD ProjectP system uses **ONLY ONE MAIN ENTRY POINT**

### ✅ AUTHORIZED MAIN ENTRY POINT
```bash
python ProjectP.py
```

### ❌ UNAUTHORIZED ENTRY POINTS
The following files are **NOT** allowed as main entry points:
- `ProjectP_Advanced.py` - Support module only
- `run_advanced.py` - Redirector to ProjectP.py
- Any other files with `if __name__ == "__main__"`

## 🎯 Why Single Entry Point?

### 1. **System Integrity**
- Ensures consistent initialization
- Prevents conflicting configurations
- Maintains enterprise compliance

### 2. **Centralized Control**
- All logging goes through main system
- Unified error handling
- Consistent menu system

### 3. **Production Safety**
- Prevents unauthorized execution paths
- Ensures all security checks are performed
- Maintains audit trail

## 🏗️ Architecture Overview

```
ProjectP.py (MAIN ENTRY)
│
├── core/menu_system.py (Menu Management)
├── core/compliance.py (Enterprise Validation)
├── core/logger.py (Logging System)
├── core/config.py (Configuration)
│
└── menu_modules/
    └── menu_1_elliott_wave.py (Full Pipeline)
```

## 🔧 For Developers

### Adding New Features
1. Create modules in appropriate directories
2. Import them in `ProjectP.py` or existing modules
3. **DO NOT** create new main entry points

### Testing
- Test files can have `if __name__ == "__main__"` for testing only
- Production code must flow through `ProjectP.py`

### Deployment
- Only `ProjectP.py` should be used in production environments
- All other files are support modules

## 🛡️ Compliance Validation

The system automatically validates:
- ✅ Only `ProjectP.py` is used as main entry
- ✅ All modules are properly imported
- ✅ No unauthorized execution paths exist

## 🚨 Violation Handling

If unauthorized entry points are detected:
1. System shows error message
2. Redirects to `ProjectP.py`
3. Logs security violation
4. Prevents execution

## 📞 Usage Examples

### ✅ CORRECT
```bash
# Main system
python ProjectP.py

# Testing (development only)
python test_installation.py
python verify_enterprise_compliance.py
```

### ❌ INCORRECT
```bash
# These will show error and redirect
python ProjectP_Advanced.py
python run_advanced.py
```

## 🎉 Summary

**REMEMBER**: Always use `python ProjectP.py` as the only main entry point for the NICEGOLD Enterprise system. All other files are support modules that work through this single, authorized entry point.

---
**Status**: 🔒 **ENFORCED** - Single Entry Point Policy Active
**Date**: July 1, 2025
**Version**: Enterprise Edition
