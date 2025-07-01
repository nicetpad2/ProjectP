# 🤖 AI AGENT ENVIRONMENT POLICY - NICEGOLD ProjectP
## ประกาศนโยบายการใช้ Environment สำหรับ AI Agents

**วันที่สร้าง:** 1 กรกฎาคม 2025  
**เวอร์ชัน:** v1.0 Enterprise Environment Policy  
**สถานะ:** 🔒 MANDATORY COMPLIANCE REQUIRED  
**วัตถุประสงค์:** กำหนดนโยบายการใช้งาน environment ที่ AI agents ต้องปฏิบัติตาม

---

## 🚨 **CRITICAL POLICY ENFORCEMENT**

### ⚖️ **MANDATORY ENVIRONMENT USAGE RULES**

```yaml
STRICT REQUIREMENTS:
  🔒 ENVIRONMENT_PATH: "/home/ACER/.cache/nicegold_env/"
  🚫 GLOBAL_INSTALLATIONS: STRICTLY FORBIDDEN
  🔧 ACTIVATION_REQUIRED: ALWAYS before Python execution
  📊 VALIDATION_MANDATORY: Check health before operations
  🛡️ COMPLIANCE_LEVEL: ENTERPRISE GRADE

VIOLATION_CONSEQUENCES:
  ❌ System instability
  ❌ Package conflicts
  ❌ Production failures
  ❌ Environment corruption
```

---

## 🏗️ **PRE-CONFIGURED ISOLATED ENVIRONMENT**

### 📍 **Environment Details**
```bash
Primary Environment: /home/ACER/.cache/nicegold_env/nicegold_enterprise_env/
Purpose: Complete isolation from system Python
Status: ✅ PRODUCTION READY
Health Score: 90-95%
Total Packages: 50+ critical libraries
Installation Date: Fully configured and tested
```

### 🔧 **Activation Requirements**
```bash
# PRIMARY activation method (ALWAYS USE THIS)
source /mnt/data/projects/ProjectP/activate_nicegold_env.sh

# ALTERNATIVE activation (backup method)
source /home/ACER/.cache/nicegold_env/nicegold_enterprise_env/bin/activate

# VALIDATION command (run after activation)
python /mnt/data/projects/ProjectP/environment_manager.py
```

---

## 🛡️ **AI AGENT COMPLIANCE FRAMEWORK**

### ✅ **APPROVED AI AGENT ACTIONS**

#### **Environment Management**
```yaml
✅ ALLOWED:
  - Use pre-configured environment only
  - Activate via provided scripts
  - Validate environment health
  - Check package versions (pip list)
  - Monitor resource usage
  - Run environment diagnostics

🚫 FORBIDDEN:
  - Create new virtual environments
  - Install global packages
  - Modify environment structure
  - Use conda/mamba/pipenv
  - Bypass activation scripts
```

#### **Python Execution Pattern**
```python
# ✅ CORRECT AI AGENT EXECUTION PATTERN
def ai_agent_python_execution(script_name: str, args: str = ""):
    """🤖 Standard execution pattern for AI agents"""
    
    # Step 1: MANDATORY environment activation
    activation_command = """
    source /mnt/data/projects/ProjectP/activate_nicegold_env.sh &&
    cd /mnt/data/projects/ProjectP
    """
    
    # Step 2: MANDATORY health validation
    validation_command = f"{activation_command} && python environment_manager.py --health"
    
    # Step 3: Execute with activated environment
    execution_command = f"{activation_command} && python {script_name} {args}"
    
    return execution_command
```

#### **Package Management**
```yaml
✅ APPROVED:
  - Check installed packages: pip list
  - Verify package versions: pip show <package>
  - Report missing dependencies (DO NOT install)
  - Use environment_manager.py for diagnostics

🚫 STRICTLY FORBIDDEN:
  - pip install <package> (without environment)
  - pip install --upgrade <package>
  - pip install --force-reinstall <package>
  - conda install <package>
  - Any global package modifications
```

---

## 🎯 **AI AGENT OPERATION PATTERNS**

### 🌊 **Running NICEGOLD ProjectP Menu 1**
```python
def ai_agent_run_menu1():
    """🌊 AI Agent standard pattern for Menu 1 execution"""
    
    commands = [
        "source /mnt/data/projects/ProjectP/activate_nicegold_env.sh",
        "cd /mnt/data/projects/ProjectP", 
        "python environment_manager.py --health",  # Validate first
        "python ProjectP.py"  # Run main system
    ]
    
    full_command = " && ".join(commands)
    return full_command
```

### 🧪 **Testing Resource Management System**
```python
def ai_agent_test_resources():
    """🧪 AI Agent pattern for testing resource management"""
    
    test_commands = [
        "source /mnt/data/projects/ProjectP/activate_nicegold_env.sh",
        "cd /mnt/data/projects/ProjectP",
        "python quick_start_resource_system.py",  # Quick validation
        "python test_menu1_with_intelligent_resources.py"  # Full test
    ]
    
    full_command = " && ".join(test_commands)
    return full_command
```

### 📊 **Environment Health Monitoring**
```python
def ai_agent_monitor_environment():
    """📊 AI Agent pattern for environment monitoring"""
    
    monitoring_commands = [
        "source /mnt/data/projects/ProjectP/activate_nicegold_env.sh",
        "cd /mnt/data/projects/ProjectP",
        "python environment_manager.py --status",
        "python environment_manager.py --verify",
        "pip list | head -20"  # Show key packages
    ]
    
    full_command = " && ".join(monitoring_commands)
    return full_command
```

---

## 🔍 **ENVIRONMENT VALIDATION PROTOCOLS**

### 📋 **Pre-Execution Checklist for AI Agents**

#### **Level 1: Basic Validation**
```bash
# Quick environment check (MANDATORY before any Python execution)
source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && python --version
```

#### **Level 2: Health Validation**  
```bash
# Comprehensive health check (RECOMMENDED for important operations)
source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && 
python /mnt/data/projects/ProjectP/environment_manager.py --health
```

#### **Level 3: Full System Validation**
```bash
# Complete system verification (USE for critical operations)
source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && 
python /mnt/data/projects/ProjectP/environment_manager.py --verify &&
python -c "import numpy, pandas, sklearn, tensorflow; print('✅ Core packages OK')"
```

### 🎯 **Expected Health Indicators**
```yaml
HEALTHY_ENVIRONMENT:
  Health_Score: "90-95%"
  Package_Count: "45+ packages"
  Critical_Imports: "✅ No errors"
  Disk_Space: ">1GB free"
  Memory_Usage: "<80%"
  
WARNING_INDICATORS:
  Health_Score: "70-89%"
  Missing_Packages: "Some optional packages"
  Import_Warnings: "Non-critical warnings"
  
CRITICAL_ISSUES:
  Health_Score: "<70%"
  Import_Errors: "Critical package failures"
  Disk_Space: "<500MB"
```

---

## 📊 **AI AGENT MONITORING & REPORTING**

### 📈 **Required Monitoring for AI Agents**
```python
def ai_agent_generate_environment_report():
    """📈 Generate environment status report for AI monitoring"""
    
    report_commands = [
        "source /mnt/data/projects/ProjectP/activate_nicegold_env.sh",
        "cd /mnt/data/projects/ProjectP",
        "python environment_manager.py --status > env_status.log",
        "pip list > current_packages.log",
        "df -h /home/ACER/.cache/nicegold_env/ > disk_usage.log"
    ]
    
    return " && ".join(report_commands)
```

### 🔔 **Alert Conditions for AI Agents**
```yaml
IMMEDIATE_ATTENTION_REQUIRED:
  - Environment activation fails
  - Health score drops below 70%
  - Critical package import errors
  - Disk space below 500MB
  - Memory usage above 90%

MONITORING_REQUIRED:
  - Health score 70-89%
  - Non-critical package warnings
  - Disk space below 1GB
  - Unusual resource usage patterns
```

---

## 🚨 **VIOLATION DETECTION & RESPONSE**

### 🔍 **Common AI Agent Violations**
```yaml
VIOLATION_TYPES:
  ❌ Running Python without environment activation
  ❌ Installing packages globally
  ❌ Creating new environments
  ❌ Modifying system Python
  ❌ Bypassing health checks
  ❌ Using alternative package managers

DETECTION_METHODS:
  - Environment path verification
  - Package installation monitoring
  - Python execution tracking
  - System modification detection
```

### 🛠️ **Violation Response Protocol**
```python
def ai_agent_violation_response():
    """🛠️ Response protocol for environment violations"""
    
    steps = [
        "1. STOP current operations immediately",
        "2. REVERT to last known good state",
        "3. REACTIVATE proper environment",
        "4. VALIDATE environment health",
        "5. RESUME operations with compliance"
    ]
    
    return steps
```

---

## 📋 **ENTERPRISE COMPLIANCE DOCUMENTATION**

### 🏢 **Corporate Environment Standards**
```yaml
ENTERPRISE_REQUIREMENTS:
  Isolation_Level: "Complete separation from system Python"
  Security_Level: "Production-grade environment security"
  Stability_Level: "99.9% uptime requirement"
  Performance_Level: "Sub-30 second startup time"
  Monitoring_Level: "Continuous health monitoring"
  
COMPLIANCE_METRICS:
  Environment_Usage: "100% isolated environment usage"
  Activation_Rate: "100% proper activation compliance"
  Health_Checks: "100% pre-execution validation"
  Violation_Rate: "0% tolerance for violations"
```

### 📊 **Audit Trail Requirements**
```python
def ai_agent_audit_logging():
    """📊 Required audit logging for AI agent operations"""
    
    audit_items = [
        "Environment activation timestamp",
        "Health check results", 
        "Python execution commands",
        "Package usage tracking",
        "Resource utilization metrics",
        "Error/warning incidents",
        "Compliance verification status"
    ]
    
    return audit_items
```

---

## 🎯 **SUCCESS METRICS FOR AI AGENTS**

### ✅ **AI Agent Performance Indicators**
```yaml
OPTIMAL_PERFORMANCE:
  Environment_Activation: "100% success rate"
  Health_Validation: "90-95% health score"
  Python_Execution: "Zero import errors"
  Resource_Usage: "Efficient allocation"
  Compliance_Rate: "100% policy adherence"
  
PERFORMANCE_TRACKING:
  Startup_Time: "< 30 seconds"
  Memory_Efficiency: "< 4GB typical usage"
  CPU_Utilization: "80% allocation strategy"
  Error_Rate: "< 1% system errors"
  Uptime: "> 99% availability"
```

### 🏆 **Excellence Standards**
```python
def ai_agent_excellence_criteria():
    """🏆 Excellence criteria for AI agent operations"""
    
    criteria = {
        "environment_compliance": "Perfect adherence to isolation policy",
        "health_monitoring": "Proactive health checking",
        "resource_efficiency": "Optimal resource utilization", 
        "error_handling": "Graceful error recovery",
        "performance_optimization": "Continuous performance improvement"
    }
    
    return criteria
```

---

## 🔄 **EMERGENCY PROCEDURES FOR AI AGENTS**

### 🚨 **Emergency Response Protocol**
```bash
# EMERGENCY: Environment corruption detected
# Step 1: Immediate isolation
deactivate  # Exit any active environment

# Step 2: Environment health diagnosis
cd /mnt/data/projects/ProjectP
./environment_manager.py --emergency-check

# Step 3: Environment recovery (if needed)
./environment_manager.py --repair-environment

# Step 4: Validation and restart
source activate_nicegold_env.sh
python environment_manager.py --health
```

### 🛠️ **Recovery Procedures**
```yaml
RECOVERY_STEPS:
  Level_1: "Environment reactivation"
  Level_2: "Package verification and repair"
  Level_3: "Environment rebuild from backup"
  Level_4: "Complete environment recreation"
  
ESCALATION_CRITERIA:
  - Multiple activation failures
  - Persistent package errors
  - System-wide conflicts
  - Unrecoverable corruption
```

---

## 📋 **QUICK REFERENCE FOR AI AGENTS**

### ⚡ **Essential Commands Checklist**
```bash
# 1. Navigate to project directory
cd /mnt/data/projects/ProjectP

# 2. Activate environment (MANDATORY)
source activate_nicegold_env.sh

# 3. Validate environment health
python environment_manager.py --health

# 4. Run main system
python ProjectP.py

# 5. Test resource management (optional)
python quick_start_resource_system.py

# 6. Emergency health check
python environment_manager.py --emergency-check
```

### 🎯 **One-Command Environment Verification**
```bash
# Complete verification for AI agents
source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && 
cd /mnt/data/projects/ProjectP && 
python environment_manager.py --health && 
echo "✅ Environment ready for AI agent operations"
```

---

## 📊 **FINAL COMPLIANCE SUMMARY**

### 🎯 **AI Agent Requirements Summary**
1. **🔒 MANDATORY**: Use only pre-configured isolated environment
2. **🔧 MANDATORY**: Always activate environment before Python execution  
3. **📊 MANDATORY**: Validate environment health before operations
4. **🚫 FORBIDDEN**: Install packages globally or create new environments
5. **🛡️ MANDATORY**: Follow enterprise compliance standards
6. **📈 MANDATORY**: Monitor and report environment status
7. **🚨 MANDATORY**: Follow emergency procedures when needed

### ✅ **Success Criteria**
- [x] Environment isolation: 100% compliance
- [x] Activation procedure: 100% adherence
- [x] Health validation: 90-95% health score
- [x] Resource efficiency: Optimal utilization
- [x] Error handling: Graceful degradation
- [x] Monitoring: Continuous status tracking
- [x] Compliance: Zero policy violations

---

**📅 Effective Date:** 1 กรกฎาคม 2025  
**🔄 Version:** 1.0 Enterprise Policy  
**⚖️ Authority:** NICEGOLD Enterprise IT Security  
**🛡️ Compliance Level:** MANDATORY - No Exceptions  
**📋 Review Schedule:** Monthly compliance audits

*This policy is binding for all AI agents operating on NICEGOLD ProjectP systems. Violations will result in immediate operation suspension and security review.*
