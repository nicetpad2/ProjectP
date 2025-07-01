# 🧠 AI ENVIRONMENT INSTRUCTIONS - NICEGOLD ProjectP
## ระบบ Environment Management สำหรับ AI Agents

**วันที่สร้าง:** 1 กรกฎาคม 2025  
**เวอร์ชัน:** v1.0 Production Environment Guide  
**วัตถุประสงค์:** ให้ AI Agents เข้าใจและใช้งาน isolated environment ที่สร้างไว้แล้ว

---

## 🚨 **CRITICAL: ENVIRONMENT USAGE POLICY**

### ✅ **MANDATORY REQUIREMENTS**
```yaml
Environment Policy:
  🔒 USE ONLY: Pre-configured isolated environment in /home/ACER/.cache/nicegold_env/
  🚫 NEVER: Create new environments or install global packages
  🔧 ACTIVATE: Always use activation scripts before running Python
  📊 VALIDATE: Check environment health before execution
```

---

## 🏗️ **PRE-CONFIGURED ENVIRONMENT ARCHITECTURE**

### 📍 **Environment Location**
```bash
Environment Path: /home/ACER/.cache/nicegold_env/
Purpose: Isolated library installation (separate from main disk)
Status: ✅ ACTIVE & PRODUCTION READY
Health: 90-95% (All critical dependencies installed)
```

### 🔧 **Activation Scripts**
```bash
# Primary activation script
./activate_nicegold_env.sh

# Alternative activation method
source /home/ACER/.cache/nicegold_env/bin/activate

# Environment management
./environment_manager.py
```

### 📦 **Installed Dependencies**
```yaml
Critical ML Libraries:
  - numpy==1.26.4 (FIXED VERSION - DO NOT UPGRADE)
  - tensorflow==2.16.1
  - torch==2.3.1
  - scikit-learn==1.5.0
  - pandas==2.2.2
  - joblib==1.4.2

Advanced ML:
  - shap==0.45.0 (For feature importance)
  - optuna==3.6.1 (For hyperparameter optimization)
  - xgboost==2.0.3
  - lightgbm==4.4.0

Data Processing:
  - matplotlib==3.9.0
  - seaborn==0.13.2
  - plotly==5.22.0

System Utilities:
  - psutil==5.9.8
  - tqdm==4.66.4
  - rich==13.7.1 (For beautiful UI)
  - click==8.1.7

Total Packages: 50+ (Complete ecosystem)
```

---

## 🎯 **AI USAGE INSTRUCTIONS**

### 🚀 **Standard Workflow for AI Agents**

#### **Step 1: Environment Activation**
```python
# ALWAYS activate environment before any Python execution
import subprocess
import os

def activate_nicegold_environment():
    """🔧 Activate the pre-configured NICEGOLD environment"""
    activation_script = "/mnt/data/projects/ProjectP/activate_nicegold_env.sh"
    
    # Method 1: Using subprocess with source
    result = subprocess.run(
        f"source {activation_script} && python --version", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("✅ NICEGOLD environment activated successfully")
        return True
    else:
        print("❌ Environment activation failed")
        return False
```

#### **Step 2: Environment Validation**
```python
def validate_environment_health():
    """📊 Check environment health before execution"""
    try:
        # Run environment manager
        result = subprocess.run(
            "source activate_nicegold_env.sh && python environment_manager.py",
            shell=True,
            capture_output=True,
            text=True,
            cwd="/mnt/data/projects/ProjectP"
        )
        
        if "Environment Health: 9" in result.stdout:
            print("✅ Environment health: EXCELLENT")
            return True
        else:
            print("⚠️ Environment health check required")
            return False
            
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False
```

#### **Step 3: Python Execution**
```python
def run_python_with_environment(script_name, args=""):
    """🐍 Execute Python scripts with proper environment"""
    
    command = f"""
    source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && 
    cd /mnt/data/projects/ProjectP && 
    python {script_name} {args}
    """
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    
    return result
```

---

## 🛡️ **ENTERPRISE COMPLIANCE FOR AI**

### ✅ **APPROVED ACTIONS**
```yaml
Environment Management:
  ✅ Use existing environment: /home/ACER/.cache/nicegold_env/
  ✅ Activate via provided scripts
  ✅ Validate health before execution
  ✅ Run ProjectP.py with activated environment
  ✅ Monitor environment status

Package Management:
  ✅ Use pre-installed packages only
  ✅ Check package versions via pip list
  ✅ Report missing dependencies (don't install)
  ✅ Use environment_manager.py for diagnostics
```

### 🚫 **FORBIDDEN ACTIONS**
```yaml
Environment Violations:
  🚫 Create new virtual environments
  🚫 Install packages globally (pip install without env)
  🚫 Modify system Python installation
  🚫 Use conda/mamba/pipenv alternatives
  🚫 Upgrade/downgrade pre-configured packages

System Violations:
  🚫 Bypass activation scripts
  🚫 Run Python without environment activation
  🚫 Modify environment paths manually
  🚫 Delete or recreate environment directories
```

---

## 🔧 **ENVIRONMENT MANAGEMENT TOOLS**

### 📊 **Health Monitoring**
```bash
# Check environment health
./environment_manager.py

# Validate all packages
source activate_nicegold_env.sh && pip list

# Test critical imports
source activate_nicegold_env.sh && python -c "
import numpy; print(f'NumPy: {numpy.__version__}')
import pandas; print(f'Pandas: {pandas.__version__}')
import sklearn; print(f'Scikit-learn: {sklearn.__version__}')
"
```

### 🔍 **Diagnostic Commands**
```bash
# Environment status
./environment_manager.py --status

# Package verification
./environment_manager.py --verify

# Quick health check
./environment_manager.py --health

# Disk usage analysis
./disk_manager.sh
```

### 🔄 **Maintenance Scripts**
```bash
# Environment cleanup (if needed)
./environment_manager.py --cleanup

# Package list export
source activate_nicegold_env.sh && pip freeze > current_packages.txt

# Environment backup
tar -czf nicegold_env_backup.tar.gz /home/ACER/.cache/nicegold_env/
```

---

## 🎯 **AI EXECUTION PATTERNS**

### 🌊 **Running Menu 1 (Full Pipeline)**
```python
def execute_menu1_with_environment():
    """🌊 Execute Menu 1 with proper environment setup"""
    
    # Step 1: Validate environment
    if not validate_environment_health():
        return False
    
    # Step 2: Activate and run
    command = """
    source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && 
    cd /mnt/data/projects/ProjectP && 
    python ProjectP.py
    """
    
    # Step 3: Execute with monitoring
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Step 4: Monitor output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    
    return process.returncode == 0
```

### 🧪 **Testing Environment Components**
```python
def test_resource_management():
    """🧠 Test intelligent resource management system"""
    
    command = """
    source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && 
    cd /mnt/data/projects/ProjectP && 
    python test_menu1_with_intelligent_resources.py
    """
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr
```

### 📊 **Quick Resource System Validation**
```python
def quick_resource_validation():
    """⚡ Quick validation of resource management system"""
    
    command = """
    source /mnt/data/projects/ProjectP/activate_nicegold_env.sh && 
    cd /mnt/data/projects/ProjectP && 
    python quick_start_resource_system.py
    """
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr
```

---

## 📋 **ENVIRONMENT STATUS INDICATORS**

### ✅ **Healthy Environment Signs**
```yaml
Environment Health Indicators:
  ✅ Health Score: 90-95%
  ✅ All critical packages: ✅ INSTALLED
  ✅ NumPy version: 1.26.4 (STABLE)
  ✅ No import errors
  ✅ Sufficient disk space: >2GB free
  ✅ Memory usage: <80%
  ✅ CPU availability: >20%
```

### ⚠️ **Warning Signs**
```yaml
Environment Issues:
  ⚠️ Health Score: 70-89%
  ⚠️ Some optional packages missing
  ⚠️ Disk space: <1GB free
  ⚠️ Memory usage: >80%
  ⚠️ Import warnings (not errors)
```

### ❌ **Critical Issues**
```yaml
Environment Failures:
  ❌ Health Score: <70%
  ❌ Critical packages missing
  ❌ NumPy import errors
  ❌ Disk space: <500MB
  ❌ Python execution failures
  ❌ Environment activation fails
```

---

## 🔄 **TROUBLESHOOTING FOR AI**

### 🔧 **Common Issues & Solutions**

#### **Environment Activation Failed**
```bash
# Symptom: "activate_nicegold_env.sh: No such file or directory"
# Solution:
cd /mnt/data/projects/ProjectP
chmod +x activate_nicegold_env.sh
./activate_nicegold_env.sh
```

#### **Python Package Import Errors**
```bash
# Symptom: "ModuleNotFoundError: No module named 'numpy'"
# Solution:
source activate_nicegold_env.sh
python environment_manager.py --verify
pip list | grep numpy
```

#### **Environment Health Low**
```bash
# Symptom: Health Score < 90%
# Solution:
./environment_manager.py --health
./environment_manager.py --fix-missing
source activate_nicegold_env.sh
```

#### **Disk Space Issues**
```bash
# Symptom: "No space left on device"
# Solution:
./disk_manager.sh
./environment_manager.py --cleanup
df -h /home/ACER/.cache/
```

---

## 🚀 **QUICK START FOR AI AGENTS**

### 📋 **Essential Commands Checklist**
```bash
# 1. Navigate to project
cd /mnt/data/projects/ProjectP

# 2. Check environment health
./environment_manager.py

# 3. Activate environment
source activate_nicegold_env.sh

# 4. Validate packages
python -c "import numpy, pandas, sklearn; print('✅ Core packages OK')"

# 5. Run main system
python ProjectP.py

# 6. Test resource management (optional)
python quick_start_resource_system.py
```

### 🎯 **One-Command Environment Check**
```bash
# Complete environment validation
source activate_nicegold_env.sh && python environment_manager.py && echo "✅ Environment Ready"
```

---

## 📊 **MONITORING & REPORTING**

### 📈 **Environment Metrics**
```yaml
Key Metrics to Monitor:
  - Environment Health Score (target: ≥90%)
  - Package Count (target: ≥45 packages)
  - Disk Usage (target: <5GB total)
  - Memory Efficiency (target: <4GB RAM)
  - CPU Performance (adaptive allocation)
  - Import Success Rate (target: 100%)
```

### 📋 **Status Reports**
```python
def generate_environment_report():
    """📊 Generate comprehensive environment status report"""
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "environment_path": "/home/ACER/.cache/nicegold_env/",
        "activation_status": check_activation(),
        "health_score": get_health_score(),
        "package_count": count_packages(),
        "disk_usage": get_disk_usage(),
        "memory_usage": get_memory_usage(),
        "import_tests": run_import_tests(),
        "status": "READY" if all_checks_pass() else "NEEDS_ATTENTION"
    }
    
    return report
```

---

## 🎉 **SUCCESS CRITERIA FOR AI**

### ✅ **Environment Readiness Checklist**
- [x] Environment exists: `/home/ACER/.cache/nicegold_env/`
- [x] Activation script works: `activate_nicegold_env.sh`
- [x] Health score ≥90%: Environment Manager validation
- [x] Core packages installed: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch
- [x] SHAP + Optuna available: Advanced ML features
- [x] ProjectP.py executable: Main system ready
- [x] Resource management active: Intelligent optimization
- [x] No critical errors: All systems operational

### 🎯 **AI Agent Success Metrics**
```yaml
Execution Success:
  ✅ Environment activation: 100% success rate
  ✅ Package imports: No failures
  ✅ ProjectP.py execution: Successful startup
  ✅ Menu 1 accessibility: Full pipeline available
  ✅ Resource optimization: Automatic detection active
  ✅ Error handling: Graceful degradation
  ✅ Performance: Sub-30 second startup
```

---

## 📋 **SUMMARY FOR AI AGENTS**

### 🔑 **Critical Points**
1. **🏠 Pre-configured Environment**: Use `/home/ACER/.cache/nicegold_env/` ONLY
2. **🔧 Always Activate**: Never run Python without activation
3. **📊 Health First**: Check environment health before execution
4. **🚫 No Installation**: Use existing packages only
5. **🎯 Single Entry**: ProjectP.py is the authorized entry point
6. **🧠 Resource Management**: Intelligent system is pre-integrated
7. **📝 Monitoring**: Track environment status continuously
8. **🛡️ Enterprise Compliance**: Follow all established rules

### 🚀 **Quick Reference Commands**
```bash
# Activate environment
source /mnt/data/projects/ProjectP/activate_nicegold_env.sh

# Check health
cd /mnt/data/projects/ProjectP && ./environment_manager.py

# Run main system
cd /mnt/data/projects/ProjectP && python ProjectP.py

# Test resources
cd /mnt/data/projects/ProjectP && python quick_start_resource_system.py
```

---

**📅 Created:** 1 กรกฎาคม 2025  
**🔄 Version:** 1.0 Production Environment Guide  
**🎯 Purpose:** Complete AI Agent environment understanding  
**✅ Status:** PRODUCTION READY with Isolated Environment

*This guide provides complete instructions for AI agents to properly use the pre-configured isolated environment in NICEGOLD Enterprise ProjectP system.*
