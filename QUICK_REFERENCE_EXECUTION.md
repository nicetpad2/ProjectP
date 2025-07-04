# 🎯 QUICK REFERENCE - Environment & Execution Paths
## NICEGOLD ProjectP - ใช้งานด่วน

---

## 🚨 **CRITICAL RULES - กฎสำคัญ**

```yaml
MANDATORY:
  ✅ ใช้ isolated environment เท่านั้น
  ✅ activate environment ก่อนรันทุกครั้ง
  ✅ รันใน /mnt/data/projects/ProjectP เท่านั้น
  ✅ ใช้ ProjectP.py เป็น entry point เท่านั้น

FORBIDDEN:
  ❌ ห้ามรันด้วย system Python
  ❌ ห้าม pip install ใน system
  ❌ ห้ามรันนอก project directory
```

---

## ⚡ **QUICK EXECUTION - รันด่วน**

### 🎯 **Method 1: Standard (แนะนำ)**
```bash
cd /mnt/data/projects/ProjectP
source activate_nicegold_env.sh
python ProjectP.py
```

### 🎯 **Method 2: One Command**
```bash
cd /mnt/data/projects/ProjectP && source activate_nicegold_env.sh && python ProjectP.py
```

### 🎯 **Method 3: Manual Activation**
```bash
cd /mnt/data/projects/ProjectP
source /home/ACER/.cache/nicegold_env/bin/activate
python ProjectP.py
```

---

## 🔍 **QUICK CHECKS - ตรวจสอบด่วน**

### ✅ **Environment Health**
```bash
cd /mnt/data/projects/ProjectP
./environment_manager.py
```

### ✅ **Package Check**
```bash
source /home/ACER/.cache/nicegold_env/bin/activate
python -c "import numpy, pandas; print('✅ Ready')"
```

### ✅ **Full Verification**
```bash
cd /mnt/data/projects/ProjectP && ./environment_manager.py && source /home/ACER/.cache/nicegold_env/bin/activate && python -c "import numpy; print('✅ All OK')"
```

---

## 🗂️ **IMPORTANT PATHS - Path สำคัญ**

```yaml
Project Directory:
  📁 /mnt/data/projects/ProjectP/

Environment:
  🏠 /home/ACER/.cache/nicegold_env/

Main Entry Point:
  🚀 /mnt/data/projects/ProjectP/ProjectP.py

Activation Script:
  🔧 /mnt/data/projects/ProjectP/activate_nicegold_env.sh

Environment Manager:
  ⚙️ /mnt/data/projects/ProjectP/environment_manager.py
```

---

## 🆘 **TROUBLESHOOTING - แก้ปัญหา**

### ❓ **Common Issues**

#### **Environment Not Found**
```bash
chmod +x activate_nicegold_env.sh
source activate_nicegold_env.sh
```

#### **Python Not Found**
```bash
source /home/ACER/.cache/nicegold_env/bin/activate
which python
```

#### **Package Missing**
```bash
source /home/ACER/.cache/nicegold_env/bin/activate
./environment_manager.py --fix
```

#### **Permission Denied**
```bash
chmod +x *.sh *.py
ls -la activate_nicegold_env.sh
```

---

## 📊 **STATUS INDICATORS - ตัวบ่งชี้สถานะ**

### ✅ **Good Signs**
```
Environment Health: ✅ 95% HEALTHY
Python Path: /home/ACER/.cache/nicegold_env/bin/python
Current Directory: /mnt/data/projects/ProjectP
Import Test: ✅ All packages OK
```

### ❌ **Warning Signs**
```
Environment Health: ⚠️ <90%
Python Path: /usr/bin/python (WRONG!)
Current Directory: /home/user (WRONG!)
Import Errors: ModuleNotFoundError
```

---

## 🔧 **FOR DEVELOPERS - สำหรับนักพัฒนา**

### 📝 **Development Commands**
```bash
# Check environment status
./environment_manager.py --status

# Fix missing packages
./environment_manager.py --fix

# Disk space check
./disk_manager.sh

# Full diagnostics
./environment_manager.py --diagnose
```

### 🧪 **Testing Commands**
```bash
# Test core imports
python -c "import numpy, pandas, sklearn, tensorflow; print('✅ Core OK')"

# Test project imports
python -c "from core.menu_system import MenuSystem; print('✅ Project OK')"

# Test resource system
python -c "from core.intelligent_resource_manager import IntelligentResourceManager; print('✅ Resources OK')"
```

---

## 🤖 **FOR AI AGENTS - สำหรับ AI**

### 🔧 **AI Execution Pattern**
```python
# 1. Navigate to project
run_in_terminal("cd /mnt/data/projects/ProjectP")

# 2. Check environment
run_in_terminal("./environment_manager.py")

# 3. Activate and run
run_in_terminal("./activate_nicegold_env.sh && python ProjectP.py")
```

### ✅ **AI Validation Steps**
```python
# Validate before execution
def validate_environment():
    # Check project directory
    # Check environment health
    # Check activation script
    # Check ProjectP.py
    return ready_status
```

---

## 📋 **CHECKLIST - รายการตรวจสอบ**

### ✅ **Pre-Execution Checklist**
- [ ] อยู่ใน directory `/mnt/data/projects/ProjectP`
- [ ] Environment health ≥90%
- [ ] Activation script พร้อมใช้
- [ ] ProjectP.py มีอยู่และอ่านได้
- [ ] Core packages import ได้

### ✅ **Post-Execution Checklist**
- [ ] Menu system โหลดสำเร็จ
- [ ] ไม่มี import errors
- [ ] Resource manager ทำงาน
- [ ] Logging system active

---

## 🎉 **SUCCESS EXAMPLE - ตัวอย่างความสำเร็จ**

```bash
user@system:~$ cd /mnt/data/projects/ProjectP
user@system:/mnt/data/projects/ProjectP$ ./activate_nicegold_env.sh
🔧 Activating NICEGOLD Environment...
✅ Environment activated: /home/ACER/.cache/nicegold_env/

user@system:/mnt/data/projects/ProjectP$ python ProjectP.py
🚀 Initializing Advanced Terminal Logger System...
✅ Advanced Terminal Logger integrated successfully!
🧠 Initializing Intelligent Resource Management...
✅ Resource Management System Ready!

🏢 NICEGOLD ENTERPRISE PROJECTP - DIVINE EDITION
========================================================

Main Menu:
1. 🌊 Full Pipeline (Elliott Wave CNN-LSTM + DQN)
2. 📊 Data Analysis & Preprocessing  
3. 🤖 Model Training & Optimization
4. 🎯 Strategy Backtesting
5. 📈 Performance Analytics
E. 🚪 Exit System
R. 🔄 Reset & Restart

🎯 Select option (1-5, E, R): 
```

---

**🎯 This is your go-to reference for running NICEGOLD ProjectP correctly!**  
**📖 For detailed information, see: ENVIRONMENT_REQUIREMENTS_ANNOUNCEMENT.md**
