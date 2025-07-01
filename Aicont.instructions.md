# 🧠 NICEGOLD ENTERPRISE PROJECTP - AI CONTEXT INSTRUCTIONS (AICONT)
## สำหรับ AI Agent - Complete Understanding

**เวอร์ชัน:** v2.0 DIVINE EDITION  
**สถานะ:** 95% Production Ready  
**อัปเดต:** 1 กรกฎาคม 2025  

---

## 🎯 ระบบภาพรวม (System Overview)

### 🏢 **ชื่อระบบและวัตถุประสงค์**
```yaml
ชื่อเต็ม: NICEGOLD Enterprise ProjectP
วัตถุประสงค์: AI-Powered Algorithmic Trading System สำหรับ XAU/USD (ทองคำ)
ระดับ: Enterprise-Grade Production System
เวอร์ชัน: 2.0 DIVINE EDITION
สถานะ: 95% Production Ready (รอแก้ไข NumPy dependency)
```

### 🎪 **เทคโนโลยีหลัก**
```yaml
AI/ML Stack:
  - Elliott Wave Pattern Recognition (CNN-LSTM)
  - Deep Q-Network (DQN) Reinforcement Learning
  - SHAP + Optuna Feature Selection
  - Enterprise ML Protection System

Core Technologies:
  - Python 3.8+
  - TensorFlow 2.17.0 (CNN-LSTM)
  - PyTorch 2.4.1 (DQN)
  - NumPy 1.26.4 (SHAP compatible)
  - Pandas 2.2.3
  - Scikit-learn 1.5.2
  - SHAP 0.45.0
  - Optuna 3.5.0

Data:
  - XAUUSD_M1.csv (1,771,970 rows / 131MB)
  - XAUUSD_M15.csv (118,173 rows / 8.6MB)
  - 100% Real Market Data Only
```

---

## 🔑 Entry Point และ Architecture

### 🚀 **ProjectP.py (ONLY AUTHORIZED ENTRY POINT)**
```python
Purpose: Single authorized entry point for entire system
Critical Functions:
  1. CUDA Environment Setup:
     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only
     
  2. Enterprise Compliance Validation:
     from core.compliance import EnterpriseComplianceValidator
     
  3. System Initialization:
     - Logger setup
     - Configuration loading
     - Menu system startup

Key Rules:
  ✅ THIS IS THE ONLY AUTHORIZED ENTRY POINT
  🚫 Never create alternative main files
  🚫 Never bypass this entry point
```

### 🏗️ **โครงสร้างไดเรกทอรี**
```
ProjectP/
├── 🚀 ProjectP.py                    # MAIN ENTRY POINT (ONLY)
├── 📁 core/                          # Enterprise Core System
│   ├── 🎛️ menu_system.py             # Menu Management
│   ├── 📝 logger.py                  # Main Logger
│   ├── 🛡️ compliance.py              # Enterprise Rules
│   └── ⚙️ config.py                  # Configuration
│
├── 📁 elliott_wave_modules/          # AI/ML Components
│   ├── 📊 data_processor.py          # Data Processing
│   ├── 🧠 cnn_lstm_engine.py         # CNN-LSTM Model
│   ├── 🤖 dqn_agent.py               # DQN Agent
│   ├── 🎯 feature_selector.py        # SHAP + Optuna
│   └── 🎼 pipeline_orchestrator.py   # Pipeline Control
│
├── 📁 menu_modules/                  # Menu Implementations
│   └── 🌊 menu_1_elliott_wave.py     # Menu 1: Full Pipeline
│
├── 📁 datacsv/                       # Real Market Data
│   ├── 📊 XAUUSD_M1.csv             # 1-minute data
│   └── 📈 XAUUSD_M15.csv            # 15-minute data
│
├── 📁 config/                        # Configuration Files
├── 📁 outputs/                       # Generated Outputs
└── 📁 logs/                          # System Logs
```

---

## 🛡️ Enterprise Compliance Rules

### 🏢 **Enterprise Compliance Framework**
```yaml
Single Entry Point Policy:
  ✅ Only ProjectP.py authorized
  🚫 No alternative main files
  🚫 No bypassing main entry
  
Real Data Policy:
  ✅ datacsv/ real market data only
  🚫 No mock/simulation data
  🚫 No hard-coded values
  
Performance Standards:
  ✅ AUC ≥ 70% minimum
  ✅ Enterprise quality gates
  ✅ Production-ready code
  
Code Security:
  ✅ No time.sleep() simulation
  ✅ No fallback to simple methods
  ✅ Enterprise error handling
```

---

## 🌊 Main Feature: Menu 1 (Full Pipeline)

### 📊 **Pipeline Steps**
```python
Pipeline Steps:
  1. 📊 Real Data Loading (datacsv/)
  2. 🌊 Elliott Wave Detection
  3. ⚙️ Feature Engineering (50+ features)
  4. 🎯 ML Data Preparation
  5. 🧠 SHAP + Optuna Feature Selection
  6. 🏗️ CNN-LSTM Training
  7. 🤖 DQN Training
  8. 🔗 Model Integration
  9. 📈 Performance Analysis & Validation

Quality Gates:
  - Data quality validation
  - Feature selection validation (AUC ≥ 70%)
  - Model performance validation
  - Enterprise compliance checks
```

### 🔧 **Current Status**
```yaml
Status: 95% Production Ready
Blocking Issue: NumPy DLL compatibility (auto-fixing available)
Ready Components: All core systems, logging, menu system
Ready AI/ML: CNN-LSTM, DQN, Feature Engineering
Pending: SHAP + Optuna activation (after NumPy fix)
Solution: Menu option 'D' for automatic dependency fix
```

---

## 🔧 Dependency Management

### 📄 **Critical Dependencies**
```txt
numpy==1.26.4          # SHAP compatible version
pandas==2.2.3          # Data manipulation
tensorflow==2.17.0     # CNN-LSTM models
torch==2.4.1           # DQN models
scikit-learn==1.5.2    # ML utilities
shap==0.45.0           # Feature selection
optuna==3.5.0          # Hyperparameter optimization
```

### 🛠️ **Dependency Fix Process**
```bash
# Automatic fix (recommended)
python ProjectP.py -> Select 'D' (Dependency Check & Fix)

# Manual fix
python windows_numpy_fix.py  # For Windows NumPy issues
python ultimate_numpy_fix.py # Ultimate solution
```

---

## 📝 Logging System

### 📊 **Log Structure**
```
logs/
├── menu1/
│   ├── sessions/YYYYMMDD_HHMMSS/
│   │   ├── session_log.txt
│   │   ├── progress_log.json
│   │   └── performance_metrics.json
│   └── errors/
├── errors/
└── warnings/
```

### 🎯 **Progress Tracking**
```python
Progress Stages:
  🔄 Data Loading (10%)
  🌊 Elliott Wave Detection (20%)
  ⚙️ Feature Engineering (30%)
  🧠 Feature Selection (40%)
  🏗️ CNN-LSTM Training (60%)
  🤖 DQN Training (80%)
  📊 Performance Analysis (90%)
  ✅ Results Compilation (100%)
```

---

## 🚀 การใช้งาน (Usage)

### 📋 **Standard Operating Procedure**
```bash
# 1. System Startup
python ProjectP.py

# 2. First-Time Setup (if needed)
# Select 'D' for dependency check & fix
# Wait for automatic resolution (5-10 minutes)

# 3. Full Pipeline Execution
# Select '1' for Elliott Wave Full Pipeline
# Monitor progress through beautiful tracking
# Results saved automatically

# 4. Review Results
# Check outputs/ folder for session results
# Review logs/ for execution details
```

---

## 🎯 AI Development Guidelines

### ✅ **DO**
```python
✅ Always use ProjectP.py as main entry
✅ Use real data from datacsv/ only
✅ Implement SHAP + Optuna for feature selection
✅ Enforce AUC ≥ 70% targets
✅ Follow enterprise logging patterns
✅ Validate against enterprise standards
```

### 🚫 **DON'T**
```python
🚫 Create alternative main files
🚫 Use mock/simulation data
🚫 Use fallback to simple methods
🚫 Skip feature selection
🚫 Accept poor performance
🚫 Bypass enterprise compliance
```

---

## 📊 System Status Summary

```yaml
Current Status: 95% Production Ready
Ready Components: All core systems, logging, menu system
Ready AI/ML: CNN-LSTM, DQN, Feature Engineering
Ready Backtest: Menu 5 production ready backtest engine
Pending: SHAP + Optuna activation (after NumPy fix)
Next Steps: Complete dependency fix, full pipeline activation
```

---

## 🔍 Critical Understanding Points

1. **🚀 Single Entry Point**: ProjectP.py is the ONLY authorized entry point
2. **🛡️ Enterprise Compliance**: Real data only, AUC ≥ 70%, no simulation
3. **🌊 Menu 1 Priority**: Full Pipeline is the main feature (95% ready)
4. **🔧 Dependency Issue**: NumPy compatibility needs fixing (auto-fix available)
5. **📊 Data Policy**: Use datacsv/ real market data exclusively
6. **🧠 AI Stack**: CNN-LSTM + DQN + SHAP + Optuna integration
7. **📝 Logging**: Beautiful progress tracking and comprehensive reports
8. **⚙️ Configuration**: YAML-based enterprise configuration
9. **🔄 Pipeline**: 9-step process from data to deployment
10. **📈 Performance**: Target AUC ≥ 70% with enterprise quality gates

---

**หมายเหตุ:** นี่คือคู่มือหลักสำหรับ AI Agent เพื่อเข้าใจระบบ NICEGOLD ProjectP อย่างสมบูรณ์ 100%
