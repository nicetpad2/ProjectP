# 📁 NICEGOLD ProjectP - Project Structure Map

> **อัพเดตล่าสุด:** 2025-06-30

---

## โครงสร้างโปรเจค (Project Directory Structure)

```
ProjectP/
├── ProjectP.py                        # Main entry point (menu system)
├── requirements.txt                   # Production dependencies
├── requirements_explanation.md        # Dependency explanations
├── install_all.sh                     # Linux/Mac auto-install script
├── install_all.ps1                    # Windows auto-install script
├── install_all.py                     # Python fallback installer
├── MENU1_ELLWAVE_SHAP_OPTUNA_PLAN.md  # Menu 1 integration plan (this file)
├── PROJECT_STRUCTURE.md               # <== THIS FILE (Project structure map)
│
├── core/
│   ├── __init__.py
│   ├── compliance.py
│   ├── config.py
│   ├── logger.py
│   ├── menu_system.py                 # Menu system logic
│   └── ...
│
├── elliott_wave_modules/
│   ├── __init__.py
│   ├── cnn_lstm_engine.py
│   ├── cnn_lstm_engine_clean.py
│   ├── data_processor.py
│   ├── dqn_agent.py
│   ├── feature_selector.py
│   ├── performance_analyzer.py
│   ├── pipeline_orchestrator.py
│   └── ...
│
├── menu_modules/
│   ├── __init__.py
│   ├── menu_1_elliott_wave.py         # Menu 1 logic
│   └── ...
│
├── logs/
│   └── nicegold_enterprise_20250630.log
│
├── config/
│   ├── enterprise_config.yaml
│   └── ...
│
├── dashboard_modules/                 # (If dashboard modularized)
│   ├── __init__.py
│   ├── ui_components.py
│   ├── pipeline_runner.py
│   ├── backtest_engine.py
│   ├── data_manager.py
│   └── report_generator.py
│
├── src/                               # (If advanced/legacy code present)
│   ├── cnn_lstm_elliott_wave.py
│   ├── dqn_reinforcement_agent.py
│   ├── integrated_elliott_wave_dqn.py
│   └── ...
│
├── models/                            # (Trained models, auto-generated)
│   └── ...
│
├── datacsv/                           # (Real data CSVs)
│   └── ...
│
├── results/                           # (Backtest, reports, analysis)
│   └── ...
│
├── temp_dir/                          # (Temporary files)
│   └── ...
│
├── pip_cache/                         # (pip cache)
│   └── ...
│
└── ... (other folders/files as created)
```

---

## กฎการอัพเดต
- **ไฟล์นี้ต้องอัพเดตทุกครั้ง** เมื่อมีการสร้าง/ลบ/ย้ายไฟล์หรือโฟลเดอร์ใหม่ในโปรเจค
- ให้ระบุไฟล์ใหม่, ตำแหน่ง, และหน้าที่โดยย่อ
- หากมีโมดูลใหม่หรือไฟล์ขนาดใหญ่ (>2000 บรรทัด) ให้แยกโฟลเดอร์และอัพเดตที่นี่
- หากมี dashboard, pipeline, หรือ menu module ใหม่ ให้เพิ่มในโครงสร้างนี้

---

## หมายเหตุ
- ใช้ไฟล์นี้เป็นแผนที่หลักสำหรับการพัฒนา, review, และ onboarding
- ทุกโมดูลควรมี README/MD อธิบายหน้าที่และการเชื่อมโยง
- หากมีการเปลี่ยนแปลงโครงสร้าง ให้ commit ไฟล์นี้พร้อมกันทุกครั้ง

---

*สำหรับการอัพเดต: ให้เพิ่มไฟล์/โฟลเดอร์ใหม่ในแผนผังนี้ พร้อมคำอธิบายสั้นๆ*
