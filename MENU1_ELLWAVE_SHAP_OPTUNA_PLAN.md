# 📈 NICEGOLD ProjectP Menu 1: ELLIOTT WAVE + SHAP+Optuna Integration Plan

## 🎯 เป้าหมาย
- รวมพลัง Elliott Wave (CNN-LSTM + DQN) กับ SHAP + Optuna AutoTune Feature Selection, Walk-Forward Validation, Anti-Overfitting, และ AUC ≥ 0.70
- ให้ได้ Full Pipeline ที่สมบูรณ์แบบระดับ Enterprise, ไม่มี mock/dummy/simulation, ใช้ข้อมูลจริง 100%
- ตรงตาม compliance และ enterprise standard ทุกข้อ

---

## 1. โครงสร้าง Pipeline ใหม่ (Menu 1)

```
Raw Data
  ↓
Feature Engineering (Elliott Wave, TA, Price Action, etc.)
  ↓
SHAP + Optuna Feature Selection
  ↓
Selected Features → CNN-LSTM (Elliott Wave) + DQN Agent
  ↓
Walk-Forward Validation (TimeSeriesSplit)
  ↓
Performance Gate (AUC ≥ 0.70, Overfitting Check)
  ↓
Model Save/Deploy
```

---

## 2. ขั้นตอนการพัฒนา

### 2.1 Feature Engineering
- รวมฟีเจอร์จาก Elliott Wave, Technical Indicators, Price Action, ฯลฯ
- เตรียมข้อมูลสำหรับ SHAP/Optuna และ Deep Learning

### 2.2 SHAP + Optuna Feature Selection
- ใช้ SHAP วิเคราะห์ความสำคัญของฟีเจอร์
- ใช้ Optuna หา subset ฟีเจอร์ที่ดีที่สุด (feature selection + hyperparameter tuning)
- ผลลัพธ์: ได้ชุดฟีเจอร์ที่เหมาะสมที่สุดสำหรับโมเดล

### 2.3 Model Training
- ป้อน selected features เข้า CNN-LSTM (Elliott Wave) และ/หรือ DQN agent
- ใช้ Optuna ปรับ hyperparameter ของ deep learning model ได้ด้วย

### 2.4 Walk-Forward Validation
- ใช้ TimeSeriesSplit ในทุกขั้นตอน (feature selection, model training, evaluation)
- ตรวจสอบ AUC ≥ 0.70, ตรวจสอบ overfitting, และ noise reduction

### 2.5 Quality Gate & Compliance
- ถ้า AUC < 0.70 หรือ overfitting > 10%: ไม่ deploy, แจ้งเตือน
- ถ้า pass: save model, export feature importance, log metadata

---

## 3. ตัวอย่างโค้ด (Pseudo-Workflow)

```python
# 1. Feature Engineering
features = engineer_features(raw_data)  # รวม Elliott Wave, TA, etc.

# 2. SHAP + Optuna Feature Selection
selector = SHAPOptunaFeatureSelector(target_auc=0.70, max_features=30)
selected_features, results = selector.select_features(features, y)

# 3. Model Training (Elliott Wave CNN-LSTM)
model = train_cnn_lstm_elliott_wave(features[selected_features], y, optuna_params)

# 4. Walk-Forward Validation
auc_scores = walk_forward_validate(model, features[selected_features], y)

# 5. Quality Gate
if np.mean(auc_scores) >= 0.70:
    save_model(model)
else:
    raise Exception("AUC < 0.70: Model not production ready")
```

---

## 4. Compliance Checklist
- [x] ไม่มี mock, dummy, simulation, time.sleep, หรือ fallback ใดๆ
- [x] ใช้ข้อมูลจริง 100% ทุกขั้นตอน
- [x] ใช้ SHAP+Optuna feature selection ก่อน deep learning
- [x] Walk-Forward Validation ทุกขั้นตอน
- [x] AUC ≥ 0.70 เท่านั้น
- [x] แยกโมดูล/โค้ดตามหลัก modular architecture
- [x] มี markdown doc อธิบายโครงสร้างและการเชื่อมโยง

---

## 5. Roadmap & Next Steps
1. Refactor feature engineering ให้รองรับฟีเจอร์จากทุกแหล่ง (Elliott Wave, TA, etc.)
2. Integrate SHAP+Optuna selector ก่อน training deep model
3. ปรับ training pipeline ให้รับ selected features
4. เพิ่ม Walk-Forward Validation และ Quality Gate
5. ทดสอบกับข้อมูลจริง, ตรวจสอบ compliance ทุกข้อ
6. อัปเดตเอกสารและตัวอย่างการใช้งาน

---

## 6. หมายเหตุ
- ทุกขั้นตอนต้อง production-ready, ไม่มี fallback/simple pipeline
- หากไฟล์ใดเกิน 2000 บรรทัด ให้แยกเป็นโมดูลใหม่และมีไฟล์ README อธิบาย
- Dashboard/แดชบอร์ดต้องแยก logic UI, pipeline, backtest, data manager, report

---

**Status:** วางแผนสมบูรณ์ พร้อมลงมือพัฒนา/Refactor ทันที

---

*Update: 2025-06-30*
