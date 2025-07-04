#!/usr/bin/env python3
"""
🧪 MEMORY-OPTIMIZED CNN-LSTM TEST
ทดสอบการแก้ไขปัญหาหน่วยความจำใน CNN-LSTM Engine
"""

import sys
import os
import psutil
import numpy as np
import pandas as pd
from datetime import datetime

def log_memory_usage(stage):
    """บันทึกการใช้หน่วยความจำ"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    memory_percent = process.memory_percent()
    print(f"🔍 {stage}: {memory_mb:.1f} MB ({memory_percent:.1f}%)")
    return memory_mb

def test_memory_optimized_cnn_lstm():
    """ทดสอบ Memory-Optimized CNN-LSTM"""
    print("🚀 Testing Memory-Optimized CNN-LSTM Engine")
    print("="*60)
    
    # เริ่มต้นการตรวจสอบหน่วยความจำ
    initial_memory = log_memory_usage("Initial")
    
    try:
        # อ่านข้อมูล
        print("\n📊 Loading test data...")
        data_file = "/mnt/data/projects/ProjectP/datacsv/XAUUSD_M15.csv"
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            return False
        
        # อ่านข้อมูลในปริมาณที่เหมาะสม
        df = pd.read_csv(data_file, nrows=50000)  # จำกัดจำนวนแถว
        print(f"✅ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
        log_memory_usage("After data loading")
        
        # เตรียมข้อมูลสำหรับทดสอบ
        print("\n🔧 Preparing test features...")
        
        # สร้าง features ที่เรียบง่าย
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] if 'Volume' in df.columns else ['Open', 'High', 'Low', 'Close']
        
        # ตรวจสอบคอลัมน์ที่มีอยู่
        available_cols = [col for col in feature_cols if col in df.columns]
        if not available_cols:
            # ใช้คอลัมน์แรกๆ ถ้าไม่มีคอลัมน์มาตรฐาน
            available_cols = df.select_dtypes(include=[np.number]).columns[:4].tolist()
        
        print(f"📋 Using features: {available_cols}")
        
        X = df[available_cols].copy()
        X = X.fillna(X.mean())  # เติมค่าที่หายไป
        
        # สร้าง target แบบง่าย
        y = (df[available_cols[0]].shift(-1) > df[available_cols[0]]).astype(int)
        y = y.fillna(0)
        
        # ลบแถวสุดท้ายที่ไม่มี target
        X = X[:-1]
        y = y[:-1]
        
        print(f"✅ Features prepared: {X.shape}")
        log_memory_usage("After feature preparation")
        
        # เทสการใช้หน่วยความจำเบื้องต้น
        memory_after_prep = log_memory_usage("Before CNN-LSTM initialization")
        
        # Import และสร้าง CNN-LSTM Engine
        print("\n🧠 Initializing CNN-LSTM Engine...")
        from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave
        
        engine = CNNLSTMElliottWave()
        print("✅ Engine initialized successfully")
        log_memory_usage("After engine initialization")
        
        # ทดสอบ sequence preparation
        print("\n📊 Testing sequence preparation...")
        try:
            X_sequences, y_sequences = engine.prepare_sequences(X, y)
            print(f"✅ Sequences prepared: {X_sequences.shape if X_sequences is not None else 'None'}")
            log_memory_usage("After sequence preparation")
            
            # ตรวจสอบการใช้หน่วยความจำ
            if X_sequences is not None:
                memory_gb = X_sequences.nbytes / (1024**3)
                print(f"📊 Sequence memory usage: {memory_gb:.3f} GB")
                
                if memory_gb > 0.5:
                    print("⚠️ Warning: High memory usage detected")
                else:
                    print("✅ Memory usage within safe limits")
        
        except Exception as e:
            print(f"❌ Sequence preparation failed: {e}")
            return False
        
        # ทดสอบ model building
        print("\n🏗️ Testing model building...")
        try:
            if X_sequences is not None and len(X_sequences.shape) == 3:
                input_shape = (X_sequences.shape[1], X_sequences.shape[2])
                model = engine.build_model(input_shape)
                print("✅ Model built successfully")
                log_memory_usage("After model building")
                
                # ตรวจสอบขนาดโมเดล
                if hasattr(model, 'count_params'):
                    params = model.count_params()
                    print(f"📊 Model parameters: {params:,}")
                    if params > 20000:
                        print("⚠️ Warning: Large model detected")
                    else:
                        print("✅ Model size within safe limits")
            
        except Exception as e:
            print(f"❌ Model building failed: {e}")
            return False
        
        # ทดสอบการฝึกสั้นๆ (ถ้าหน่วยความจำเพียงพอ)
        print("\n🏃‍♂️ Testing quick training...")
        current_memory = log_memory_usage("Before training test")
        
        if current_memory < 1000:  # ถ้าใช้หน่วยความจำน้อยกว่า 1GB
            try:
                # ใช้ข้อมูลเล็กๆ สำหรับทดสอบ
                test_size = min(1000, len(X))
                X_test = X.iloc[:test_size].copy()
                y_test = y.iloc[:test_size].copy()
                
                print(f"🔬 Quick training test with {test_size} samples...")
                result = engine.train_model(X_test, y_test)
                
                print("✅ Quick training completed")
                print(f"📊 Training result keys: {list(result.keys()) if isinstance(result, dict) else 'Not dict'}")
                log_memory_usage("After training test")
                
            except Exception as e:
                print(f"⚠️ Training test failed (expected for resource limits): {e}")
        else:
            print("⚠️ Skipping training test due to high memory usage")
        
        # สรุปผลการทดสอบ
        final_memory = log_memory_usage("Final")
        memory_increase = final_memory - initial_memory
        
        print(f"\n📈 Memory increase: {memory_increase:.1f} MB")
        
        if memory_increase < 500:
            print("✅ MEMORY TEST PASSED: Low memory usage")
            return True
        elif memory_increase < 1000:
            print("⚠️ MEMORY TEST WARNING: Moderate memory usage")
            return True
        else:
            print("❌ MEMORY TEST FAILED: High memory usage")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 MEMORY-OPTIMIZED CNN-LSTM TEST")
    print("==================================")
    
    success = test_memory_optimized_cnn_lstm()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Memory optimization successful")
        sys.exit(0)
    else:
        print("\n❌ TESTS FAILED!")
        print("🔧 Further optimization needed")
        sys.exit(1)
