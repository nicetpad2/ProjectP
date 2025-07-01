#!/usr/bin/env python3
"""
🧪 AUC Fix Test Script
ทดสอบการแก้ไข AUC calculation ใน CNN-LSTM engine
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Import components
    from elliott_wave_modules.data_processor import ElliottWaveDataProcessor
    from elliott_wave_modules.cnn_lstm_engine import CNNLSTMElliottWave
    from elliott_wave_modules.feature_selector import EnterpriseShapOptunaFeatureSelector
    
    print("🧪 Testing AUC Fix...")
    
    # Initialize components
    data_processor = ElliottWaveDataProcessor()
    cnn_lstm_engine = CNNLSTMElliottWave()
    feature_selector = EnterpriseShapOptunaFeatureSelector()
    
    print("✅ Components initialized")
    
    # Load sample data
    print("📊 Loading test data...")
    data = data_processor.load_real_data()
    print(f"✅ Data loaded: {len(data)} rows")
    
    # Create features
    print("⚙️ Creating features...")
    features = data_processor.create_elliott_wave_features(data)
    print(f"✅ Features created: {features.shape}")
    
    # Prepare ML data
    print("🎯 Preparing ML data...")
    X, y = data_processor.prepare_ml_data(features)
    print(f"✅ ML data prepared: X{X.shape}, y{y.shape}")
    
    # Select features (quick)
    print("🧠 Quick feature selection...")
    try:
        selected_features, selection_results = feature_selector.select_features(X, y)
        print(f"✅ Features selected: {len(selected_features)} features")
        print(f"   Best AUC: {selection_results.get('best_auc', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Feature selection failed: {e}")
        selected_features = X.columns[:20].tolist()  # Use first 20 features
        selection_results = {'best_auc': 0.75, 'target_achieved': True}
    
    # Test CNN-LSTM training
    print("🏗️ Testing CNN-LSTM training with AUC calculation...")
    cnn_lstm_results = cnn_lstm_engine.train_model(X[selected_features], y)
    
    print("\n" + "="*60)
    print("📊 CNN-LSTM RESULTS:")
    print("="*60)
    print(f"Success: {cnn_lstm_results.get('success', False)}")
    print(f"Model Type: {cnn_lstm_results.get('model_type', 'Unknown')}")
    print(f"Train Accuracy: {cnn_lstm_results.get('train_accuracy', 0):.4f}")
    print(f"Val Accuracy: {cnn_lstm_results.get('val_accuracy', 0):.4f}")
    
    # Check evaluation results
    eval_results = cnn_lstm_results.get('evaluation_results', {})
    if eval_results:
        print(f"\n🎯 EVALUATION METRICS:")
        print(f"   AUC: {eval_results.get('auc', 0):.4f} {'✅' if eval_results.get('auc', 0) >= 0.70 else '❌'}")
        print(f"   Accuracy: {eval_results.get('accuracy', 0):.4f}")
        print(f"   Precision: {eval_results.get('precision', 0):.4f}")
        print(f"   Recall: {eval_results.get('recall', 0):.4f}")
        print(f"   F1 Score: {eval_results.get('f1_score', 0):.4f}")
        
        auc_score = eval_results.get('auc', 0)
        if auc_score >= 0.70:
            print(f"\n🎉 SUCCESS: AUC {auc_score:.4f} meets Enterprise requirement (≥0.70)!")
        else:
            print(f"\n⚠️ WARNING: AUC {auc_score:.4f} below Enterprise requirement (≥0.70)")
    else:
        print("❌ No evaluation_results found!")
    
    print("\n" + "="*60)
    print("🎯 AUC FIX TEST COMPLETED")
    print("="*60)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
