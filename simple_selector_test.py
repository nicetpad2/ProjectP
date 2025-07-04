#!/usr/bin/env python3
"""Simple test for feature selectors"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🧪 Simple Feature Selector Test")
    
    # Create tiny test data
    np.random.seed(42)
    X = pd.DataFrame({
        'f1': np.random.normal(0, 1, 100),
        'f2': np.random.normal(0, 1, 100),
        'f3': np.random.normal(0, 1, 100),
        'f4': np.random.normal(0, 1, 100),
        'f5': np.random.normal(0, 1, 100)
    })
    y = (X['f1'] + X['f2'] + np.random.normal(0, 0.1, 100) > 0).astype(int)
    
    print(f"✅ Test data: {X.shape}")
    
    try:
        from fast_feature_selector import FastEnterpriseFeatureSelector
        
        selector = FastEnterpriseFeatureSelector(target_auc=0.6, max_features=3)
        features, results = selector.select_features(X, y)
        
        print(f"✅ Fast selector: {len(features)} features")
        print(f"   Results keys: {list(results.keys())}")
        print(f"   AUC: {results.get('final_auc', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Fast selector failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
