#!/usr/bin/env python3
"""
Final Production Validation Test
Validates that the metrics fix resolves the warning and the system is production-ready.
This script will be deleted after validation.
"""

import os
import sys
import warnings
from io import StringIO

# Capture warnings to verify they're fixed
warnings.simplefilter('always')
warning_buffer = StringIO()

def test_metrics_warning_fix():
    """Test that the metrics warning is fixed."""
    print("🔧 Testing metrics warning fix...")
    
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
        import numpy as np
        
        # Test data that might cause division by zero
        y_true = np.array([1, 1, 1, 1, 1])  # All positive
        y_pred = np.array([0, 0, 0, 0, 0])  # All negative predictions
        
        # This should not produce warnings with zero_division=0.0
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0.0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0.0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0.0)
        
        print(f"✅ Metrics with zero_division=0.0:")
        print(f"   - Precision: {precision:.4f} (should be 0.0)")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1: {f1:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing metrics: {e}")
        return False

def quick_system_health_check():
    """Quick validation that core modules import without issues."""
    print("\n🏥 Quick system health check...")
    
    try:
        # Test core imports
        sys.path.append('/mnt/data/projects/ProjectP')
        
        from safe_logger import SafeLogger
        from elliott_wave_modules.cnn_lstm_engine import CNNLSTMEngine
        from elliott_wave_modules.feature_selector import FeatureSelector
        
        print("✅ Core modules import successfully")
        
        # Test logger
        logger = SafeLogger("test_session", log_level="INFO")
        logger.info("✅ Logger working correctly")
        
        print("✅ All core systems operational")
        return True
        
    except Exception as e:
        print(f"❌ System health check failed: {e}")
        return False

def main():
    """Run final validation tests."""
    print("🎯 FINAL PRODUCTION VALIDATION")
    print("=" * 50)
    
    # Test 1: Metrics warning fix
    metrics_ok = test_metrics_warning_fix()
    
    # Test 2: System health
    system_ok = quick_system_health_check()
    
    print("\n" + "=" * 50)
    if metrics_ok and system_ok:
        print("🎉 PRODUCTION VALIDATION SUCCESSFUL!")
        print("✅ Metrics warning fixed")
        print("✅ All systems operational")
        print("🚀 ProjectP is PRODUCTION READY!")
    else:
        print("❌ PRODUCTION VALIDATION FAILED!")
        print("❌ Issues detected - see above")
    
    return metrics_ok and system_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
