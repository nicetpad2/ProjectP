
"""
🛠️ TensorFlow CPU-Only Import Module
Safe TensorFlow import for NICEGOLD ProjectP
"""

import os
import warnings

# Force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def safe_tensorflow():
    """Import TensorFlow safely for CPU-only operation"""
    try:
        import tensorflow as tf
        
        # Configure for CPU only
        tf.config.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('ERROR')
        
        return tf
    except Exception as e:
        print(f"TensorFlow import error: {e}")
        return None

# Safe import
tf = safe_tensorflow()
