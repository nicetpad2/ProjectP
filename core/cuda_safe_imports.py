
# CUDA-Safe ML Imports for NICEGOLD ProjectP
import os
import warnings

# Force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress warnings
warnings.filterwarnings('ignore')

def safe_tensorflow_import():
    """Import TensorFlow safely without CUDA errors"""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        tf.get_logger().setLevel('ERROR')
        return tf
    except Exception as e:
        print(f"TensorFlow import error: {e}")
        return None

def safe_pytorch_import():
    """Import PyTorch safely without CUDA errors"""
    try:
        import torch
        torch.set_default_tensor_type('torch.FloatTensor')
        return torch
    except Exception as e:
        print(f"PyTorch import error: {e}")
        return None

def safe_sklearn_import():
    """Import scikit-learn safely"""
    try:
        import sklearn
        return sklearn
    except Exception as e:
        print(f"Scikit-learn import error: {e}")
        return None
