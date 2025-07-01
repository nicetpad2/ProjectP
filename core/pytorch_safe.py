
"""
🛠️ PyTorch CPU-Only Import Module
Safe PyTorch import for NICEGOLD ProjectP
"""

import os

# Force CPU-only operation
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def safe_pytorch():
    """Import PyTorch safely for CPU-only operation"""
    try:
        import torch
        
        # Set default to CPU
        torch.set_default_tensor_type('torch.FloatTensor')
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return torch
    except Exception as e:
        print(f"PyTorch import error: {e}")
        return None

# Safe import
torch = safe_pytorch()
