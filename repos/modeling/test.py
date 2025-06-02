# test_cuda.py
from __future__ import annotations

import os

import torch

print("=== Environment Check ===")
print(f"NVIDIA_VISIBLE_DEVICES: {os.getenv('NVIDIA_VISIBLE_DEVICES', 'Not set')}")
print(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH', 'Not set')}")

print("\n=== PyTorch CUDA Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
else:
    print("CUDA not available - checking why...")
    try:
        # Try to manually load CUDA
        torch.cuda.init()
        print("CUDA init successful")
    except Exception as e:
        print(f"CUDA init failed: {e}")
