"""Cihaz seçimi: cuda (ROCm/CUDA), dml (Windows DirectML), cpu."""

import torch


def get_device(name=None) -> torch.device:
    if name == "dml":
        import torch_directml  # pip install torch-directml
        return torch_directml.device()
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


def is_dml(device: torch.device) -> bool:
    return device.type == "privateuseone"
