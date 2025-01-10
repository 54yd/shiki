"""
"""

from .static import CUDA_MEM_GET_INFO


def cudaMemGetInfo(device: int, /):
    return CUDA_MEM_GET_INFO
