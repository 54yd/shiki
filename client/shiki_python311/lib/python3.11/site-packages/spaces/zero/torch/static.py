"""
"""

from types import SimpleNamespace as _SimpleNamespace

import torch as _torch


# Nvidia A100.80G MIG (drivers 535) / Torch 2.2.0
CUDA_DEVICE_NAME = 'NVIDIA A100-SXM4-80GB MIG 3g.40gb'
CUDA_TOTAL_MEMORY = 42144366592
CUDA_MEM_GET_INFO = (41911451648, CUDA_TOTAL_MEMORY)
CUDA_DEVICE_CAPABILITY = (8, 0)
CUDA_DEVICE_PROPERTIES = _SimpleNamespace(name=CUDA_DEVICE_NAME, major=8, minor=0, total_memory=CUDA_TOTAL_MEMORY, multi_processor_count=42)

if _torch.version.cuda.startswith("12."): # pyright: ignore [reportAttributeAccessIssue]
    CUDA_MEMORY_STATS_AS_NESTED_DICT = {
        "num_alloc_retries": 0,
        "num_ooms": 0,
        "max_split_size": -1,
        "num_sync_all_streams": 0,
        "num_device_alloc": 0,
        "num_device_free": 0,
        "allocation": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "segment": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "allocated_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "reserved_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "requested_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "oversize_allocations": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        "oversize_segments": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
    }
else: # pragma: no cover (CUDA 11)
    CUDA_MEMORY_STATS_AS_NESTED_DICT = {
        "num_alloc_retries": 0,
        "num_ooms": 0,
        "max_split_size": -1,
        "allocation": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "segment": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "allocated_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "reserved_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "active_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "inactive_split_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "requested_bytes": {
            "all": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "small_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
            "large_pool": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        },
        "oversize_allocations": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
        "oversize_segments": {"current": 0, "peak": 0, "allocated": 0, "freed": 0},
    }
