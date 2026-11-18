from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed() -> None:
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()
