import argparse
import pprint
import torch
import random
import numpy as np
import os
import time
import socket
from datetime import datetime
import logging
from loguru import logger
import psutil
from typing import Optional

try:
    # Provided by the package `nvidia-ml-py` or `pynvml`
    import pynvml  # type: ignore
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def llama_down_proj_groupsize(model, groupsize):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logger.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logger.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def cleanup_memory(verbos=True) -> None:
    
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logger.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def _nvml_gpu_mem() -> list:
    """Return per-GPU memory (total, used, free) using NVML if available.
    Falls back to torch.cuda.mem_get_info per device if NVML isn't present.
    """
    results = []
    if not torch.cuda.is_available():
        return results

    device_count = torch.cuda.device_count()
    if _HAS_NVML:
        try:
            pynvml.nvmlInit()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                results.append({
                    'index': i,
                    'total': int(mem.total),
                    'used': int(mem.used),
                    'free': int(mem.free),
                })
        except Exception:
            # Fallback silently to torch-based query
            results.clear()
    if not results:
        for i in range(device_count):
            try:
                with torch.cuda.device(i):
                    free, total = torch.cuda.mem_get_info()
                used = int(total - free)
                results.append({
                    'index': i,
                    'total': int(total),
                    'used': int(used),
                    'free': int(free),
                })
            except Exception:
                results.append({
                    'index': i,
                    'total': 0,
                    'used': 0,
                    'free': 0,
                })
    return results


def _torch_gpu_proc_mem() -> list:
    """Return torch memory stats per device for this process."""
    stats = []
    if not torch.cuda.is_available():
        return stats
    for i in range(torch.cuda.device_count()):
        try:
            allocated = int(torch.cuda.memory_allocated(i))
            reserved = int(torch.cuda.memory_reserved(i))
            max_alloc = int(torch.cuda.max_memory_allocated(i))
        except Exception:
            allocated = reserved = max_alloc = 0
        stats.append({
            'index': i,
            'allocated': allocated,
            'reserved': reserved,
            'max_allocated': max_alloc,
        })
    return stats


def _cpu_mem() -> dict:
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    rss = int(proc.memory_info().rss)
    return {
        'system_total': int(vm.total),
        'system_used': int(vm.total - vm.available),
        'system_available': int(vm.available),
        'process_rss': rss,
    }


def _rank_info() -> dict:
    node = socket.gethostname()
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', '0')))
    world_rank = None
    world_size = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            world_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        except Exception:
            pass
    return {
        'node': node,
        'local_rank': local_rank,
        'world_rank': world_rank,
        'world_size': world_size,
    }


def distributed_memory_snapshot(phase: str, output_dir: Optional[str] = None, barrier: bool = True) -> None:
    """
    Capture and print memory usage across all ranks and GPUs.

    - Prints per-GPU NVML memory and per-process torch memory stats
    - Prints CPU system memory and per-process RSS
    - Aggregates on rank 0 and prints a concise summary (no file output)

    Parameters
    ----------
    phase: str
        Short label for the code section being measured.
    output_dir: Optional[str]
        Directory to write memory_profile.jsonl. If None, current directory.
    barrier: bool
        Whether to synchronize ranks before/after snapshot.
    """
    ts = time.time()
    if torch.distributed.is_available() and torch.distributed.is_initialized() and barrier:
        try:
            torch.distributed.barrier()
        except Exception:
            pass

    record = {
        'phase': phase,
        'timestamp': ts,
        'rank': _rank_info(),
        'gpu_nvml': _nvml_gpu_mem(),
        'gpu_torch': _torch_gpu_proc_mem(),
        'cpu': _cpu_mem(),
    }

    gathered = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, record)
        except Exception:
            gathered = [record]
    else:
        gathered = [record]

    # Only rank 0 writes/logs aggregated results
    is_rank0 = True
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            is_rank0 = torch.distributed.get_rank() == 0
        except Exception:
            is_rank0 = True

    if is_rank0:
        # Human-readable summary
        def fmt_bytes(b: int) -> str:
            return f"{b/1024**3:.2f} GB"

        rows = []
        for rec in gathered:
            rank = rec['rank']['world_rank']
            node = rec['rank']['node']
            cpu = rec['cpu']
            for i, gpu in enumerate(rec['gpu_nvml']):
                torch_stat = next((s for s in rec['gpu_torch'] if s['index'] == gpu['index']), None)
                rows.append({
                    'node': node,
                    'rank': rank,
                    'gpu': gpu['index'],
                    'gpu_used': gpu['used'],
                    'gpu_total': gpu['total'],
                    'torch_alloc': 0 if torch_stat is None else torch_stat['allocated'],
                    'torch_reserved': 0 if torch_stat is None else torch_stat['reserved'],
                    'torch_max_alloc': 0 if torch_stat is None else torch_stat['max_allocated'],
                    'proc_rss': cpu['process_rss'],
                    'sys_used': cpu['system_used'],
                    'sys_total': cpu['system_total'],
                })

        # Compute simple bottleneck heuristic
        gpu_ratios = []
        cpu_ratios = []
        for r in rows:
            if r['gpu_total'] > 0:
                gpu_ratios.append(r['gpu_used']/r['gpu_total'])
            if r['sys_total'] > 0:
                cpu_ratios.append(r['sys_used']/r['sys_total'])
        max_gpu = max(gpu_ratios) if gpu_ratios else 0.0
        max_cpu = max(cpu_ratios) if cpu_ratios else 0.0
        bottleneck = 'gpu' if max_gpu >= 0.90 and max_gpu >= max_cpu else ('cpu' if max_cpu >= 0.90 else 'unknown')

        summary = f"[Memory] phase={phase} ranks={len(gathered)} max_gpu_usage={max_gpu:.2%} max_cpu_usage={max_cpu:.2%} bottleneck={bottleneck}"
        print(summary)
        logger.info(summary)
        for r in rows:
            line = (
                f"node={r['node']} rank={r['rank']} gpu={r['gpu']} "
                f"nvml used/total={fmt_bytes(r['gpu_used'])}/{fmt_bytes(r['gpu_total'])} "
                f"torch alloc/resv/max={fmt_bytes(r['torch_alloc'])}/{fmt_bytes(r['torch_reserved'])}/{fmt_bytes(r['torch_max_alloc'])} "
                f"proc_rss={fmt_bytes(r['proc_rss'])}"
            )
            print(line)
            logger.info(line)

    if torch.distributed.is_available() and torch.distributed.is_initialized() and barrier:
        try:
            torch.distributed.barrier()
        except Exception:
            pass

def distribute_model(model) -> None:
    
    no_split_module_classes = ['LlamaDecoderLayer','QuantDecoderLayer',"RotateModule","SmoothModule"]
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )
    restore_manifold(model)
    cleanup_memory()

def restore_manifold(model) -> None:
    from geoopt import ManifoldTensor,Stiefel
    for p in model.parameters():
        if isinstance(p,ManifoldTensor):
            p.manifold = Stiefel()