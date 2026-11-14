"""Standalone ctypes launcher; prepared metadata and buffers stay caller-owned."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import torch

_lib = None


def library():
    global _lib
    if _lib is None:
        path = os.environ.get("VLLM_SM120_KATT_LIBRARY", str(Path(__file__).with_name("libkatt.so")))
        lib = C.CDLL(path)
        ptr, integer, long = C.c_void_p, C.c_int, C.c_int64
        lib.katt_run.argtypes = [ptr]*8 + [integer]*3 + [long]*10 + [C.c_float]*2 + [ptr, integer]
        lib.katt_run.restype = integer
        lib.katt_dequant.argtypes = [ptr]*4 + [integer]*2 + [long]*6 + [ptr]
        lib.katt_dequant.restype = integer
        lib.katt_gather.argtypes = [ptr]*5 + [integer]*2 + [long]*6 + [ptr]
        lib.katt_gather.restype = integer
        lib.katt_init.argtypes = []
        lib.katt_init.restype = integer
        err = lib.katt_init()
        if err:
            raise RuntimeError(f"katt_init CUDA error {err}")
        _lib = lib
    return _lib


def run(q, k, v, out, qo, indptr, pages, last, max_q, page_size,
        sm_scale=1/16, v_scale=1.0, tile_m=64):
    lib = library()
    err = lib.katt_run(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                       qo.data_ptr(), indptr.data_ptr(), pages.data_ptr(), last.data_ptr(),
                       qo.numel()-1, max_q, page_size, q.stride(0), q.stride(1),
                       *k.stride()[:3], *v.stride()[:3], out.stride(0), out.stride(1),
                       sm_scale, v_scale, torch.cuda.current_stream().cuda_stream, tile_m)
    if err:
        raise RuntimeError(f"katt_run CUDA error {err}")
    return out


def dequant(k, v, ko, vo, page_size):
    lib = library()
    err = lib.katt_dequant(k.data_ptr(), v.data_ptr(), ko.data_ptr(), vo.data_ptr(),
                           k.shape[0], page_size, *k.stride()[:3], *v.stride()[:3],
                           torch.cuda.current_stream().cuda_stream)
    if err:
        raise RuntimeError(f"katt_dequant CUDA error {err}")


def gather(k, v, ko, vo, pages, num_pages, page_size):
    lib = library()
    err = lib.katt_gather(k.data_ptr(), v.data_ptr(), ko.data_ptr(), vo.data_ptr(),
                          pages.data_ptr(), num_pages, page_size,
                          *k.stride()[:3], *v.stride()[:3],
                          torch.cuda.current_stream().cuda_stream)
    if err:
        raise RuntimeError(f"katt_gather CUDA error {err}")
