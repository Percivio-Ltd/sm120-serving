"""Class B: compact exact FP8 widening followed by pinned BF16 FA2.

Imported only by the env-gated vLLM prefill construction. Decode is unaffected.
The wrapper owns its scratch, and plans before capture. run() has no allocation
or host/device synchronization. All KV scale placement remains in FlashInfer.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from vllm.logger import init_logger

logger = init_logger("vllm.v1.attention.backends.flashinfer")
_POOLS = {}
_WORKSPACES = {}


def reserve_pool(device, capacity_pages, phase):
    device = torch.device(device)
    key = (device.index, capacity_pages)
    if key not in _POOLS:
        k = torch.empty((capacity_pages,4,16,256),dtype=torch.bfloat16,device=device)
        v = torch.empty_like(k)
        ids = torch.arange(capacity_pages,dtype=torch.int32,device=device)
        _POOLS[key] = (k,v,ids)
        logger.info("KATT2_RESERVE phase=%s staging_bytes=%d index_bytes=%d device=%s",
                    phase,k.numel()*4,ids.numel()*4,device)
    return _POOLS[key]


def reserve_from_model(config):
    capacity = config.scheduler_config.max_num_seqs * ((config.model_config.max_model_len+63)//64)*4
    device = torch.device("cuda",torch.cuda.current_device())
    pool = reserve_pool(device,capacity,"model_load")
    key = (device.index,capacity)
    if key not in _WORKSPACES:
        from vllm import envs
        size = max(envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE,
                   config.scheduler_config.max_num_batched_tokens*24*256*16)
        if envs.VLLM_BATCH_INVARIANT:
            size = 2048*1024*1024
        _WORKSPACES[key] = torch.zeros(size,dtype=torch.uint8,device=device)
        logger.info("KATT2_RESERVE phase=model_load attention_workspace_bytes=%d total_persistent_bytes=%d",
                    size,pool[0].numel()*4+pool[2].numel()*4+size)
    return pool


def workspace_for_model(config):
    capacity = config.scheduler_config.max_num_seqs * ((config.model_config.max_model_len+63)//64)*4
    key = (torch.cuda.current_device(),capacity)
    if key not in _WORKSPACES:
        raise RuntimeError("KATT2 model-load reservation missing before KV profiling")
    return _WORKSPACES[key]

_spec = importlib.util.spec_from_file_location("_sm120_katt_launcher", Path(__file__).with_name("katt.py"))
_katt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_katt)


class StagedPrefillWrapper(BatchPrefillWithPagedKVCacheWrapper):
    def __init__(self, *args, staging_capacity_pages=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._stage_capacity = staging_capacity_pages
        self._stage_source_layout = self._kv_layout
        self._stage_k = self._stage_v = self._stage_ids = None
        self._stage_active = False

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices,
             paged_kv_last_page_len, num_qo_heads, num_kv_heads,
             head_dim_qk, page_size, *args, **kwargs):
        eligible = (
            not args and self._stage_source_layout in ("NHD","HND") and self._backend in ("auto", "fa2")
            and num_qo_heads == 24 and num_kv_heads == 4 and head_dim_qk == 256
            and kwargs.get("head_dim_vo", 256) in (None,256) and page_size in (16,32,64)
            and kwargs.get("q_data_type") == torch.bfloat16
            and kwargs.get("kv_data_type") == torch.float8_e4m3fn
            and kwargs.get("o_data_type", torch.bfloat16) in (None,torch.bfloat16)
            and kwargs.get("causal",False) and kwargs.get("pos_encoding_mode","NONE") == "NONE"
            and kwargs.get("custom_mask") is None and kwargs.get("packed_custom_mask") is None
            and kwargs.get("prefix_len_ptr") is None
            and kwargs.get("window_left",-1) == -1
            and kwargs.get("logits_soft_cap") in (None,0,0.0)
            and qo_indptr.device.type == "cpu" and paged_kv_indptr.device.type == "cpu"
            and paged_kv_indices.dtype == torch.int32
        )
        self._stage_active = eligible
        if not eligible:
            self._kv_layout = self._stage_source_layout
            logger.info_once("KATT2_DISABLED reason=plan_contract layout=%s q_dtype=%s kv_dtype=%s head_dim=%d",
                             self._stage_source_layout,kwargs.get("q_data_type"),kwargs.get("kv_data_type"),head_dim_qk)
            return super().plan(qo_indptr,paged_kv_indptr,paged_kv_indices,
                                paged_kv_last_page_len,num_qo_heads,num_kv_heads,
                                head_dim_qk,page_size,*args,**kwargs)
        num_pages = int(paged_kv_indptr[-1])
        # Capacity remains in 16-token units, matching the early reservation.
        # Reinterpret that same storage at the live kernel page size; no growth.
        capacity = self._stage_capacity or ((num_pages*page_size+63)//64)*4
        capacity_tokens = capacity*16
        if num_pages*page_size > capacity_tokens:
            raise RuntimeError(f"KATT2 configured capacity {capacity_tokens} tokens < required {num_pages*page_size}; no unprofiled growth")
        base_k,base_v,self._stage_ids = reserve_pool(self.device,capacity,"plan")
        self._stage_k = base_k.view(-1,4,page_size,256)
        self._stage_v = base_v.view(-1,4,page_size,256)
        self._stage_num_pages = num_pages
        self._stage_page_size = page_size
        self._stage_source_ids = paged_kv_indices
        self._stage_max_q = int((qo_indptr[1:]-qo_indptr[:-1]).max())
        self._stage_kv_tokens = int(((paged_kv_indptr[1:]-paged_kv_indptr[:-1]-1)*page_size+paged_kv_last_page_len).sum())
        self._stage_cache = (self._stage_k[:num_pages],self._stage_v[:num_pages])
        _katt.library()  # Resolve/initialize before any graph capture.
        self._kv_layout = "HND"
        kwargs["kv_data_type"] = torch.bfloat16
        return super().plan(qo_indptr,paged_kv_indptr,self._stage_ids[:num_pages],
                            paged_kv_last_page_len,num_qo_heads,num_kv_heads,
                            head_dim_qk,page_size,**kwargs)

    def run(self, q, paged_kv_cache, *args, **kwargs):
        if self._stage_active:
            k,v = paged_kv_cache
            if self._stage_source_layout == "NHD":
                k,v = k.transpose(1,2),v.transpose(1,2)
            _katt.gather(k,v,self._stage_k,self._stage_v,self._stage_source_ids,
                         self._stage_num_pages,self._stage_page_size)
            paged_kv_cache = self._stage_cache
        result = super().run(q,paged_kv_cache,*args,**kwargs)
        trace = os.environ.get("VLLM_SM120_KATT_TRACE_PREFILL") == "1"
        if self._stage_active and (trace or not getattr(self,"_stage_logged",False)):
            logger.info("KATT2_ACTIVE input_layout=%s output_layout=HND page_size=%d q_tokens=%d kv_tokens=%d staging_bytes=%d index_bytes=%d activation_floor_gib=%s",
                        self._stage_source_layout,self._stage_page_size,q.shape[0],self._stage_kv_tokens,
                        self._stage_k.numel()*4,self._stage_ids.numel()*4,
                        os.environ.get("VLLM_SM120_KATT_PROFILE_FLOOR_GIB","3.4"))
            self._stage_logged = True
        if (self._stage_active and q.shape[0]>0
                and (q.shape[0]>=8192 or self._stage_kv_tokens>=65536)
                and (trace or not getattr(self,"_stage_long_logged",False))):
            logger.info("KATT2_LONG_PREFILL_ACTIVE input_layout=%s page_size=%d q_tokens=%d kv_tokens=%d staging_bytes=%d",
                        self._stage_source_layout,self._stage_page_size,q.shape[0],self._stage_kv_tokens,self._stage_k.numel()*4)
            self._stage_long_logged = True
        return result


def make_wrapper(workspace, layout, backend, capacity_pages):
    if (workspace.device.index,capacity_pages) not in _WORKSPACES:
        raise RuntimeError("KATT2 persistent buffers must be reserved during model load")
    return StagedPrefillWrapper(workspace,layout,backend=backend,
                                staging_capacity_pages=capacity_pages)
