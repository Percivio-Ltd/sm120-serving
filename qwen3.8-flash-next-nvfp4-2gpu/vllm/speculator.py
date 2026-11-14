# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 Percivio Ltd (modifications)
#
# Modified by Percivio Ltd in October 2026: adds a pruned greedy draft head with FP8 or INT4 rows, sharded over
# tensor-parallel ranks, switched on by the VLLM_DRAFT_VOCAB_* environment variables.
# Based on vllm/v1/worker/gpu/spec_decode/mtp/speculator.py at vLLM commit
# c50aca14416ef29f6ed0ed256302f30646726dcd; speculator.diff holds the full change.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.logger import init_logger

from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)


logger = init_logger(__name__)


@triton.jit
def _ph_add_kernel(tok_ptr, n, start, end, mask_ptr, ids_ptr, dst_ptr, src_ptr, sdst_ptr, ssrc_ptr,
                   count_ptr, R, H: tl.constexpr, BLOCK_H: tl.constexpr, HAS_S: tl.constexpr,
                   SW: tl.constexpr = 1, BLOCK_S: tl.constexpr = 1):
    # One program walks the (few) decode tokens in order, so duplicates within the batch are
    # caught by the mask update of the earlier copy. Appends unseen local ids and their rows
    # (plus the SW scales of a quantized row: one for FP8, H / group for INT4).
    cnt = tl.load(count_ptr)
    for i in range(n):
        t = tl.load(tok_ptr + i).to(tl.int64)
        ok = (t >= start) & (t < end)
        tt = tl.where(ok, t, start)
        seen = tl.load(mask_ptr + tt)
        if ok & (seen == 0) & (cnt < R):
            tl.store(mask_ptr + tt, 1)
            tl.store(ids_ptr + cnt, t)
            for h0 in tl.static_range(0, H, BLOCK_H):
                offs = h0 + tl.arange(0, BLOCK_H)
                row = tl.load(src_ptr + (tt - start) * H + offs, mask=offs < H)
                tl.store(dst_ptr + cnt * H + offs, row, mask=offs < H)
            if HAS_S:
                so = tl.arange(0, BLOCK_S)
                tl.store(sdst_ptr + cnt * SW + so, tl.load(ssrc_ptr + (tt - start) * SW + so, mask=so < SW),
                         mask=so < SW)
            cnt += 1
        tl.debug_barrier()
    tl.store(count_ptr, cnt)


@triton.jit
def _ph_gemv_kernel(x_ptr, stride_x, w_ptr, s_ptr, val_ptr, idx_ptr, M, R, NB,
                    H: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    # Scores BLOCK_N FP8 rows (per-row scale) against every draft hidden state and writes the
    # block's best (value, slot) per row of x; the caller reduces over the NB blocks.
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rm[:, None] * stride_x + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < H), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * H + rk[None, :],
                    mask=(rn[:, None] < R) & (rk[None, :] < H), other=0.0)
        acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
    sc = tl.load(s_ptr + rn, mask=rn < R, other=0.0)
    acc = tl.where(rn[None, :] < R, acc * sc[None, :], float("-inf"))
    v = tl.max(acc, axis=1)
    i = tl.argmax(acc, axis=1)
    tl.store(val_ptr + rm * NB + pid, v, mask=rm < M)
    tl.store(idx_ptr + rm * NB + pid, pid * BLOCK_N + i, mask=rm < M)


@triton.jit
def _ph_gemv4_kernel(x_ptr, stride_x, w_ptr, s_ptr, val_ptr, idx_ptr, M, R, NB,
                     H: tl.constexpr, G: tl.constexpr, GPI: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    # INT4 rows: byte j of group g holds elements g*G + j (low nibble) and g*G + G/2 + j (high
    # nibble), each stored as q + 8 for q in [-8, 7], so both halves are plain tiles without a
    # shuffle; group g has the FP16 scale s[row, g]. The integer dots are exact in BF16 and
    # scaled once per group in FP32. GPI adjacent groups per iteration keep the row reads long.
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = tl.arange(0, BLOCK_M)
    rb = tl.arange(0, G // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for g0 in range(0, H // G, GPI):
        for gg in tl.static_range(GPI):
            g = g0 + gg
            q = tl.load(w_ptr + rn[:, None] * (H // 2) + g * (G // 2) + rb[None, :],
                        mask=rn[:, None] < R, other=0x88)
            # BF16 bits 0x4300 | n are exactly 128 + n for n < 16, so q - 8 = that - 136 without
            # an int-to-float conversion.
            lo = ((q & 15).to(tl.uint16) | 0x4300).to(tl.bfloat16, bitcast=True) - 136.0
            hi = ((q >> 4).to(tl.uint16) | 0x4300).to(tl.bfloat16, bitcast=True) - 136.0
            xp = x_ptr + rm[:, None] * stride_x + g * G + rb[None, :]
            x0 = tl.load(xp, mask=rm[:, None] < M, other=0.0)
            x1 = tl.load(xp + G // 2, mask=rm[:, None] < M, other=0.0)
            sc = tl.load(s_ptr + rn * (H // G) + g, mask=rn < R, other=0.0).to(tl.float32)
            acc += (tl.dot(x0, tl.trans(lo)) + tl.dot(x1, tl.trans(hi))) * sc[None, :]
    acc = tl.where(rn[None, :] < R, acc, float("-inf"))
    v = tl.max(acc, axis=1)
    i = tl.argmax(acc, axis=1)
    tl.store(val_ptr + rm * NB + pid, v, mask=rm < M)
    tl.store(idx_ptr + rm * NB + pid, pid * BLOCK_N + i, mask=rm < M)


@triton.jit
def _ph_gemv4v_kernel(x_ptr, w_ptr, s_ptr, val_ptr, idx_ptr, R, NB,
                      H: tl.constexpr, G: tl.constexpr, GPI: tl.constexpr, BLOCK_N: tl.constexpr):
    # As _ph_gemv4_kernel for one row of x on CUDA cores: each iteration reads a contiguous
    # [BLOCK_N, GPI groups] tile, products accumulate per element and are reduced once at the
    # end, so no tile goes through the dot-operand layout. FP32 bits 0x4B000000 | n are exactly
    # 2^23 + n, which gives q - 8 without an int-to-float conversion.
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    gi = tl.arange(0, GPI)
    rb = tl.arange(0, G // 2)
    acc = tl.zeros((BLOCK_N, GPI, G // 2), dtype=tl.float32)
    for g0 in range(0, H // G, GPI):
        g = g0 + gi
        q = tl.load(w_ptr + rn[:, None, None] * (H // 2) + g[None, :, None] * (G // 2) + rb[None, None, :],
                    mask=(rn < R)[:, None, None], other=0x88).to(tl.int32)
        lo = ((q & 15) | 0x4B000000).to(tl.float32, bitcast=True) - 8388616.0
        hi = ((q >> 4) | 0x4B000000).to(tl.float32, bitcast=True) - 8388616.0
        x0 = tl.load(x_ptr + g[:, None] * G + rb[None, :]).to(tl.float32)
        x1 = tl.load(x_ptr + g[:, None] * G + G // 2 + rb[None, :]).to(tl.float32)
        sc = tl.load(s_ptr + rn[:, None] * (H // G) + g[None, :], mask=(rn < R)[:, None],
                     other=0.0).to(tl.float32)
        acc += (lo * x0[None, :, :] + hi * x1[None, :, :]) * sc[:, :, None]
    s = tl.sum(tl.sum(acc, axis=2), axis=1)
    s = tl.where(rn < R, s, float("-inf"))
    tl.store(val_ptr + pid, tl.max(s, axis=0))
    tl.store(idx_ptr + pid, pid * BLOCK_N + tl.argmax(s, axis=0))


def _quant_rows_int4(w: torch.Tensor, group: int = 128, chunk: int = 4096,
                     clips=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75)):
    """Symmetric INT4 quantization of a [V, H] weight with one FP16 scale per group of `group`
    elements, the clip ratio chosen per group for least squared error. Returns packed [V, H / 2]
    uint8 rows (see _ph_gemv4_kernel) and [V, H / group] FP16 scales."""
    V, H = w.shape
    q = torch.empty(V, H // 2, dtype=torch.uint8, device=w.device)
    sc = torch.empty(V, H // group, dtype=torch.float16, device=w.device)
    for i in range(0, V, chunk):
        blk = w[i:i + chunk].float().view(-1, H // group, group)
        amax = blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        best = None
        for c in clips:
            s = (amax * (c / 7.0)).half().float()  # the error is measured with the stored scale
            qq = (blk / s).round().clamp(-8, 7)
            err = (qq * s - blk).square().sum(dim=-1, keepdim=True)
            if best is None:
                best = [err, qq, s]
            else:
                m = err < best[0]
                best = [torch.where(m, err, best[0]), torch.where(m, qq, best[1]), torch.where(m, s, best[2])]
        u = (best[1] + 8).to(torch.uint8)
        q[i:i + chunk] = (u[..., :group // 2] | (u[..., group // 2:] << 4)).view(-1, H // 2)
        sc[i:i + chunk] = best[2].view(-1, H // group).half()
    return q, sc


def _quant_rows_fp8(w: torch.Tensor, chunk: int = 8192):
    """Per-row symmetric e4m3 quantization of a [V, H] weight, chunked to bound temporaries."""
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    sc = torch.empty(w.shape[0], dtype=torch.float32, device=w.device)
    for i in range(0, w.shape[0], chunk):
        blk = w[i:i + chunk].float()
        s_blk = blk.abs().amax(dim=1).clamp(min=1e-12) / 448.0
        q[i:i + chunk] = (blk / s_blk[:, None]).to(torch.float8_e4m3fn)
        sc[i:i + chunk] = s_blk
    return q, sc


class MTPSpeculator(TargetDependentARSpeculator):
    share_mtp_topk_indices: bool = False
    # Pruned greedy draft head (VLLM_DRAFT_VOCAB_KEEP): the drafter only scores a fixed-size
    # [R, H] slice of lm_head holding a static keep list plus every token id seen in the
    # target's inputs (prompt chunks and verified tokens). Buffers have fixed addresses and are
    # updated in place, so the captured draft CUDA graphs stay valid. Verification is untouched,
    # so greedy output is unchanged; only drafts whose argmax lies outside the slice differ.
    _ph_w: torch.Tensor | None = None

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        self._init_pruned_head()

    def _init_pruned_head(self) -> None:
        path = os.environ.get("VLLM_DRAFT_VOCAB_KEEP")
        if not path:
            return
        if (self.speculative_config.draft_sample_method != "greedy"
                or self.use_local_argmax_reduction or self.use_acceptance_estimator
                or self.draft_watermarker is not None):
            logger.warning("Pruned draft head needs plain greedy drafting; disabled.")
            return
        lm_head = self.model.lm_head
        si = lm_head.shard_indices
        self._ph_start, self._ph_end = si.org_vocab_start_index, si.org_vocab_end_index
        w = getattr(lm_head, "weight", None)
        src = os.environ.get("VLLM_DRAFT_LM_HEAD_FILE")
        if src and w is not None:
            # Quantized (e.g. NVFP4) lm_head: score the draft against dense rows of the
            # original lm_head read from a safetensors file; only shard rows are kept.
            from safetensors import safe_open
            with safe_open(src, "pt", device=str(w.device)) as f:
                w = f.get_slice("lm_head.weight")[self._ph_start:self._ph_end].to(torch.bfloat16)
        if w is None or w.dtype not in (torch.bfloat16, torch.float16):
            logger.warning("Pruned draft head needs a dense lm_head weight; disabled.")
            return
        self._ph_tp = get_tensor_model_parallel_world_size()
        R = int(os.environ.get("VLLM_DRAFT_VOCAB_MAX", "65536"))
        V, dev = self.vocab_size, w.device
        static = torch.load(path).to(dev).long()
        static = static[(static >= self._ph_start) & (static < self._ph_end)]
        if static.numel() > R:
            static = static[:R]
        self._ph_dynamic = os.environ.get("VLLM_DRAFT_VOCAB_DYNAMIC", "1") == "1"
        self._ph_triton = os.environ.get("VLLM_DRAFT_VOCAB_TRITON", "1") == "1"
        self._ph_int4 = os.environ.get("VLLM_DRAFT_VOCAB_INT4", "0") == "1"
        self._ph_fp8 = os.environ.get("VLLM_DRAFT_VOCAB_FP8", "0") == "1" and not self._ph_int4
        self._ph_cfg = tuple(int(v) for v in os.environ.get(
            "VLLM_DRAFT_VOCAB_GEMV", "64,256,4,3").split(","))  # BLOCK_N, BLOCK_K, warps, stages
        self._ph_group = int(os.environ.get("VLLM_DRAFT_VOCAB_INT4_GROUP", "128"))
        # BLOCK_N, warps, stages, groups per iteration: one row of x (vector kernel), several (dot).
        self._ph_cfg4 = tuple(int(v) for v in os.environ.get("VLLM_DRAFT_VOCAB_GEMV4", "32,4,3,4").split(","))
        self._ph_cfg4d = tuple(int(v) for v in os.environ.get("VLLM_DRAFT_VOCAB_GEMV4D", "32,4,3,2").split(","))
        if self._ph_int4:
            # INT4 rows with group scales: a quarter of BF16's bytes; only the draft argmax sees them.
            H = w.shape[1]
            assert H % self._ph_group == 0 and self._ph_group & (self._ph_group - 1) == 0
            self._ph_W, self._ph_S = _quant_rows_int4(w, self._ph_group)
            del w
            self._ph_s = torch.ones(R + 1, H // self._ph_group, dtype=torch.float16, device=dev)
        elif self._ph_fp8:
            # FP8 rows halve the draft head's bytes; only the draft argmax sees the rounding.
            self._ph_W, self._ph_S = _quant_rows_fp8(w)
            del w
            self._ph_s = torch.ones(R + 1, dtype=torch.float32, device=dev)
        else:
            self._ph_W, self._ph_S, self._ph_s = w.contiguous(), None, None
        self._ph_R = R
        # Row R is a trash slot for overflowing or duplicate writes; it is never scored.
        self._ph_w = torch.empty(R + 1, self._ph_W.shape[1], dtype=self._ph_W.dtype, device=dev)
        self._ph_ids = torch.empty(R + 1, dtype=torch.long, device=dev)
        self._ph_mask = torch.zeros(V + 1, dtype=torch.bool, device=dev)
        self._ph_first = torch.empty(V + 1, dtype=torch.long, device=dev)
        self._ph_count = torch.zeros(1, dtype=torch.long, device=dev)
        self._ph_static = static
        self._ph_reset()
        logger.info("Pruned draft head: %d static of %d slots (shard %d-%d of %d, tp=%d, "
                    "dynamic=%s, triton=%s, fp8=%s, int4=%s group %d, gemv4 %s/%s, src=%s)", static.numel(), R,
                    self._ph_start, self._ph_end, V, self._ph_tp, self._ph_dynamic, self._ph_triton,
                    self._ph_fp8, self._ph_int4, self._ph_group, self._ph_cfg4, self._ph_cfg4d, src or "lm_head")

    @staticmethod
    def _rows(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t

    def _ph_reset(self) -> None:
        static, R = self._ph_static, self._ph_R
        pad = static[:1].expand(R + 1 - static.numel())
        ids = torch.cat([static, pad])
        self._ph_ids.copy_(ids)
        torch.index_select(self._rows(self._ph_W), 0, ids - self._ph_start, out=self._rows(self._ph_w))
        if self._ph_S is not None:
            torch.index_select(self._ph_S, 0, ids - self._ph_start, out=self._ph_s)
        self._ph_mask.zero_()
        self._ph_mask[static] = True
        self._ph_count.fill_(static.numel())

    def _ph_add(self, tokens: torch.Tensor) -> None:
        """Append unseen local-shard token ids without a host sync."""
        R, V = self._ph_R, self.vocab_size
        n = tokens.numel()
        if n <= 64 and self._ph_triton:
            H = self._ph_w.shape[1]
            has_s = self._ph_S is not None
            sw = self._ph_S.shape[1] if has_s and self._ph_S.dim() == 2 else 1
            _ph_add_kernel[(1,)](tokens, n, self._ph_start, self._ph_end,
                                 self._ph_mask.view(torch.uint8), self._ph_ids,
                                 self._rows(self._ph_w), self._rows(self._ph_W),
                                 self._ph_s if has_s else self._ph_count,
                                 self._ph_S if has_s else self._ph_count,
                                 self._ph_count, R, H=H,
                                 BLOCK_H=min(1024, triton.next_power_of_2(H)), HAS_S=has_s,
                                 SW=sw, BLOCK_S=triton.next_power_of_2(sw))
            return
        t = tokens.long()
        local = (t >= self._ph_start) & (t < self._ph_end)
        t = torch.where(local, t, V)  # V is a sentinel slot in mask/first
        ar = torch.arange(n, device=t.device)
        self._ph_first.fill_(n)
        self._ph_first.scatter_reduce_(0, t, ar, reduce="amin")
        new = local & (self._ph_first[t] == ar) & ~self._ph_mask[t]
        pos = self._ph_count + torch.cumsum(new, 0) - 1
        pos = torch.where(new & (pos < R), pos, R)
        self._ph_ids[pos] = t
        src = torch.where(new, t - self._ph_start, 0)
        self._rows(self._ph_w)[pos] = self._rows(self._ph_W)[src]
        if self._ph_S is not None:
            self._ph_s[pos] = self._ph_S[src]
        self._ph_mask[t] = True
        self._ph_count.add_(new.sum()).clamp_(max=R)

    def propose(self, input_batch, *args, **kwargs):
        if (self._ph_w is not None and self._ph_dynamic
                and not torch.cuda.is_current_stream_capturing()):
            n = input_batch.num_tokens
            if n > 256 and int(self._ph_count) > 0.95 * self._ph_R:
                self._ph_reset()  # host sync only on large (prefill) batches
            self._ph_add(input_batch.input_ids[:n])
        return super().propose(input_batch, *args, **kwargs)

    def sample_draft(self, hidden_states, sample_src_positions, idx_mapping, temperature,
                     seeds, draft_step, draft_logits, spec_step_idx=0, num_steps=1):
        if self._ph_w is None or draft_logits is not None:
            return super().sample_draft(hidden_states, sample_src_positions, idx_mapping,
                                        temperature, seeds, draft_step, draft_logits,
                                        spec_step_idx, num_steps)
        if self._ph_fp8 or self._ph_int4:
            argmax = self._ph_int4_argmax if self._ph_int4 else self._ph_fp8_argmax
            val, tok = argmax(hidden_states)
            if self._ph_tp == 1:
                return tok
        else:
            logits = F.linear(hidden_states, self._ph_w[:self._ph_R])
            if self._ph_tp == 1:
                return self._ph_ids[logits.argmax(dim=-1)]
            val, idx = logits.float().max(dim=-1)
            tok = self._ph_ids[idx]
        # [n, 2 * tp] after gathering (value, token) pairs from every vocab shard.
        both = tensor_model_parallel_all_gather(
            torch.stack([val, tok.to(torch.float32)], dim=-1).contiguous(), dim=-1)
        vals, toks = both[:, 0::2], both[:, 1::2]
        return toks.gather(1, vals.argmax(dim=-1, keepdim=True)).squeeze(1).long()

    def _ph_fp8_argmax(self, x: torch.Tensor):
        """(best scaled logit, token id) per row of x over the FP8 pruned rows."""
        R, H = self._ph_R, self._ph_w.shape[1]
        if x.stride(-1) != 1:
            x = x.contiguous()
        M = x.shape[0]
        if M > 16:
            logits = F.linear(x, self._ph_w[:R].to(x.dtype)).float() * self._ph_s[:R]
            val, idx = logits.max(dim=-1)
            return val, self._ph_ids[idx]
        bn, bk, warps, stages = self._ph_cfg
        nb = triton.cdiv(R, bn)
        val = torch.empty(M, nb, dtype=torch.float32, device=x.device)
        idx = torch.empty(M, nb, dtype=torch.int32, device=x.device)
        _ph_gemv_kernel[(nb,)](x, x.stride(0), self._ph_w, self._ph_s, val, idx, M, R, nb, H=H,
                               BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps,
                               num_stages=stages)
        j = val.argmax(dim=1, keepdim=True)
        return val.gather(1, j).squeeze(1), self._ph_ids[idx.gather(1, j).squeeze(1).long()]

    def _ph_int4_argmax(self, x: torch.Tensor):
        """(best scaled logit, token id) per row of x over the INT4 pruned rows: the vector kernel
        for one row (a single decode stream), the dot kernel for up to 16 rows per launch."""
        R, H = self._ph_R, self._ph_w.shape[1] * 2
        if x.stride(-1) != 1:
            x = x.contiguous()
        M = x.shape[0]
        bn, warps, stages, gpi = self._ph_cfg4 if M == 1 else self._ph_cfg4d
        nb = triton.cdiv(R, bn)
        val = torch.empty(M, nb, dtype=torch.float32, device=x.device)
        idx = torch.empty(M, nb, dtype=torch.int32, device=x.device)
        if M == 1:
            _ph_gemv4v_kernel[(nb,)](x, self._ph_w, self._ph_s, val, idx, R, nb, H=H, G=self._ph_group,
                                     GPI=gpi, BLOCK_N=bn, num_warps=warps, num_stages=stages)
        else:
            for m0 in range(0, M, 16):
                _ph_gemv4_kernel[(nb,)](x[m0:], x.stride(0), self._ph_w, self._ph_s, val[m0:], idx[m0:],
                                        min(16, M - m0), R, nb, H=H, G=self._ph_group, GPI=gpi,
                                        BLOCK_M=16, BLOCK_N=bn, num_warps=warps, num_stages=stages)
        j = val.argmax(dim=1, keepdim=True)
        return val.gather(1, j).squeeze(1), self._ph_ids[idx.gather(1, j).squeeze(1).long()]

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
