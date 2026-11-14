# SPDX-License-Identifier: Apache-2.0
"""Fused Qwen4Exp HyperConnection decode path: two kernels per HC boundary instead of four.

The stock ``GatedResidual.combine_and_mix`` runs combine+RMSNorm, the down+inject GEMM with SiLU,
the up GEMM and the sigmoid gated mix as four launches. For decode token counts (M <= 16):

* K1 (``_hcx_down_kernel``, split-K over the HC streams, one weight tile per program): combine
  ``res + 2 sigmoid(inj / hc) * block`` (rounded to BF16 and written back as the new multi-stream
  state), per-stream sums of squares, and the down projection of the *un-normalised* streams
  against weights that have the norm affine ``(1 + w)`` folded in. Per-stream RMS scales are
  applied after the reduction, so the normalised activations never round-trip through memory. The
  last program of each N tile (ticket counter) applies the per-stream scales, the FP8 row scales
  and SiLU(x / hc), and writes the low-rank activations, the injection logits and the per-stream
  rrms. Workspace and counters are left zeroed, so the launch replays safely in CUDA graphs.
  v4: the last program reads all workspace tiles before a CTA barrier and only then clears them
  (v3 raced on tiles whose clearing stores had a different layout from the loads), and the
  injection rows carry a second per-row FP8 term for their quantization residual.
* K2 (``_hcx_up_kernel``): one tile holding the up-projection rows of all streams for a D slice,
  so a single dot gives every stream's gate; then the sigmoid gate and the mean over streams of
  ``sigmoid(gate) * norm(x)``, writing the block input directly.

Both kernels launch with programmatic dependent launch where supported: weight tiles (which do
not depend on the previous kernel) are loaded before the grid-dependency wait, and each kernel
releases its dependents right after the wait, so K2's weight fetch overlaps K1.

Weights are per-output-row FP8 (e4m3) by default; ``QWEN4_EXP_HC_FUSED_BF16=1`` keeps BF16 (with
the folded norm affine). Enable with ``QWEN4_EXP_HC_FUSED=1``; the patched low_latency_gemm module
calls ``enable(module)``, and the HC linears it takes over are no longer plain unquantized linears,
so its own GEMM selection skips them. Larger token counts use the stock path and the original BF16
weights.
"""

import os
import weakref

import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

MAX_M = 16
_BF16 = os.environ.get("QWEN4_EXP_HC_FUSED_BF16", "0") == "1"
_PDL_ENV = os.environ.get("QWEN4_EXP_HC_PDL", "1") == "1"


def _cfg(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(t) for t in os.environ.get(name, default).split("/"))


# K1: BLOCK_N / K_PER / warps / stages; K2: BLOCK_D / warps / stages.
_K1 = _cfg("QWEN4_EXP_HC_K1", "32/256/4/1")
_K2 = _cfg("QWEN4_EXP_HC_K2", "8/4/1")
_MIN_BN = 16
_MAX_BN = 64
# Smallest K1 K_PER; bounds the per-CTA sum-of-squares slots (K / K_PER per N tile).
_MIN_KPER = 128


def _pdl() -> bool:
    return _PDL_ENV and current_platform.is_arch_support_pdl()


@triton.jit
def _hcx_down_kernel(res_ptr, blk_ptr, inj_ptr, w_ptr, s_ptr, newh_ptr, low_ptr, injo_ptr,
                     rrms_ptr, ws_ptr, sq_ptr, cnt_ptr, M, stride_res, stride_blk, stride_inj,
                     stride_newh, inv_hc, eps,
                     HAS_BLOCK: tl.constexpr, HAS_INJ: tl.constexpr, N_LOW: tl.constexpr,
                     N_INJ: tl.constexpr, N: tl.constexpr, NPAD: tl.constexpr, HC: tl.constexpr,
                     HC_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     K_PER: tl.constexpr, KS_PAD: tl.constexpr, launch_pdl: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    KS = tl.num_programs(1)
    K: tl.constexpr = HC * HC_DIM
    KS_C: tl.constexpr = K // K_PER
    KPS: tl.constexpr = HC_DIM // K_PER
    k0 = pid_k * K_PER
    s = k0 // HC_DIM
    rm = tl.arange(0, BLOCK_M)
    mmask = rm < M
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = rn < N
    rk = k0 + tl.arange(0, K_PER)
    # The weight tile does not depend on the previous kernel: fetch it before the dependency wait.
    w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=nmask[:, None], other=0.0)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(res_ptr + rm[:, None] * stride_res + rk[None, :], mask=mmask[:, None],
                other=0.0).to(tl.float32)
    if HAS_BLOCK:
        b = tl.load(blk_ptr + rm[:, None] * stride_blk + (rk - s * HC_DIM)[None, :],
                    mask=mmask[:, None], other=0.0).to(tl.float32)
        if HAS_INJ:
            iv = tl.load(inj_ptr + rm * stride_inj + s, mask=mmask, other=0.0).to(tl.float32)
            b = b * (2.0 * tl.sigmoid(iv * inv_hc))[:, None]
        x = x + b
    xb = x.to(tl.bfloat16)
    if HAS_BLOCK:
        if pid_n == 0:
            tl.store(newh_ptr + rm[:, None] * stride_newh + rk[None, :], xb, mask=mmask[:, None])
    xf = xb.to(tl.float32)
    sq = tl.sum(xf * xf, axis=1)
    acc = tl.dot(xb, tl.trans(w.to(tl.bfloat16)))
    # Partial products per stream. The partial sums of squares go to this CTA's own slot with plain
    # stores: v2 accumulated them with a 1-D atomic_add, and K1 tiles with BLOCK_N 64 and 4 warps
    # gave wrong, run-to-run varying results for M >= 4 (hc_sweep.log), consistent with duplicated
    # atomics on a replicated layout. A replicated store writes the same value and is harmless.
    tl.atomic_add(ws_ptr + (s * BLOCK_M + rm[:, None]) * NPAD + rn[None, :], acc,
                  mask=mmask[:, None] & nmask[None, :], sem="relaxed")
    tl.store(sq_ptr + (pid_n * KS_C + pid_k) * BLOCK_M + rm, sq, mask=mmask)
    tl.debug_barrier()
    ticket = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
    if ticket == KS - 1:
        rs = tl.arange(0, HC)
        rj = tl.arange(0, KS_PAD)
        part = tl.load(sq_ptr + (pid_n * KS_C + rj[:, None]) * BLOCK_M + rm[None, :],
                       mask=(rj[:, None] < KS_C) & mmask[None, :], other=0.0, cache_modifier=".cg")
        sqv = tl.zeros((BLOCK_M, HC), dtype=tl.float32)
        for st in tl.static_range(HC):
            col = tl.sum(tl.where((rj // KPS == st)[:, None], part, 0.0), axis=0)
            sqv += tl.where(rs[None, :] == st, col[:, None], 0.0)
        rr = tl.rsqrt(sqv / HC_DIM + eps)
        if pid_n == 0:
            tl.store(rrms_ptr + rm[:, None] * HC + rs[None, :], rr, mask=mmask[:, None])
        z = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        zr = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        pmask = mmask[:, None] & nmask[None, :]
        icol = (rn >= N_LOW) & (rn < N_LOW + N_INJ)
        for st in tl.static_range(HC):
            p_ptrs = ws_ptr + (st * BLOCK_M + rm[:, None]) * NPAD + rn[None, :]
            pv = tl.load(p_ptrs, mask=pmask, other=0.0, cache_modifier=".cg")
            r_st = tl.sum(tl.where(rs[None, :] == st, rr, 0.0), axis=1)
            z += pv * r_st[:, None]
            if N_INJ > 0:
                # Residual-term columns of the injection rows, shifted onto the injection columns.
                pr = tl.load(p_ptrs + N_INJ, mask=mmask[:, None] & icol[None, :], other=0.0,
                             cache_modifier=".cg")
                zr += pr * r_st[:, None]
        # Every thread finishes its workspace reads before any thread clears the workspace. Triton may
        # give the clearing stores a different register layout from the loads (K1 64/256/4/1 and
        # 64/128/4/1 with 4 warps did, hc_ptx.py), so v3 let one thread zero a partial sum that another
        # thread had not read yet: the run-to-run varying parity failures of hc_sweep/hc_sweep3.
        tl.debug_barrier()
        for st in tl.static_range(HC):
            p_ptrs = ws_ptr + (st * BLOCK_M + rm[:, None]) * NPAD + rn[None, :]
            tl.store(p_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=pmask)
        z = z * tl.load(s_ptr + rn, mask=nmask, other=0.0)[None, :]
        if N_INJ > 0:
            z += zr * tl.load(s_ptr + rn + N_INJ, mask=icol, other=0.0)[None, :]
        v = z * inv_hc
        lo = v * tl.sigmoid(v)
        tl.store(low_ptr + rm[:, None] * N_LOW + rn[None, :], lo.to(tl.bfloat16),
                 mask=mmask[:, None] & (rn[None, :] < N_LOW))
        if N_INJ > 0:
            tl.store(injo_ptr + rm[:, None] * N_INJ + (rn - N_LOW)[None, :], z.to(tl.bfloat16),
                     mask=mmask[:, None] & (rn[None, :] >= N_LOW) & (rn[None, :] < N_LOW + N_INJ))
        tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed")


@triton.jit
def _hcx_up_kernel(low_ptr, w_ptr, s_ptr, h_ptr, nw_ptr, rrms_ptr, y_ptr, M, stride_h, stride_y,
                   N_LOW: tl.constexpr, HC: tl.constexpr, HC_DIM: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_L: tl.constexpr,
                   launch_pdl: tl.constexpr):
    pid = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    mmask = rm < M
    # Column j of the tile is stream j // BLOCK_D, hidden index pid * BLOCK_D + j % BLOCK_D.
    rj = tl.arange(0, HC * BLOCK_D)
    st = rj // BLOCK_D
    rows = st * HC_DIM + pid * BLOCK_D + rj % BLOCK_D
    rl = tl.arange(0, BLOCK_L)
    lmask = rl < N_LOW
    w = tl.load(w_ptr + rows[:, None] * N_LOW + rl[None, :], mask=lmask[None, :], other=0.0)
    sc = tl.load(s_ptr + rows)
    nw = tl.load(nw_ptr + rows).to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    lo = tl.load(low_ptr + rm[:, None] * N_LOW + rl[None, :], mask=mmask[:, None] & lmask[None, :],
                 other=0.0)
    g = tl.dot(lo, tl.trans(w.to(tl.bfloat16)))
    g = (g * sc[None, :]).to(tl.bfloat16).to(tl.float32)
    h = tl.load(h_ptr + rm[:, None] * stride_h + rows[None, :], mask=mmask[:, None],
                other=0.0).to(tl.float32)
    rr = tl.load(rrms_ptr + rm[:, None] * HC + st[None, :], mask=mmask[:, None], other=0.0)
    y = h * rr
    y = (y + y * nw[None, :]).to(tl.bfloat16).to(tl.float32)
    p = tl.sigmoid(g) * y
    out = tl.sum(tl.reshape(p, (BLOCK_M, HC, BLOCK_D)), axis=1) / HC
    rd = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    tl.store(y_ptr + rm[:, None] * stride_y + rd[None, :], out.to(tl.bfloat16), mask=mmask[:, None])


def _hc_fused(hidden: torch.Tensor, block: torch.Tensor | None, inj: torch.Tensor | None,
              wd: torch.Tensor, sd: torch.Tensor, wu: torch.Tensor, su: torch.Tensor,
              nw: torch.Tensor, ws: torch.Tensor, sq: torch.Tensor, cnt: torch.Tensor,
              hc_count: int, n_low: int, n_inj: int,
              eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (new multi-stream state, block input, injection logits); the state is an empty
    placeholder without a pending block and the logits one without injection rows."""
    M = hidden.shape[0]
    hs = hidden.shape[1] // hc_count
    # Low-rank rows, injection rows, then one residual row per injection row (prepare()).
    n = n_low + 2 * n_inj
    assert wd.shape[0] == n, (wd.shape, n_low, n_inj)
    dev = hidden.device
    has_block = block is not None
    newh = torch.empty_like(hidden) if has_block else hidden.new_empty((0,))
    low = torch.empty((M, n_low), dtype=torch.bfloat16, device=dev)
    injo = torch.empty((M, n_inj), dtype=torch.bfloat16, device=dev) if n_inj else \
        hidden.new_empty((0,))
    rrms = torch.empty((M, hc_count), dtype=torch.float32, device=dev)
    out = torch.empty((M, hs), dtype=torch.bfloat16, device=dev)
    pdl = _pdl()
    bn, kper, w1, s1 = _K1
    assert _MIN_BN <= bn <= _MAX_BN and kper >= _MIN_KPER and hs % kper == 0, _K1
    # The injection columns and their residual columns must share one N tile (one ticket).
    assert n_low % bn + 2 * n_inj <= bn, (_K1, n_low, n_inj)
    nt = triton.cdiv(n, bn)
    _hcx_down_kernel[(nt, hc_count * hs // kper)](
        hidden, block if has_block else hidden, inj if inj is not None else hidden, wd, sd,
        newh if has_block else hidden, low, injo if n_inj else low, rrms, ws, sq, cnt, M,
        hidden.stride(0),
        block.stride(0) if has_block else 0, inj.stride(0) if inj is not None else 0,
        newh.stride(0) if has_block else 0, 1.0 / hc_count, eps, HAS_BLOCK=has_block,
        HAS_INJ=inj is not None, N_LOW=n_low, N_INJ=n_inj, N=n, NPAD=nt * bn, HC=hc_count,
        HC_DIM=hs, BLOCK_M=MAX_M, BLOCK_N=bn, K_PER=kper,
        KS_PAD=triton.next_power_of_2(hc_count * hs // kper), launch_pdl=pdl, num_warps=w1,
        num_stages=s1)
    bd, w2, s2 = _K2
    src = newh if has_block else hidden
    _hcx_up_kernel[(triton.cdiv(hs, bd),)](
        low, wu, su, src, nw, rrms, out, M, src.stride(0), out.stride(0), N_LOW=n_low,
        HC=hc_count, HC_DIM=hs, BLOCK_M=MAX_M, BLOCK_D=bd, BLOCK_L=triton.next_power_of_2(n_low),
        launch_pdl=pdl, num_warps=w2, num_stages=s2)
    return newh, out, injo


def _hc_fused_fake(hidden, block, inj, wd, sd, wu, su, nw, ws, sq, cnt, hc_count, n_low, n_inj,
                   eps):
    M = hidden.shape[0]
    newh = torch.empty_like(hidden) if block is not None else hidden.new_empty((0,))
    injo = hidden.new_empty((M, n_inj)) if n_inj else hidden.new_empty((0,))
    return newh, hidden.new_empty((M, hidden.shape[1] // hc_count)), injo


direct_register_custom_op(
    op_name="qwen4_exp_hc_fused",
    op_func=_hc_fused,
    mutates_args=["ws", "sq", "cnt"],
    fake_impl=_hc_fused_fake,
)


def _rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row e4m3 weights and fp32 scales, or BF16 weights with unit scales."""
    if _BF16:
        return w.to(torch.bfloat16).contiguous(), torch.ones(w.shape[0], dtype=torch.float32,
                                                             device=w.device)
    s = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (w / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.contiguous(), s.float().contiguous()


def prepare(hc: nn.Module) -> None:
    """Build fused-path weights and workspace for one GatedResidual after its weights are loaded."""
    if getattr(hc, "_fx_ready", False):
        return
    down = (hc.input_mix_weight_down_block_inject if hc.use_combine
            else hc.input_mix_weight_down)
    up = hc.input_mix_weight_up
    nw = hc.hc_norm.weight.data
    if nw.numel() != hc.hyper_hidden_size:
        nw = nw.repeat(hc.hc_count)
    # Only the low-rank and injection rows; the merged linear's alignment padding rows are never
    # loaded from the checkpoint and stay uninitialised.
    n_inj = hc.hc_count if hc.use_combine else 0
    n = hc.hc_lowrank + n_inj
    wd = down.weight.data
    with torch.no_grad():
        wf = wd[:n].float() * (1.0 + nw.float())[None, :]
        q, s = _rows(wf)
        if n_inj:
            # The injection logits are only hc_count values per token, so a single per-row FP8 term
            # left their error at the deployed W8A16 level with a large draw-to-draw spread (hc_test3:
            # mix/M1 failed at 2.4e-2 against W8's 9.8e-3). A second per-row term quantizes the residual
            # of the injection rows; the kernel adds it back in the injection columns. 4 extra rows.
            inj = slice(hc.hc_lowrank, n)
            r = wf[inj] - q[inj].float() * s[inj, None]
            q2, s2 = _rows(r)
            q, s = torch.cat([q, q2]), torch.cat([s, s2])
            n += n_inj
        hc._fx_wd, hc._fx_sd = q.contiguous(), s.contiguous()
        hc._fx_wu, hc._fx_su = _rows(up.weight.data.float())
    hc._fx_nw = nw.contiguous()
    hc._fx_n = n
    # Sized for any BLOCK_N in [_MIN_BN, _MAX_BN], so the tile configuration can change at runtime.
    dev = wd.device
    hc._fx_ws = torch.zeros(hc.hc_count * MAX_M * (n + _MAX_BN), dtype=torch.float32, device=dev)
    # Per-CTA partial sums of squares: overwritten on every call, so (unlike ws/cnt) never zeroed.
    hc._fx_sq = torch.zeros(triton.cdiv(n, _MIN_BN) * (hc.hyper_hidden_size // _MIN_KPER) * MAX_M,
                            dtype=torch.float32, device=dev)
    hc._fx_cnt = torch.zeros(triton.cdiv(n, _MIN_BN), dtype=torch.int32, device=dev)
    hc._fx_ready = True


def fused(hc: nn.Module, hidden: torch.Tensor, block: torch.Tensor | None,
          inj: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n_inj = hc.hc_count if hc.use_combine else 0
    newh, out, injo = torch.ops.vllm.qwen4_exp_hc_fused(
        hidden, block, inj, hc._fx_wd, hc._fx_sd, hc._fx_wu, hc._fx_su, hc._fx_nw, hc._fx_ws,
        hc._fx_sq, hc._fx_cnt, hc.hc_count, hc.hc_lowrank, n_inj, hc.config.rms_norm_eps)
    return (newh if block is not None else hidden), out, (injo if n_inj else None)


def _ok(hc: nn.Module, hidden: torch.Tensor) -> bool:
    return (getattr(hc, "_fx_ready", False) and 1 <= hidden.shape[0] <= MAX_M
            and hidden.dtype == torch.bfloat16 and hidden.stride(1) == 1)


class _HcPrepMethod(UnquantizedLinearMethod):
    """Unquantized linear whose post-load hook also prepares the owning GatedResidual."""

    def __init__(self, hc: nn.Module):
        super().__init__()
        self._hc = weakref.ref(hc)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        super().process_weights_after_loading(layer)
        hc = self._hc()
        if hc is not None:
            prepare(hc)


def enable(module: nn.Module) -> set[nn.Module]:
    """Route every GatedResidual under ``module`` to the fused path; return the linears it owns."""
    from vllm.models.qwen4_exp.nvidia import hyperconnection as hcm

    cls = hcm.GatedResidual
    if not getattr(cls, "_fx_patched", False):
        orig_mix, orig_cam = cls.mix, cls.combine_and_mix

        def mix(self, hidden_states):
            if _ok(self, hidden_states):
                return fused(self, hidden_states, None, None)
            return orig_mix(self, hidden_states)

        def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
            if (_ok(self, hidden_states) and prev_block_output.stride(-1) == 1
                    and (prev_injection is None or prev_injection.stride(-1) == 1)):
                return fused(self, hidden_states, prev_block_output, prev_injection)
            return orig_cam(self, hidden_states, prev_block_output, prev_injection)

        cls.mix, cls.combine_and_mix = mix, combine_and_mix
        cls._fx_patched = True
    owned = set()
    n = 0
    for hc in module.modules():
        if not isinstance(hc, cls):
            continue
        down = hc.input_mix_weight_down_block_inject if hc.use_combine else hc.input_mix_weight_down
        # Prepare once, from the up projection's post-load hook (all checkpoint weights are loaded
        # before any post-load hook runs).
        if type(hc.input_mix_weight_up.quant_method) is UnquantizedLinearMethod:
            hc.input_mix_weight_up.quant_method = _HcPrepMethod(hc)
            owned.update((down, hc.input_mix_weight_up))
        n += 1
    print(f"QWEN4_EXP_HC_FUSED {n} GatedResidual modules (K1 {_K1}, K2 {_K2}, "
          f"{'bf16' if _BF16 else 'fp8'} weights, pdl={_PDL_ENV})", flush=True)
    return owned
