"""K-MOE Flash ph4 prefill GEMM patch, loaded as a vLLM general plugin (entry point in the mounted
sm120_flash_prefill-0.1.dist-info). Does nothing unless VLLM_SM120_FLASH_PREFILL is set.

VLLM_SM120_FLASH_PREFILL is a comma list ("1" means "moe,dense8"):
  moe     ModelOpt NVFP4 MoE layers served by Marlin also keep a FlashInfer-layout copy of their experts; calls
          with >= VLLM_SM120_FLASH_PREFILL_MOE_MIN_M tokens (default 2560) run FlashInfer's TRT-LLM CUTLASS
          fused MoE, W4A4 with the checkpoint's static input scales (class C). Smaller calls (decode, MTP
          verify, short chunks) stay on Marlin W4A16. Costs 708 MB per layer per GPU on Flash TP2 (31.6 GiB).
  dense8  Qwen4Exp W8A16 linears run prefill (M > 16) as per-token dynamic e4m3 activations x the existing
          e4m3 rows and fp32 row scales through cutlass_scaled_mm (class C), instead of dequant + BF16 GEMM.
  densea  Same linears use a BF16 copy of bf16(fp32(w8) * s) cached at load instead of the per-call dequant
          pass: bit-identical to the incumbent (class A), +2.4 GiB per GPU on Flash TP2. Ignored with dense8.
VLLM_SM120_FLASH_PREFILL_DENSE_SHAPES (default "8192x2560,7296x2560,2560x3072,2560x320") limits the dense
change to these (N, K) weights. Patches attach when their vLLM modules are imported, so the plugin itself
imports nothing heavy in the API-server process.
"""
import importlib.abc
import inspect
import os
import sys
import time

_RAW = os.environ.get("VLLM_SM120_FLASH_PREFILL", "").strip()
OPTS = {t.strip() for t in _RAW.split(",") if t.strip() and t.strip() != "0"}
if "1" in OPTS:
    OPTS = (OPTS - {"1"}) | {"moe", "dense8"}
MOE_MIN_M = int(os.environ.get("VLLM_SM120_FLASH_PREFILL_MOE_MIN_M", "2560"))
# Measurement only (host sync per call): directory for the topk_ids of up to 2000 MoE calls with M >= 1024.
DUMP = os.environ.get("VLLM_SM120_FLASH_PREFILL_DUMP")
DENSE_SHAPES = {tuple(int(v) for v in s.split("x")) for s in os.environ.get(
    "VLLM_SM120_FLASH_PREFILL_DENSE_SHAPES", "8192x2560,7296x2560,2560x3072,2560x320").split(",") if s}

_MOE_MOD = "vllm.model_executor.layers.quantization.modelopt"
_MARLIN_MOD = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
_LLG_MOD = "vllm.models.qwen4_exp.nvidia.low_latency_gemm"
_CACHING_MOD = "vllm.compilation.caching"

_STATS = {"fi_layers": 0, "dense_layers": 0}


def _log(msg: str) -> None:
    print(f"SM120_FLASH_PREFILL {msg}", flush=True)


# ------------------------------------------------------------------------------------------- MoE

def build_fi_state(w13, s13, g13, a13, w2, s2, g2, a2):
    """FlashInfer CUTLASS NVFP4 MoE operands from ModelOpt checkpoint tensors (as vLLM's FLASHINFER_CUTLASS
    backend prepares them): rows reordered [w1, w3] -> [w3, w1], block scales swizzled and viewed as int32,
    one input scale per layer (amax over experts), alphas = weight_scale_2 * input_scale."""
    import torch
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    E, two_i = w13.shape[:2]
    i = two_i // 2
    w31 = torch.cat([w13[:, i:], w13[:, :i]], dim=1).contiguous()
    s31 = swizzle_blockscale(torch.cat([s13[:, i:], s13[:, :i]], dim=1).contiguous())
    s2z = swizzle_blockscale(s2.contiguous())
    a1 = a13.max().float()
    a2m = a2.max().float()
    g1 = (g13.reshape(E, -1)[:, 0].float() * a1).contiguous()
    qs = [(1.0 / a1).repeat(E).contiguous(), s31.view(torch.int32), g1,
          (1.0 / a2m).repeat(E).contiguous(), s2z.view(torch.int32), (g2.reshape(E).float() * a2m).contiguous()]
    return w31.view(torch.long), w2.contiguous().view(torch.long), qs


def run_fi(state, x, topk_ids, topk_weights, out):
    import torch
    from flashinfer.fused_moe import cutlass_fused_moe
    w1, w2, qs = state
    cutlass_fused_moe(input=x, token_selected_experts=topk_ids.to(torch.int32),
                      token_final_scales=topk_weights.to(torch.float32), fc1_expert_weights=w1,
                      fc2_expert_weights=w2, output_dtype=x.dtype, quant_scales=qs, output=out)
    return out


def _patch_modelopt(mod) -> None:
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    cls = mod.ModelOptNvFp4FusedMoE
    orig = cls.process_weights_after_loading

    def process_weights_after_loading(self, layer) -> None:
        state = None
        if (self.nvfp4_backend == NvFp4MoeBackend.MARLIN and self.moe.is_act_and_mul
                and not getattr(self, "per_token_activation", False)
                and getattr(layer, "activation", MoEActivation.SILU) == MoEActivation.SILU
                and getattr(layer, "expert_map", None) is None
                and getattr(layer, "swiglu_limit", None) is None):
            with torch.no_grad():
                state = build_fi_state(layer.w13_weight.data, layer.w13_weight_scale.data,
                                       layer.w13_weight_scale_2.data, layer.w13_input_scale.data,
                                       layer.w2_weight.data, layer.w2_weight_scale.data,
                                       layer.w2_weight_scale_2.data, layer.w2_input_scale.data)
        orig(self, layer)
        experts = getattr(getattr(self, "moe_kernel", None), "fused_experts", None)
        if state is not None and type(experts).__name__ == "MarlinExperts":
            experts._sm120_fi_state = state  # one experts object per layer (make_nvfp4_moe_kernel)
            _STATS["fi_layers"] += 1
            if _STATS["fi_layers"] in (1, 48):
                _log(f"moe: FlashInfer W4A4 copy for {_STATS['fi_layers']} layers (M >= {MOE_MIN_M})")
    cls.process_weights_after_loading = process_weights_after_loading


def _patch_marlin(mod) -> None:
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    cls = mod.MarlinExperts
    orig = cls.apply
    sig = inspect.signature(orig)

    def apply(self, *args, **kwargs):
        state = getattr(self, "_sm120_fi_state", None)
        if state is None:
            return orig(self, *args, **kwargs)
        a = sig.bind(self, *args, **kwargs).arguments
        x = a["hidden_states"]
        if (DUMP and x.shape[0] >= 1024 and _STATS.get("dumps", 0) < 2000
                and not torch.cuda.is_current_stream_capturing()):
            _STATS["dumps"] = _STATS.get("dumps", 0) + 1
            torch.save({"t": time.time(), "M": x.shape[0], "ids": a["topk_ids"].to(torch.int16).cpu()},
                       f"{DUMP}/topk-{os.getpid()}-{_STATS['dumps']:05d}.pt")
        if (x.dim() != 2 or x.shape[0] < MOE_MIN_M or x.dtype != torch.bfloat16 or not x.is_contiguous()
                or not a["output"].is_contiguous() or a["expert_map"] is not None
                or a["apply_router_weight_on_input"] or getattr(self, "_lora_context", None) is not None
                or a["activation"] != MoEActivation.SILU):
            return orig(self, *args, **kwargs)
        if not _STATS.get("fi_calls"):
            _STATS["fi_calls"] = 1
            _log(f"moe: first FlashInfer call, M={x.shape[0]}")
        run_fi(state, x, a["topk_ids"], a["topk_weights"], a["output"])
    cls.apply = apply


# ----------------------------------------------------------------------------------------- dense

def _patch_llg(mod) -> None:
    import torch
    from vllm import _custom_ops as ops
    from vllm.utils.torch_utils import direct_register_custom_op
    fp8 = "dense8" in OPTS
    max_m = mod._W8_MAX_M
    small = mod._qwen4_exp_w8a16_gemm  # original implementation (decode Triton kernel; dequant path above)

    def _gemm(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor, ws: torch.Tensor, cnt: torch.Tensor,
              wb: torch.Tensor, silu_n: int = 0, inv_hc: float = 1.0) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        M = x2.shape[0]
        if M <= max_m:
            return small(x, w8, scale, ws, cnt, silu_n, inv_hc)
        if not _STATS.get("dense_calls"):
            _STATS["dense_calls"] = 1
            _log(f"{'dense8' if fp8 else 'densea'}: first prefill call, M={M} N={w8.shape[0]} K={shape[-1]}")
        N = w8.shape[0]
        if fp8:
            x8, sx = ops.scaled_fp8_quant(x2.contiguous(), use_per_token_if_dynamic=True)
            y = ops.cutlass_scaled_mm(x8, w8.t(), sx, scale.view(-1, 1), x.dtype)
        else:
            y = torch.nn.functional.linear(x2, wb)
        if silu_n:
            v = y[:, :silu_n].float() * inv_hc
            y[:, :silu_n] = (v * torch.sigmoid(v)).to(y.dtype)
        return y.reshape(*shape[:-1], N)

    def _gemm_fake(x, w8, scale, ws, cnt, wb, silu_n=0, inv_hc=1.0):
        return x.new_empty((*x.shape[:-1], w8.shape[0]))

    direct_register_custom_op(op_name="sm120_flash_w8a16_gemm", op_func=_gemm, mutates_args=["ws", "cnt"],
                              fake_impl=_gemm_fake)
    orig_linear = mod.w8a16_linear
    cls = mod.Qwen4ExpW8A16LinearMethod
    orig_pwal = cls.process_weights_after_loading

    def process_weights_after_loading(self, layer) -> None:
        orig_pwal(self, layer)
        if tuple(layer.w8_weight.shape) not in DENSE_SHAPES:
            return
        if fp8:
            layer._sm120_w8_bf16 = layer.w8_weight.new_empty(0, dtype=torch.bfloat16)
        else:  # bf16(fp32(w8) * s), exactly the incumbent's per-call dequant, in row chunks
            w8, s = layer.w8_weight, layer.w8_scale
            wb = torch.empty(w8.shape, dtype=torch.bfloat16, device=w8.device)
            for r in range(0, w8.shape[0], 4096):
                wb[r:r + 4096] = (w8[r:r + 4096].to(torch.float32) * s[r:r + 4096, None]).to(torch.bfloat16)
            layer._sm120_w8_bf16 = wb
        _STATS["dense_layers"] += 1
        if _STATS["dense_layers"] in (1, 50, 100):
            _log(f"{'dense8' if fp8 else 'densea'}: {_STATS['dense_layers']} W8A16 linears "
                 f"(last {tuple(layer.w8_weight.shape)})")

    def w8a16_linear(layer, x, silu_n: int = 0, inv_hc: float = 1.0):
        wb = getattr(layer, "_sm120_w8_bf16", None)
        if wb is None:
            return orig_linear(layer, x, silu_n, inv_hc)
        return torch.ops.vllm.sm120_flash_w8a16_gemm(x, layer.w8_weight, layer.w8_scale, layer.w8_ws,
                                                     layer.w8_cnt, wb, silu_n, inv_hc)

    cls.process_weights_after_loading = process_weights_after_loading
    mod.w8a16_linear = w8a16_linear


def _patch_caching(mod) -> None:
    """vLLM's AOT compile cache key hashes only known VLLM_* variables and checks only the sources inlined in the
    cached graph, so a cache written without this plugin would load and skip the patched w8a16_linear."""
    orig = mod.aot_compile_hash_factors
    shapes = ",".join(f"{n}x{k}" for n, k in sorted(DENSE_SHAPES))
    tag = f"sm120_flash_prefill:{','.join(sorted(OPTS))}:{shapes}"

    def aot_compile_hash_factors(vllm_config):
        return [*orig(vllm_config), tag]
    mod.aot_compile_hash_factors = aot_compile_hash_factors


# ------------------------------------------------------------------------------------ activation

class _PostImport(importlib.abc.MetaPathFinder):
    """Runs a patch function right after the named module executes."""

    def __init__(self, targets):
        self.targets = targets

    def find_spec(self, name, path, target=None):
        fn = self.targets.get(name)
        if fn is None:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        self.targets.pop(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            fn(module)
        spec.loader.exec_module = exec_and_patch
        return spec


def register() -> None:
    if not OPTS or getattr(register, "done", False):
        return
    register.done = True
    targets = {}
    if "moe" in OPTS:
        targets[_MOE_MOD] = _patch_modelopt
        targets[_MARLIN_MOD] = _patch_marlin
    if OPTS & {"dense8", "densea"}:
        targets[_LLG_MOD] = _patch_llg
        targets[_CACHING_MOD] = _patch_caching
    pending = {}
    for name, fn in targets.items():
        if name in sys.modules:
            fn(sys.modules[name])
        else:
            pending[name] = fn
    if pending:
        sys.meta_path.insert(0, _PostImport(pending))
    _log(f"enabled {sorted(OPTS)} moe_min_m={MOE_MIN_M} dense_shapes={sorted(DENSE_SHAPES)} pid={os.getpid()}")
