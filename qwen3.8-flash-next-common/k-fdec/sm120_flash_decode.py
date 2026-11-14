"""K-FDEC Flash ph4 decode MoE patch, loaded as a vLLM general plugin (entry point in the mounted
sm120_flash_decode-0.1.dist-info). Does nothing but log one "disabled" line unless VLLM_SM120_FLASH_DECODE is set.

VLLM_SM120_FLASH_DECODE=1 (or "moe"): ModelOpt NVFP4 MoE layers served by Marlin W4A16 run calls with
1 <= M <= VLLM_SM120_FLASH_DECODE_MAX_M tokens (default 4: one MTP3 verify row group) through the K-FDEC sm_120
kernels instead of fused_marlin_moe (align, sort, w13, act, w2, sum): kernel A (w13 GEMM + SiLU*up per routed slot)
and kernel B (w2 GEMM + top-k weighted sum, cluster reduction), chained with programmatic dependent launch. Both
read the Marlin tensors already resident (no weight copy); numerics class B (fp32 accumulation in a different
order). Larger M and anything ineligible stay on the incumbent (including the kmoe1 prefill wrapper, if mounted).
VLLM_SM120_FLASH_DECODE_LIB: the prebuilt extension (default: kfdec_moe_v8.so next to this file; v8 = v7 plus unrouted
slots: routing ids outside [0, E), vLLM's topk_ids == -1, contribute nothing, as in vLLM's expert_map path).
VLLM_SM120_FLASH_DECODE_CFG: "G:PDL:RBA:RBB:PFB:REGS" kernel schedule (default FROZEN_CFG below).
VLLM_SM120_FLASH_DECODE_LOG_S: seconds between calls-counter log lines (default 30).

Markers (server log, prefix SM120_FLASH_DECODE): "enabled ..." or "disabled: <reason>" at plugin load;
"moe: decode state for N layers" (N = 1 and 48); "moe: layer kept on Marlin: <reason>" once per reason;
"moe: first K-FDEC call, M=..." and "moe: first Marlin fallback, M=... (<reason>)" once each; then
"moe: K-FDEC calls N" whenever the count changed in the last LOG_S seconds. The count is kept by the kernels (kernel B
bumps a device counter and mirrors it to mapped pinned host memory, set up at weight load), so it keeps rising
during CUDA-graph replay, when this wrapper no longer runs; the log thread only reads host memory (no CUDA call).
MarlinExperts.apply runs inside vLLM's moe_forward custom op, so the compiled graph does not inline it and the
AOT compile cache needs no extra hash factor; decode CUDA graphs capture the two kernel launches (no host sync, no
allocation per call: the act scratch and fp32 global scales are made at weight load).
"""
import functools
import importlib.abc
import inspect
import os
import sys
import threading
import time

_RAW = os.environ.get("VLLM_SM120_FLASH_DECODE", "").strip()
OPTS = {t.strip() for t in _RAW.split(",") if t.strip() and t.strip() != "0"}
if "1" in OPTS:
    OPTS = (OPTS - {"1"}) | {"moe"}
MAX_M = int(os.environ.get("VLLM_SM120_FLASH_DECODE_MAX_M", "4"))
FROZEN_CFG = "4:1:4:5:0:0"
_CFG_RAW = os.environ.get("VLLM_SM120_FLASH_DECODE_CFG", FROZEN_CFG)
LIB = os.environ.get("VLLM_SM120_FLASH_DECODE_LIB",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "kfdec_moe_v8.so"))

_MOE_MOD = "vllm.model_executor.layers.quantization.modelopt"
_MARLIN_MOD = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
# MarlinExperts.apply parameters in this image, used when another wrapper hides the signature.
_APPLY_PARAMS = ("self", "output", "hidden_states", "w1", "w2", "topk_weights", "topk_ids", "activation",
                 "global_num_experts", "expert_map", "a1q_scale", "a2_scale", "workspace13", "workspace2",
                 "expert_tokens_meta", "apply_router_weight_on_input")

LOG_S = float(os.environ.get("VLLM_SM120_FLASH_DECODE_LOG_S", "30"))

_STATS = {"layers": 0, "reasons": set()}
_EXT = {}


def _log(msg: str) -> None:
    print(f"SM120_FLASH_DECODE {msg}", flush=True)


def _cfg():
    v = [int(t) for t in _CFG_RAW.split(":")]
    if len(v) != 6:
        raise ValueError(f"VLLM_SM120_FLASH_DECODE_CFG={_CFG_RAW!r} needs G:PDL:RBA:RBB:PFB:REGS")
    return {"groups": v[0], "pdl": bool(v[1]), "rba": v[2], "rbb": v[3], "pfb": bool(v[4]), "regs": v[5]}


def _ext():
    """The prebuilt extension module, or None after logging why (once)."""
    if "mod" in _EXT:
        return _EXT["mod"]
    _EXT["mod"] = None
    try:
        import importlib.machinery
        import importlib.util

        import torch  # noqa: F401  (loads libc10 / libtorch for the extension)
        name = os.path.splitext(os.path.basename(LIB))[0]
        loader = importlib.machinery.ExtensionFileLoader(name, LIB)
        spec = importlib.util.spec_from_file_location(name, LIB, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        _EXT["cfg"] = _cfg()
        _EXT["mod"] = mod
    except Exception as e:  # noqa: BLE001
        _log(f"disabled: K-FDEC extension {LIB} failed to load ({type(e).__name__}: {e})")
    return _EXT["mod"]


def _count_on() -> bool:
    """Set up the kernels' calls counter on the current device once (at weight load, before graph capture)."""
    if "count" not in _EXT:
        try:
            _EXT["mod"].counter_init()
            _EXT["count"] = True
        except Exception as e:  # noqa: BLE001
            _EXT["count"] = False
            _log(f"moe: calls counter unavailable ({type(e).__name__}: {e})")
    return _EXT["count"]


def _log_calls() -> None:
    last = None
    while True:
        time.sleep(LOG_S)
        n = _EXT["mod"].calls()
        if n != last:
            _log(f"moe: K-FDEC calls {n} at {time.strftime('%H:%M:%S', time.gmtime())} UTC (pid {os.getpid()})")
            last = n


def _kept(reason: str) -> None:
    if reason not in _STATS["reasons"]:
        _STATS["reasons"].add(reason)
        _log(f"moe: layer kept on Marlin: {reason}")


def _patch_modelopt(mod) -> None:
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    cls = mod.ModelOptNvFp4FusedMoE
    orig = cls.process_weights_after_loading

    def process_weights_after_loading(self, layer) -> None:
        orig(self, layer)
        experts = getattr(getattr(self, "moe_kernel", None), "fused_experts", None)
        if self.nvfp4_backend != NvFp4MoeBackend.MARLIN:
            return _kept(f"backend {self.nvfp4_backend}")
        if type(experts).__name__ != "MarlinExperts":
            return _kept(f"experts {type(experts).__name__}")
        if _ext() is None:
            return _kept("extension unavailable")
        q = getattr(experts, "quant_config", None)
        why = None
        if not self.moe.is_act_and_mul or getattr(self, "per_token_activation", False):
            why = "not act_and_mul / per-token activation"
        elif getattr(layer, "activation", MoEActivation.SILU) != MoEActivation.SILU:
            why = f"activation {getattr(layer, 'activation', None)}"
        elif getattr(layer, "expert_map", None) is not None:
            why = "expert_map (EP)"
        elif getattr(layer, "swiglu_limit", None) is not None:
            why = "swiglu_limit"
        elif q is not None and not getattr(q, "use_nvfp4_w4a16", False):
            why = "not NVFP4 W4A16"
        elif any(getattr(experts, a, None) is not None for a in ("w1_bias", "w2_bias", "w1_zp", "w2_zp",
                                                                  "a1_gscale", "a2_gscale")):
            why = "bias / zero point / activation global scale"
        elif getattr(experts, "input_dtype", None) not in (None, torch.bfloat16):
            why = f"Marlin input dtype {experts.input_dtype}"
        if why is None:
            w13, w2 = layer.w13_weight, layer.w2_weight
            s13, s2, g13, g2 = experts.w1_scale, experts.w2_scale, experts.g1_alphas, experts.g2_alphas
            E, I = w13.shape[0], w2.shape[1] * 16
            H = w13.shape[1] * 16
            if not (w13.dtype == torch.int32 and w2.dtype == torch.int32 and w13.shape[2] == 4 * I
                    and w2.shape[2] == 2 * H and E <= 512 and H % 64 == 0 and I % 64 == 0
                    and s13 is not None and s13.shape == (E, H // 16, 2 * I) and s2 is not None
                    and s2.shape == (E, I // 16, H) and g13 is not None and g13.numel() == E
                    and g2 is not None and g2.numel() == E
                    and all(t.is_contiguous() for t in (w13, w2, s13, s2))):
                why = f"shapes w13 {tuple(w13.shape)} w2 {tuple(w2.shape)}"
        if why is not None:
            return _kept(why)
        topk = layer.top_k if hasattr(layer, "top_k") else self.moe.experts_per_token
        _count_on()
        with torch.no_grad():
            experts._sm120_fdec = {
                "g13": g13.reshape(E).float().contiguous(), "g2": g2.reshape(E).float().contiguous(),
                "act": torch.empty(16 * topk, I, dtype=torch.bfloat16, device=w13.device),
                "H": H, "I": I, "topk": topk}
        _STATS["layers"] += 1
        if _STATS["layers"] in (1, 48):
            _log(f"moe: decode state for {_STATS['layers']} layers (E={E} H={H} I={I} topk={topk}, "
                 f"M <= {MAX_M})")
    cls.process_weights_after_loading = process_weights_after_loading


def _patch_marlin(mod) -> None:
    import torch
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    cls = mod.MarlinExperts
    orig = cls.apply
    sig = inspect.signature(orig)
    if "hidden_states" not in sig.parameters:  # another wrapper without functools.wraps
        sig = inspect.Signature([inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                 for p in _APPLY_PARAMS])

    def fallback(M, why):
        if not _STATS.get("fallback"):
            _STATS["fallback"] = 1
            _log(f"moe: first Marlin fallback, M={M} ({why})")

    @functools.wraps(orig)
    def apply(self, *args, **kwargs):
        st = getattr(self, "_sm120_fdec", None)
        if st is None:
            return orig(self, *args, **kwargs)
        a = sig.bind(self, *args, **kwargs).arguments
        x, out, ids, tw = a["hidden_states"], a["output"], a["topk_ids"], a["topk_weights"]
        M = x.shape[0] if x.dim() == 2 else -1
        if not 1 <= M <= MAX_M:
            fallback(M, f"M outside 1..{MAX_M}")
            return orig(self, *args, **kwargs)
        if (x.dtype != torch.bfloat16 or not x.is_contiguous() or x.shape[1] != st["H"]
                or out.dtype != torch.bfloat16 or not out.is_contiguous() or tuple(out.shape) != (M, st["H"])
                or ids.dim() != 2 or ids.shape != (M, st["topk"]) or not ids.is_contiguous()
                or ids.dtype not in (torch.int32, torch.int64) or tw.dtype != torch.float32
                or not tw.is_contiguous() or tuple(tw.shape) != (M, st["topk"])
                or a["expert_map"] is not None or a["apply_router_weight_on_input"]
                or a["activation"] != MoEActivation.SILU or getattr(self, "_lora_context", None) is not None):
            fallback(M, "call not eligible")
            return orig(self, *args, **kwargs)
        c = _EXT["cfg"]
        if not _STATS.get("calls"):
            _STATS["calls"] = 1
            _log(f"moe: first K-FDEC call, M={M} cfg={_CFG_RAW} counter={_EXT['count']} "
                 f"capturing={torch.cuda.is_current_stream_capturing()}")
            if _EXT["count"]:
                threading.Thread(target=_log_calls, name="sm120-fdec-calls", daemon=True).start()
        _EXT["mod"].moe_decode(x, a["w1"], self.w1_scale, st["g13"], a["w2"], self.w2_scale, st["g2"], ids, tw,
                               out, st["act"], c["groups"], c["pdl"], c["rba"], c["rbb"], c["pfb"], c["regs"],
                               _EXT["count"])
    cls.apply = apply


class _PostImport(importlib.abc.MetaPathFinder):
    """Runs a patch function right after the named module executes. Chains with other hooks of this kind on the same
    module (kmoe1's sm120_flash_prefill asks every other finder): it asks only the finders after itself (those before
    it have declined or are delegating to it), and returns None if asked again while it is searching, so neither
    order of the two hooks recurses and each patch is applied once."""

    def __init__(self, targets):
        self.targets = targets
        self.busy = set()

    def find_spec(self, name, path, target=None):
        fn = self.targets.get(name)
        if fn is None or name in self.busy:
            return None
        mp = list(sys.meta_path)
        later = mp[mp.index(self) + 1:] if self in mp else mp
        self.busy.add(name)
        try:
            for finder in later:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    break
            else:
                return None
        finally:
            self.busy.discard(name)
        self.targets.pop(name, None)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            fn(module)
        spec.loader.exec_module = exec_and_patch
        return spec


def register() -> None:
    if getattr(register, "done", False):
        return
    register.done = True
    if not OPTS:
        _log("disabled: VLLM_SM120_FLASH_DECODE unset or 0")
        return
    if OPTS != {"moe"}:
        _log(f"disabled: unknown VLLM_SM120_FLASH_DECODE options {sorted(OPTS - {'moe'})}")
        return
    try:
        cfg = _cfg()
    except ValueError as e:
        _log(f"disabled: {e}")
        return
    if not LOG_S > 0:
        _log(f"disabled: VLLM_SM120_FLASH_DECODE_LOG_S={LOG_S} must be > 0")
        return
    if not 1 <= MAX_M <= 16:
        _log(f"disabled: VLLM_SM120_FLASH_DECODE_MAX_M={MAX_M} outside 1..16")
        return
    if not os.path.exists(LIB):
        _log(f"disabled: K-FDEC extension {LIB} not found")
        return
    targets = {_MOE_MOD: _patch_modelopt, _MARLIN_MOD: _patch_marlin}
    pending = {}
    for name, fn in targets.items():
        if name in sys.modules:
            fn(sys.modules[name])
        else:
            pending[name] = fn
    if pending:
        sys.meta_path.insert(0, _PostImport(pending))
    _log(f"enabled moe max_m={MAX_M} cfg={cfg} lib={LIB} pid={os.getpid()}")
