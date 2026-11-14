#!/usr/bin/env python3
"""Quantize the BF16 dense linears of NVIDIA's NVFP4 Qwen3.8 Flash Next checkpoint to NVFP4.

flash_nvfp4_convert.py SRC OUT [--groups gdn,attn] [--ref-fp8 DIR] [--device cuda] [--check-only]

SRC is `nvidia/Qwen3.8-Flash-Next-NVFP4` (routed experts NVFP4, every dense linear BF16 and listed under
`ignore`). The selected groups of decoder-layer linears are rebuilt in ModelOpt NVFP4 layout with
nvfp4_convert.py's MSE block-scale quantizer (packed e2m1 weight [N, K/2] uint8, e4m3 block scale [N, K/16],
fp32 global weight_scale_2, fp32 input_scale) and listed as NVFP4 in `quantized_layers`:

* gdn:  linear_attn.in_proj_qkv + in_proj_z (one global scale: vLLM fuses them into in_proj_qkvz) and
        linear_attn.out_proj; in_proj_a/in_proj_b and conv1d stay BF16.
* attn: self_attn.q_proj + k_proj + v_proj + indexer.index_qk_proj (one global scale: fused qkv_proj) and
        self_attn.o_proj.

Served with Marlin W4A16 (`--kernel-config {"linear_backend_per_quant":{"nvfp4_w4a4":"marlin"}}`), so the
input_scale is a placeholder (1.0) that the W4A16 path does not read. The MTP layer, lm_head, HC, router and
shared experts are untouched (the shared expert is left BF16 so vLLM does not try to fuse it into the
routed experts). With --ref-fp8 (the FP8 release), the packing convention is checked first on an NVIDIA
NVFP4 expert against its block-FP8 counterpart. Unchanged files are hard-linked (directories such as the
hub `.cache` are skipped); rewritten shards keep their names. A rerun into the same OUT keeps shards that a
previous run already rewrote (they are written via a temporary name, so a present shard is complete).
"""
import argparse
import importlib.util
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

_spec = importlib.util.spec_from_file_location(
    'nvfp4_convert', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nvfp4_convert.py'))
NC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NC)

LAYER = 'model.language_model.layers.'
GROUPS = {
    'gdn': {'linear_attn.in_proj_qkv': 'qkvz', 'linear_attn.in_proj_z': 'qkvz',
            'linear_attn.out_proj': 'out'},
    'attn': {'self_attn.q_proj': 'qkv', 'self_attn.k_proj': 'qkv', 'self_attn.v_proj': 'qkv',
             'self_attn.indexer.index_qk_proj': 'qkv', 'self_attn.o_proj': 'o'},
}
# `ignore` wildcards that cover each group, and the BF16 linears of the group's sub-module that stay
# excluded explicitly once the wildcard is gone.
WILDCARD = {'gdn': 'linear_attn*', 'attn': 'self_attn*'}
KEEP = {'gdn': ['linear_attn.in_proj_a', 'linear_attn.in_proj_b', 'linear_attn.conv1d'], 'attn': []}


def row_fp8_err(w):
    """Relative error of the per-row e4m3 weights the W8A16 path serves today."""
    w = w.float()
    s = (w.abs().amax(dim=1) / 448.0).clamp(min=1e-12)
    q = (w / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s[:, None]
    return NC.rel(q, w)


def block_fp8_dequant(w, s):
    n, k = w.shape
    return w.float() * s.float().repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--groups', default='gdn,attn')
    ap.add_argument('--ref-fp8')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = a.device
    groups = [g for g in a.groups.split(',') if g]
    assert groups and all(g in GROUPS for g in groups), groups
    sidx = json.load(open(f'{a.src}/model.safetensors.index.json'))
    smap = sidx['weight_map']

    def get(d, m, name):
        with safe_open(f'{d}/{m[name]}', 'pt', device=dev) as f:
            return f.get_tensor(name)

    if a.ref_fp8:
        rmap = json.load(open(f'{a.ref_fp8}/model.safetensors.index.json'))['weight_map']
        p = LAYER + '0.mlp.experts.0.gate_proj'
        ref = block_fp8_dequant(get(a.ref_fp8, rmap, p + '.weight'), get(a.ref_fp8, rmap, p + '.weight_scale_inv'))
        wq, sc, s2 = (get(a.src, smap, p + s) for s in ('.weight', '.weight_scale', '.weight_scale_2'))
        e_lo = NC.rel(NC.unpack(wq, sc, s2, True), ref)
        e_hi = NC.rel(NC.unpack(wq, sc, s2, False), ref)
        mine = NC.rel(NC.unpack(*NC.quantize(ref, ref.abs().max().item()), True), ref)
        print(f'{p}: nvidia vs block-FP8 low-first {e_lo:.4f} high-first {e_hi:.4f}; '
              f'this quantizer on the FP8 weights {mine:.4f}', flush=True)
        assert e_lo < 0.2 and e_lo < e_hi, 'unexpected NVFP4 packing'

    leaf_group = {leaf: (g, fuse) for g in groups for leaf, fuse in GROUPS[g].items()}
    mods = []
    for name in smap:
        mt = re.match(re.escape(LAYER) + r'(\d+)\.(.+)\.weight$', name)
        if mt and mt.group(2) in leaf_group:
            mods.append(name[:-len('.weight')])
    mods.sort(key=lambda m: (int(m[len(LAYER):].split('.')[0]), m))
    print(f'{len(mods)} BF16 modules to convert ({",".join(groups)})', flush=True)
    fused = {}
    for m in mods:
        layer, leaf = m[len(LAYER):].split('.', 1)
        fused.setdefault((layer, leaf_group[leaf][1]), []).append(m)
    gamax = {}
    for key, ms in fused.items():
        g = max(get(a.src, smap, m + '.weight').float().abs().max().item() for m in ms)
        for m in ms:
            gamax[m] = g
    seen = set()
    for m in mods:
        leaf = m[len(LAYER):].split('.', 1)[1]
        if leaf in seen:
            continue
        seen.add(leaf)
        w = get(a.src, smap, m + '.weight')
        assert w.dtype == torch.bfloat16, (m, w.dtype)
        e4 = NC.rel(NC.unpack(*NC.quantize(w, gamax[m]), True), w)
        print(f'{m} {tuple(w.shape)}: per-row fp8 err {row_fp8_err(w):.4f} nvfp4 err {e4:.4f}', flush=True)
    if a.check_only:
        return

    os.makedirs(a.out, exist_ok=True)
    by_shard = {}
    for m in mods:
        by_shard.setdefault(smap[m + '.weight'], []).append(m)
    new_map = dict(smap)
    for shard in sorted(by_shard):
        for m in by_shard[shard]:
            for s in ('.weight_scale', '.weight_scale_2', '.input_scale'):
                new_map[m + s] = shard
        dst = f'{a.out}/{shard}'
        if os.path.exists(dst) and not os.path.samefile(dst, f'{a.src}/{shard}'):
            print(f'kept {shard} from a previous run', flush=True)
            continue
        if os.path.exists(dst):
            os.remove(dst)
        t = load_file(f'{a.src}/{shard}')
        for m in by_shard[shard]:
            packed, scale, ws2 = NC.quantize(t[m + '.weight'].to(dev), gamax[m])
            t[m + '.weight'] = packed.cpu()
            t[m + '.weight_scale'] = scale.cpu()
            t[m + '.weight_scale_2'] = ws2
            t[m + '.input_scale'] = torch.tensor(1.0, dtype=torch.float32)
        save_file(t, f'{a.out}/{shard}.tmp', metadata={'format': 'pt'})
        os.replace(f'{a.out}/{shard}.tmp', f'{a.out}/{shard}')
        print(f'rewrote {shard} ({len(by_shard[shard])} modules)', flush=True)
        del t
    for fn in os.listdir(a.src):
        dst = f'{a.out}/{fn}'
        if os.path.exists(dst) or os.path.isdir(f'{a.src}/{fn}') or \
                fn in ('hf_quant_config.json', 'config.json', 'model.safetensors.index.json'):
            continue
        try:
            os.link(f'{a.src}/{fn}', dst)
        except OSError:
            shutil.copy2(f'{a.src}/{fn}', dst)
    sidx['weight_map'] = new_map
    json.dump(sidx, open(f'{a.out}/model.safetensors.index.json', 'w'), indent=2)

    layers = sorted({m[len(LAYER):].split('.')[0] for m in mods}, key=int)
    drop = {f'{LAYER}{n}.{WILDCARD[g]}' for g in groups for n in layers}
    keep = [f'{LAYER}{n}.{k}' for g in groups for n in layers for k in KEEP[g]
            if any(x.startswith(f'{LAYER}{n}.{k}.') for x in smap)]
    entry = {'quant_algo': 'NVFP4', 'group_size': 16}

    def edit(ignore, ql):
        out = [x for x in ignore if x not in drop] + keep
        assert len(out) == len(ignore) - len(drop & set(ignore)) + len(keep)
        for m in mods:
            ql[m] = dict(entry)
        return out

    hq = json.load(open(f'{a.src}/hf_quant_config.json'))
    q = hq['quantization']
    q['exclude_modules'] = edit(q['exclude_modules'], q['quantized_layers'])
    json.dump(hq, open(f'{a.out}/hf_quant_config.json', 'w'), indent=1)
    cfg = json.load(open(f'{a.src}/config.json'))
    qc = cfg['quantization_config']
    qc['ignore'] = edit(qc['ignore'], qc['quantized_layers'])
    nv_group = next(g for g in qc['config_groups'].values() if g.get('weights', {}).get('num_bits') == 4)
    nv_group['targets'].extend(mods)
    json.dump(cfg, open(f'{a.out}/config.json', 'w'), indent=2)

    # Round-trip check on the written files: one converted module per type.
    omap = new_map
    seen = set()
    for m in mods:
        leaf = m[len(LAYER):].split('.', 1)[1]
        if leaf in seen:
            continue
        seen.add(leaf)
        w = get(a.src, smap, m + '.weight')
        deq = NC.unpack(*(get(a.out, omap, m + s) for s in ('.weight', '.weight_scale', '.weight_scale_2')), True)
        print(f'written {m}: nvfp4 err {NC.rel(deq, w):.4f}', flush=True)
    print(f'done: {len(mods)} modules NVFP4, {len(drop)} ignore wildcards removed, {len(keep)} explicit '
          f'BF16 entries added', flush=True)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()
