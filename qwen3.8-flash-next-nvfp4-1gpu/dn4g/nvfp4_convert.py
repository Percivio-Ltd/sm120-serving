#!/usr/bin/env python3
"""Re-quantize the FP8 linears of a ModelOpt MIXED_PRECISION checkpoint to NVFP4 from the BF16 originals.

nvfp4_convert.py SRC BF16 OUT [--check-only] [--device cuda]

SRC is the NVIDIA NVFP4 checkpoint (MLP in NVFP4, attention/GDN projections in per-tensor FP8), BF16 the
original model with the same tensor names, OUT the new directory. Every module listed as FP8 under
model.language_model.layers is rebuilt in ModelOpt NVFP4 layout (packed e2m1 weight [N, K/2] uint8, e4m3
block scale [N, K/16], fp32 global weight_scale_2, fp32 input_scale) and its quantized_layers entry
becomes NVFP4 with group size 16. Modules that vLLM fuses (q/k/v and in_proj_qkv/in_proj_z) share one
global scale. Block scales use an MSE search over amax fractions, like ModelOpt's NVFP4MSECalibrator.

First the packing convention is checked against an existing NVFP4 layer of SRC (dequantized against
BF16), and this quantizer's error on that layer is compared with NVIDIA's. Unchanged files are
hard-linked; rewritten shards keep their names.
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
FUSED = {'q_proj': 'qkv', 'k_proj': 'qkv', 'v_proj': 'qkv', 'in_proj_qkv': 'qkvz', 'in_proj_z': 'qkvz'}
FRACTIONS = torch.linspace(0.70, 1.0, 16).tolist()


def unpack(w_u8, scale, ws2, low_first=True):
    lo, hi = (w_u8 & 0xF).long(), (w_u8 >> 4).long()
    a, b = (lo, hi) if low_first else (hi, lo)
    codes = torch.stack([a, b], dim=-1).flatten(-2)
    vals = E2M1.to(w_u8.device)[codes & 7] * (1 - 2 * (codes >> 3)).float()
    return vals * scale.float().repeat_interleave(16, dim=1) * ws2.float()


def quantize(w, global_amax):
    """MSE block-scale NVFP4 quantization of a [N, K] tensor with a given global amax."""
    w = w.float()
    n, k = w.shape
    ws2 = global_amax / (6.0 * 448.0)
    blocks = w.view(n, k // 16, 16)
    amax = blocks.abs().amax(-1).clamp(min=1e-12)
    bounds, table = BOUNDS.to(w.device), E2M1.to(w.device)
    best_err = best_scale = None
    for f in FRACTIONS:
        s = (amax * f / 6.0 / ws2).clamp(max=448.0).to(torch.float8_e4m3fn)
        sf = s.float() * ws2
        sf = torch.where(sf > 0, sf, torch.ones_like(sf))
        x = blocks / sf[..., None]
        q = table[torch.bucketize(x.abs().clamp(max=6.0), bounds)] * x.sign()
        err = (q * sf[..., None] - blocks).square().sum(-1)
        if best_err is None:
            best_err, best_scale = err, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, s, best_scale)
    sf = best_scale.float() * ws2
    sf = torch.where(sf > 0, sf, torch.ones_like(sf))
    x = (blocks / sf[..., None]).clamp(-6.0, 6.0)
    codes = torch.bucketize(x.abs(), bounds) | ((x < 0).long() << 3)
    codes = codes.view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, best_scale, torch.tensor(ws2, dtype=torch.float32)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('bf16')
    ap.add_argument('out')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    dev = a.device
    sidx = json.load(open(f'{a.src}/model.safetensors.index.json'))
    smap = sidx['weight_map']
    bmap = json.load(open(f'{a.bf16}/model.safetensors.index.json'))['weight_map']
    hq = json.load(open(f'{a.src}/hf_quant_config.json'))
    ql = hq['quantization']['quantized_layers']

    def bf16(name):
        with safe_open(f'{a.bf16}/{bmap[name]}', 'pt', device=dev) as f:
            return f.get_tensor(name)

    def src(name):
        with safe_open(f'{a.src}/{smap[name]}', 'pt', device=dev) as f:
            return f.get_tensor(name)

    # Packing check on an existing NVIDIA NVFP4 layer.
    p = 'model.language_model.layers.0.mlp.gate_proj'
    ref = bf16(p + '.weight')
    wq, sc, s2 = src(p + '.weight'), src(p + '.weight_scale'), src(p + '.weight_scale_2')
    e_lo, e_hi = rel(unpack(wq, sc, s2, True), ref), rel(unpack(wq, sc, s2, False), ref)
    mine = quantize(ref, ref.float().abs().max().item())
    e_mine = rel(unpack(*mine, True), ref)
    print(f'{p}: nvidia err low-first {e_lo:.4f} high-first {e_hi:.4f}; this quantizer {e_mine:.4f}', flush=True)
    assert e_lo < 0.2 and e_lo < e_hi, 'unexpected NVFP4 packing'

    fp8 = sorted(m for m, info in ql.items() if info.get('quant_algo') == 'FP8'
                 and m.startswith('model.language_model.layers.'))
    print(f'{len(fp8)} FP8 modules to convert', flush=True)
    groups = {}
    for m in fp8:
        layer, leaf = m.rsplit('.', 1)
        groups.setdefault((layer, FUSED.get(leaf, leaf)), []).append(m)
    gamax = {}
    for key, mods in groups.items():
        g = max(bf16(m + '.weight').float().abs().max().item() for m in mods)
        for m in mods:
            gamax[m] = g
    # One FP8 module per type: error of FP8 (as served today) and of the new NVFP4.
    seen = set()
    for m in fp8:
        leaf = m.rsplit('.', 1)[1]
        if leaf in seen:
            continue
        seen.add(leaf)
        w = bf16(m + '.weight')
        e8 = rel(src(m + '.weight').float() * src(m + '.weight_scale').float(), w)
        e4 = rel(unpack(*quantize(w, gamax[m]), True), w)
        print(f'{m}: fp8 err {e8:.4f} nvfp4 err {e4:.4f}', flush=True)
    if a.check_only:
        return

    os.makedirs(a.out, exist_ok=True)
    by_shard = {}
    for m in fp8:
        by_shard.setdefault(smap[m + '.weight'], []).append(m)
    new_map = dict(smap)
    for shard in sorted(set(smap.values())):
        if shard not in by_shard:
            continue
        t = load_file(f'{a.src}/{shard}')
        for m in by_shard[shard]:
            packed, scale, ws2 = quantize(bf16(m + '.weight').to(dev), gamax[m])
            t[m + '.weight'] = packed.cpu()
            t[m + '.weight_scale'] = scale.cpu()
            t[m + '.weight_scale_2'] = ws2
            if m + '.input_scale' in t:
                t[m + '.input_scale'] = (t[m + '.input_scale'].float() / 6.0).reshape(())
            new_map[m + '.weight_scale_2'] = shard
            ql[m] = {'quant_algo': 'NVFP4', 'group_size': 16}
        save_file(t, f'{a.out}/{shard}.tmp', metadata={'format': 'pt'})
        os.replace(f'{a.out}/{shard}.tmp', f'{a.out}/{shard}')
        print(f'rewrote {shard} ({len(by_shard[shard])} modules)', flush=True)
        del t
    for fn in os.listdir(a.src):
        dst = f'{a.out}/{fn}'
        if os.path.exists(dst) or fn in ('hf_quant_config.json', 'config.json', 'model.safetensors.index.json'):
            continue
        try:
            os.link(f'{a.src}/{fn}', dst)
        except OSError:
            shutil.copy2(f'{a.src}/{fn}', dst)
    sidx['weight_map'] = new_map
    json.dump(sidx, open(f'{a.out}/model.safetensors.index.json', 'w'), indent=2)
    json.dump(hq, open(f'{a.out}/hf_quant_config.json', 'w'), indent=1)
    cfg = json.load(open(f'{a.src}/config.json'))
    qc = cfg.get('quantization_config')
    if qc is not None:
        if 'quantized_layers' in qc:
            qc['quantized_layers'] = ql
        groups_cfg = qc.get('config_groups') or {}
        conv = set(fp8)
        nv_group = next((g for g in groups_cfg.values() if g.get('weights', {}).get('num_bits') == 4), None)
        for g in groups_cfg.values():
            if g is nv_group:
                continue
            tg = g.get('targets', [])
            moved = [x for x in tg if x in conv]
            if moved and nv_group is not None:
                g['targets'] = [x for x in tg if x not in conv]
                nv_group.setdefault('targets', []).extend(moved)
    json.dump(cfg, open(f'{a.out}/config.json', 'w'), indent=2)
    print('done', flush=True)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()
