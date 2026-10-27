#!/usr/bin/env python3
"""Single-stream decode benchmark against an OpenAI-compatible server (vLLM or SGLang). Stdlib only.

Each request streams; we record TTFT, the decode window (first to last content chunk) and the
server-reported completion tokens. Decode rate = (completion_tokens - 1) / decode window.

Long prompts are built from the Django 5.1 source (see fetch_data.sh): 64K, 128K and 256K contexts, three tasks each
(review, tests, explain), 2,048 output tokens, greedy, thinking on. This is a capped-output decode benchmark; its
``e2e`` field is request latency, not time to a correct, completed task.

Usage: bench.py --label NAME --long-src data/django/django [--contexts 64,128,256]
"""
import argparse
import http.client
import json
import statistics
import sys
import time
import zlib
from pathlib import Path

SHORT = {
    'math': 'A train leaves city A at 9:00 travelling at 84 km/h. A second train leaves city B, 501 km away, '
            'at 10:30 travelling toward A at 96 km/h. At what time do they meet, and how far from A? '
            'Then generalise: derive a formula for the meeting time given arbitrary speeds, distance and delay.',
    'code': 'Write a production-quality Python module implementing an LRU cache with TTL expiry, thread safety, '
            'size-based eviction (bytes, not entries) and hit/miss statistics. Include type hints, docstrings '
            'and a thorough pytest test suite.',
    'prose': 'Explain to a senior software engineer how speculative decoding works in LLM inference, why it is '
             'lossless under greedy and rejection sampling, and what limits its speedup on memory-bound GPUs.',
    'refactor': None,  # filled from file below: copy-heavy edit task
}
REFACTOR_SRC = '''
def process(records):
    out = []
    for r in records:
        if r.get("type") == "a":
            v = r["value"] * 2
            if v > 10:
                out.append({"id": r["id"], "v": v, "kind": "big-a"})
            else:
                out.append({"id": r["id"], "v": v, "kind": "small-a"})
        elif r.get("type") == "b":
            v = r["value"] + 7
            if v > 10:
                out.append({"id": r["id"], "v": v, "kind": "big-b"})
            else:
                out.append({"id": r["id"], "v": v, "kind": "small-b"})
        elif r.get("type") == "c":
            v = r["value"] ** 2
            if v > 10:
                out.append({"id": r["id"], "v": v, "kind": "big-c"})
            else:
                out.append({"id": r["id"], "v": v, "kind": "small-c"})
    return out
'''
SHORT['refactor'] = ('Refactor this function to remove duplication using a dispatch table, keep behaviour identical, '
                     'add type hints, then print the complete refactored file and a short changelog.\n```python'
                     + REFACTOR_SRC * 3 + '```')


TASKS = {
    'review': 'Identify the three most important design weaknesses in this codebase. For each, cite the '
              'file and function, explain the problem, and write the corrected code.',
    'tests': 'Pick the module above with the most complex control flow. Write a complete pytest test module '
             'for it covering edge cases, using fixtures and parametrization. Output only the code.',
    'explain': 'Write a detailed technical overview of this codebase for a new maintainer: the main subsystems, '
               'how a request flows through them, the key extension points and notable conventions. Use headings.',
}


def long_prompt(path, chars, task='review'):
    files = sorted(p for p in Path(path).rglob('*.py') if p.is_file())
    parts, total = [], 0
    for p in files:
        t = p.read_text(errors='ignore')
        parts.append(f'\n### FILE: {p.relative_to(path)}\n{t}')
        total += len(parts[-1])
        if total >= chars:
            break
    body = ''.join(parts)[:chars]
    return 'You are reviewing the repository below.\n' + body + '\n\n### TASK\n' + TASKS[task]


def metrics(host, port):
    try:
        c = http.client.HTTPConnection(host, port, timeout=10)
        c.request('GET', '/metrics')
        text = c.getresponse().read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        for key in ('spec_decode_num_accepted_tokens_total', 'spec_decode_num_drafts_total',
                    'spec_decode_num_draft_tokens_total', 'prefix_cache_hits_total'):
            if key in line.split('{')[0]:
                out[key] = out.get(key, 0) + float(line.rsplit(' ', 1)[1])
    return out


def one(host, port, model, prompt, max_tokens, temp, extra):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens,
            'temperature': temp, 'stream': True, 'stream_options': {'include_usage': True}, 'seed': 42}
    if temp > 0:
        body.update(top_p=0.95, top_k=20)
    body.update(extra)
    c = http.client.HTTPConnection(host, port, timeout=3600)
    t0 = time.perf_counter()
    c.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(r.read().decode()[:500])
    first = last = None
    usage, text, reasoning, chunks = None, [], [], 0
    buf = b''
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b'data:'):
            continue
        data = line[5:].strip()
        if data == b'[DONE]':
            break
        ev = json.loads(data)
        if ev.get('usage'):
            usage = ev['usage']
        for ch in ev.get('choices', []):
            d = ch.get('delta', {})
            piece = d.get('content') or ''
            rpiece = d.get('reasoning_content') or d.get('reasoning') or ''
            if piece or rpiece:
                now = time.perf_counter()
                first = first or now
                last = now
                chunks += 1
                text.append(piece)
                reasoning.append(rpiece)
    t_end = time.perf_counter()
    n = usage['completion_tokens'] if usage else None
    window = (last - first) if first and last else None
    return {'ttft': first - t0 if first else None, 'e2e': t_end - t0, 'completion_tokens': n,
            'prompt_tokens': usage['prompt_tokens'] if usage else None, 'chunks': chunks,
            'decode_tps': (n - 1) / window if n and window else None,
            'text': ''.join(text), 'reasoning': ''.join(reasoning)}


def loop_check(out):
    """Degenerate repetition: zlib ratio of the whole output and of its last 8,000 characters (normal
    text 0.27-0.41 over the bench outputs; loops 0.03-0.25), and the share of repeated non-empty lines."""
    def z(t):
        b = t.encode()
        return round(len(zlib.compress(b, 9)) / len(b), 3) if len(b) >= 2000 else None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    dup = round(1 - len(set(lines)) / len(lines), 3) if len(lines) >= 40 else None
    zf, zt = z(out), z(out[-8000:])
    loop = bool((zf is not None and zf < 0.25) or (zt is not None and zt < 0.2) or (dup is not None and dup > 0.5))
    return {'z': zf, 'z_tail': zt, 'dup_lines': dup, 'loop': loop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--model', default='q27')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', default='results')
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--temp', type=float, default=0.0)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--long-src', required=True, help='the django/django package directory of a Django 5.1 checkout')
    ap.add_argument('--contexts', default='64,128,256', help='prompt sizes in K tokens')
    ap.add_argument('--short', action='store_true', help='also run the short prompts')
    ap.add_argument('--tasks', default='review,tests,explain', help='long-context tasks per size')
    ap.add_argument('--only', default='')
    ap.add_argument('--extra', default='{}')
    a = ap.parse_args()
    extra = json.loads(a.extra)
    prompts = dict(SHORT)
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only.split(',')}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # warmup (graphs, JIT) excluded from results
    one(a.host, a.port, a.model, 'Say hi.', 32, a.temp, extra)
    cases = [(k, v, a.max_tokens) for k, v in prompts.items()] if a.short else []
    # 3.545 chars/token measured on this corpus with the Qwen3.8 tokenizer; 256K is capped to fit 262,144 with output
    for k in [int(x) for x in a.contexts.split(',') if x]:
        tokens = min(k * 1000, 252000)
        for task in a.tasks.split(','):
            cases.append((f'ctx{k}k:{task}', long_prompt(a.long_src, int(tokens * 3.545), task), a.max_tokens))
    for rep in range(a.repeats):
        for name, prompt, mt in cases:
            m0 = metrics(a.host, a.port)
            res = one(a.host, a.port, a.model, prompt, mt, a.temp, extra)
            m1 = metrics(a.host, a.port)
            d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
            acc = d.get('spec_decode_num_accepted_tokens_total')
            drafts = d.get('spec_decode_num_drafts_total')
            res.update(case=name, rep=rep, label=a.label, spec_accepted=acc, spec_drafts=drafts,
                       accept_len=(1 + acc / drafts) if acc is not None and drafts else None,
                       **loop_check(res['reasoning'] + res['text']))
            rows.append(res)
            print(f"{a.label:28s} {name:16s} rep{rep} prompt={res['prompt_tokens']} gen={res['completion_tokens']} "
                  f"ttft={res['ttft']:.2f}s decode={res['decode_tps']:.1f} tok/s "
                  f"accept_len={res['accept_len'] if res['accept_len'] is None else round(res['accept_len'], 2)} "
                  f"z={res['z']}/{res['z_tail']}{' LOOP' if res['loop'] else ''}",
                  flush=True)
    (out / f'{a.label}.json').write_text(json.dumps(rows, indent=1))
    # Per context: pooled decode rate over all tasks and reps, mean acceptance, implied verify steps/s.
    # Looping outputs are easy to draft and inflate the rate, so the loop-free pool is reported as well.
    by = {}
    for r in rows:
        ctx = r['case'].split(':')[0]
        for key in (ctx, ctx + '/noloop') if not r['loop'] else (ctx,):
            g = by.setdefault(key, {'tok': 0, 'win': 0.0, 'acc': 0.0, 'drafts': 0.0, 'loops': 0, 'n': 0})
            g['n'] += 1
            g['loops'] += r['loop']
            if r['decode_tps']:
                g['tok'] += r['completion_tokens'] - 1
                g['win'] += (r['completion_tokens'] - 1) / r['decode_tps']
            if r.get('spec_drafts'):
                g['acc'] += r['spec_accepted']
                g['drafts'] += r['spec_drafts']

    def pooled(g):
        tps = g['tok'] / g['win'] if g['win'] else None
        acc = 1 + g['acc'] / g['drafts'] if g['drafts'] else None
        return tps, acc

    summary = {}
    for k, g in by.items():
        if k.endswith('/noloop'):
            continue
        tps, acc = pooled(g)
        summary[k] = {'tps': round(tps, 1) if tps else None, 'accept': round(acc, 2) if acc else None,
                      'steps_s': round(tps / acc, 1) if tps and acc else None, 'cases': g['n'], 'loops': g['loops']}
        if g['loops']:
            t2, a2 = pooled(by[k + '/noloop']) if k + '/noloop' in by else (None, None)
            summary[k].update(tps_noloop=round(t2, 1) if t2 else None, accept_noloop=round(a2, 2) if a2 else None)
    (out / f'{a.label}.summary.json').write_text(json.dumps(summary, indent=1))
    print('SUMMARY', a.label, json.dumps(summary))


if __name__ == '__main__':
    sys.exit(main())
