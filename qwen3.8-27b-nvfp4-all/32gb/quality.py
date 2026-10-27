#!/usr/bin/env python3
"""Small quality check against a running OpenAI-compatible server. Stdlib only.

gsm8k: the first N GSM8K test problems, greedy, thinking on, final answer in \\boxed{}; 4 concurrent requests.
needle: long Django prompts (64K/128K/200K tokens) with eight planted passphrases at seeded depths;
        each probe asks for one passphrase (thinking off) and scores an exact match.

The needle prompts reach about 200K tokens, so the server needs a max-model-len of at least 204800.

Usage: quality.py --label NAME --gsm8k-file data/gsm8k_test.jsonl --long-src data/django/django [--gsm8k N] [--needles PER_CTX]
"""
import argparse
import http.client
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

HOST, PORT, MODEL = '127.0.0.1', 8000, 'q27'
WORDS = ['amber', 'basalt', 'cobalt', 'dune', 'ember', 'fjord', 'garnet', 'harbor', 'indigo', 'juniper',
         'krypton', 'lagoon', 'marble', 'nectar', 'onyx', 'pepper', 'quartz', 'raven', 'saffron', 'tundra']
SERVICES = ['billing', 'search', 'ledger', 'gateway', 'scheduler', 'archive', 'telemetry', 'identity']


def chat(prompt, max_tokens, thinking=True):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': max_tokens,
            'temperature': 0.0, 'seed': 42}
    if not thinking:
        body['chat_template_kwargs'] = {'enable_thinking': False}
    c = http.client.HTTPConnection(HOST, PORT, timeout=3600)
    c.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
    r = json.loads(c.getresponse().read())
    m = r['choices'][0]['message']
    return m.get('content') or '', r['usage']['completion_tokens']


def gsm_answer(text):
    b = re.findall(r'\\boxed\{([^{}]*)\}', text)
    s = b[-1] if b else (re.findall(r'-?[\d,]*\.?\d+', text) or [''])[-1]
    s = s.replace(',', '').replace('$', '').strip()
    try:
        return float(s)
    except ValueError:
        return None


def run_gsm(path, n):
    items = [json.loads(line) for line in open(path)][:n]

    def solve(it):
        text, ntok = chat(it['question'] + '\nPlease reason step by step, and put your final answer '
                          'within \\boxed{}.', 8192)
        gold = float(it['answer'].split('####')[-1].replace(',', '').strip())
        got = gsm_answer(text)
        return {'q': it['question'][:60], 'gold': gold, 'got': got, 'ok': got is not None and abs(got - gold) < 1e-6,
                'tokens': ntok, 'answer': text[-300:]}
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(solve, items))


def run_needles(src, per_ctx):
    rows = []
    rng = random.Random(1234)
    for k in (64, 128, 200):
        base = bench.long_prompt(src, int(k * 1000 * 3.545))
        base = base[:base.index('\n\n### TASK')]
        lines = base.split('\n')
        for probe in range(per_ctx):
            secrets = {s: f'{rng.choice(WORDS)}-{rng.randint(1000, 9999)}' for s in SERVICES}
            pos = sorted(rng.sample(range(1, len(lines)), len(SERVICES)), reverse=True)
            ls = list(lines)
            for p, (s, v) in zip(pos, secrets.items()):
                ls.insert(p, f'# NOTE: the deployment passphrase for the {s} service is {v}.')
            ask = rng.choice(SERVICES)
            prompt = ('\n'.join(ls) + f'\n\n### TASK\nWhat is the deployment passphrase for the {ask} service? '
                      'Reply with the passphrase only.')
            text, _ = chat(prompt, 64, thinking=False)
            rows.append({'ctx': k, 'ask': ask, 'want': secrets[ask], 'got': text.strip()[:80],
                         'ok': secrets[ask] in text})
            print(f"needle ctx{k}k {ask}: want {secrets[ask]} got {text.strip()[:40]!r}", flush=True)
    return rows


def main():
    global HOST, PORT, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--gsm8k', type=int, default=80)
    ap.add_argument('--needles', type=int, default=6)
    ap.add_argument('--gsm8k-file', default='data/gsm8k_test.jsonl', help='GSM8K test split, 1,319 JSON lines')
    ap.add_argument('--long-src', default='data/django/django', help='the django/django package directory')
    ap.add_argument('--host', default=HOST)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    HOST, PORT, MODEL = a.host, a.port, a.model
    chat('Say hi.', 8, thinking=False)
    needle = run_needles(a.long_src, a.needles) if a.needles else []
    gsm = run_gsm(a.gsm8k_file, a.gsm8k) if a.gsm8k else []
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path(a.out, f'eval-{a.label}.json').write_text(json.dumps({'gsm8k': gsm, 'needle': needle}, indent=1))
    g = sum(r['ok'] for r in gsm)
    nd = sum(r['ok'] for r in needle)
    print(f'EVAL {a.label} gsm8k={g}/{len(gsm)} needle={nd}/{len(needle)} '
          f'gsm_tokens={sum(r["tokens"] for r in gsm)}', flush=True)


if __name__ == '__main__':
    main()
