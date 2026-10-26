#!/usr/bin/env python3
"""Build the draft-vocabulary keep list for the pruned MTP draft head.

Keep = top-N tokens by frequency on a generic corpus (Python stdlib and site-packages sources,
docstrings and READMEs, none of them benchmark prompts) + the R lowest BPE ranks (merge order
follows tokenizer-training frequency, which covers frequent multilingual pieces) + every added
or special token. Coverage is then reported on held-out text (model outputs saved by bench.py).

Usage (inside the vLLM container): build_keep.py TOKENIZER_DIR OUT.pt N R [heldout.json ...]
"""
import collections
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


def corpus_files():
    roots = [Path('/usr/lib/python3.12'), Path('/usr/local/lib/python3.12/dist-packages')]
    for root in roots:
        for pat in ('*.py', '*.md', '*.rst', '*.txt'):
            yield from root.rglob(pat)


def main():
    tok_dir, out, n, r = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    heldout = sys.argv[5:]
    tok = AutoTokenizer.from_pretrained(tok_dir)
    vocab = len(tok)
    counts = collections.Counter()
    budget = 400_000_000  # characters
    seen = 0
    for p in corpus_files():
        if seen >= budget:
            break
        try:
            t = p.read_text(errors='ignore')
        except OSError:
            continue
        if not t or len(t) > 2_000_000:
            continue
        seen += len(t)
        counts.update(tok.encode(t, add_special_tokens=False))
    keep = {i for i, _ in counts.most_common(n)}
    keep |= set(range(min(r, vocab)))
    keep |= set(tok.all_special_ids)
    keep |= {i for i in tok.added_tokens_decoder}
    ids = torch.tensor(sorted(keep), dtype=torch.long)
    torch.save(ids, out)
    print(f'corpus chars={seen:,} distinct={len(counts):,} keep={len(ids):,} of vocab={vocab:,}')
    for h in heldout:
        rows = json.load(open(h))
        toks = []
        for row in rows:
            toks += tok.encode((row.get('reasoning') or '') + (row.get('text') or ''), add_special_tokens=False)
        cov = sum(t in keep for t in toks) / max(1, len(toks))
        print(f'coverage {Path(h).name}: {cov:.4%} of {len(toks):,} tokens')


if __name__ == '__main__':
    main()
