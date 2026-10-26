#!/usr/bin/env python3
"""Per-context mean, sample SD and range over repeated benches, and each entry's ratio of means over a base entry.

Usage: bench_stats.py RESULTS_DIR NAME=LABEL[,LABEL...] [NAME=LABEL[,LABEL...] ...] [--base NAME]
Reads RESULTS_DIR/LABEL.summary.json (bench.py output). tok/s is loop-free where the summary records it
(`tps_noloop`), else raw. A missing label is reported and left out of its entry.
Read-only; prints one STATS line per entry and context, and one MULT line per entry when --base is given.
steps/s and accept are left out of an entry's STATS line when a summary records them as null (no speculative decoding).
"""
import json
import statistics
import sys
from pathlib import Path

CTX = ('ctx64k', 'ctx128k', 'ctx256k')


def spread(v):
    m = statistics.mean(v)
    return m, (statistics.stdev(v) if len(v) > 1 else 0.0), min(v), max(v)


def main():
    args = sys.argv[1:]
    base = None
    if '--base' in args:
        i = args.index('--base')
        base = args[i + 1]
        del args[i:i + 2]
    d, groups = Path(args[0]), {}
    for g in args[1:]:
        name, labels = g.split('=', 1)
        groups[name] = labels.split(',')
    means = {}
    for name, labels in groups.items():
        runs = []
        for label in labels:
            p = d / f'{label}.summary.json'
            if p.exists():
                runs.append(json.loads(p.read_text()))
            else:
                print(f'{name}: {label} missing')
        means[name] = {}
        for c in CTX:
            rows = [s[c] for s in runs if c in s]
            if not rows:
                continue
            m, sd, lo, hi = spread([r.get('tps_noloop', r['tps']) for r in rows])
            means[name][c] = m
            spec = ''
            if all(r.get('steps_s') is not None for r in rows):  # a no-spec summary records null steps/s and accept
                sm, ssd, _, _ = spread([r['steps_s'] for r in rows])
                am, asd, _, _ = spread([r['accept'] for r in rows])
                spec = f' steps/s {sm:.1f} sd {ssd:.1f}; accept {am:.2f} sd {asd:.2f};'
            print(f'STATS {name} {c}: n={len(rows)} tok/s {m:.1f} sd {sd:.1f} ({lo:.1f}-{hi:.1f}, cv {100 * sd / m:.1f}%);'
                  f'{spec} loops {sum(r.get("loops", 0) for r in rows)}')
    if base in means:
        for name in groups:
            if name != base:
                r = [f'{means[name][c] / means[base][c]:.3f}' for c in CTX if c in means[name] and c in means[base]]
                print(f'MULT {name} over {base} (ratio of mean loop-free tok/s, 64K / 128K / 256K): {" / ".join(r)}')


if __name__ == '__main__':
    main()
