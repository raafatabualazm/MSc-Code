# -*- coding: utf-8 -*-
"""Entropy-collapse check for any T5Gemma-2 pass@k score file.

Run this the moment a new arm's score lands, BEFORE reading pass@10 as a result.
RS-SFT and GRPO both sharpen the policy; pass@10 is a tail statistic. The collapse
shows up in distinct-candidate count per task before it costs you solved tasks.

Usage:
    python check_collapse.py <score_full175.json> [<baseline_score.json>]

Reads only sealed score files. No GPU, no API, no writes.
"""
import io
import json
import sys
from math import comb

# Guardrails fixed in RS_SFT_FOLD_PREREGISTRATION.md section 3.
BASELINE_DISTINCT = 10.00   # typed-contract 2-epoch SFT, optstep348
MAX_DROP = 0.10             # promotion bar
COLLAPSE_FLOOR = 9.50       # report as collapsed below this


def per_task(path):
    d = json.load(io.open(path, encoding='utf-8'))
    o = {}
    for c in d['candidate_results']:
        t = o.setdefault(c['task_id'], {'sha': [], 'comp': 0, 'pass': 0, 'pass0': 0})
        t['sha'].append(c['code_sha256'])
        if c['compiled']:
            t['comp'] += 1
        if c['passed']:
            t['pass'] += 1
            if c['sample_index'] == 0:
                t['pass0'] = 1
    return o


def mcnemar(gain, loss):
    n = gain + loss
    if n == 0:
        return 1.0, 1.0
    k = min(gain, loss)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return p, min(1.0, 2.0 / 2.0 ** n)


def summarise(name, per):
    m = len(per)
    dis = [len(set(v['sha'])) for v in per.values()]
    dmean = sum(dis) / m
    r = {
        'name': name, 'm': m,
        'pass1': sum(1 for v in per.values() if v['pass0']),
        'passk': sum(1 for v in per.values() if v['pass'] > 0),
        'compk': sum(1 for v in per.values() if v['comp'] > 0),
        'distinct': dmean,
        'below10': sum(1 for x in dis if x < 10),
        'hist': [sum(1 for x in dis if x == i) for i in range(1, 11)],
        'solved_counts': sorted(v['pass'] for v in per.values() if v['pass'] > 0),
    }
    return r


def show(r):
    print('  tasks              : %d' % r['m'])
    print('  pass@1             : %d' % r['pass1'])
    print('  pass@10            : %d' % r['passk'])
    print('  compile@10         : %d' % r['compk'])
    print('  distinct/10 (mean) : %.2f      tasks below 10 distinct: %d'
          % (r['distinct'], r['below10']))
    print('  distinct histogram : %s' % ' '.join(
        '%d:%d' % (i + 1, v) for i, v in enumerate(r['hist']) if v))
    print('  successes/solved   : %s' % r['solved_counts'])


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cur = summarise(sys.argv[1], per_task(sys.argv[1]))
    print('=' * 78)
    print('ARM: %s' % cur['name'])
    show(cur)

    print()
    print('=' * 78)
    print('COLLAPSE VERDICT (bars pre-registered before results)')
    drop = BASELINE_DISTINCT - cur['distinct']
    print('  distinct/10 %.2f vs baseline %.2f  ->  drop %.2f (bar %.2f)'
          % (cur['distinct'], BASELINE_DISTINCT, drop, MAX_DROP))
    if cur['distinct'] < COLLAPSE_FLOOR:
        print('  *** COLLAPSED - report as collapsed regardless of other metrics ***')
    elif drop > MAX_DROP:
        print('  *** FAILS the promotion bar on diversity ***')
    else:
        print('  diversity within bar')

    if len(sys.argv) >= 3:
        base = per_task(sys.argv[2])
        b = summarise(sys.argv[2], base)
        print()
        print('=' * 78)
        print('PAIRED vs %s' % b['name'])
        print('  baseline: pass@1 %d  pass@10 %d  compile@10 %d  distinct %.2f'
              % (b['pass1'], b['passk'], b['compk'], b['distinct']))
        cp = per_task(sys.argv[1])
        tasks = sorted(set(cp) & set(base))
        for label, fn in (('pass@10', lambda d: d['pass'] > 0),
                          ('compile@10', lambda d: d['comp'] > 0),
                          ('pass@1', lambda d: d['pass0'] > 0)):
            g = sum(1 for t in tasks if fn(cp[t]) and not fn(base[t]))
            l = sum(1 for t in tasks if fn(base[t]) and not fn(cp[t]))
            p, minp = mcnemar(g, l)
            print('  %-11s gain=%2d loss=%2d disc=%2d  exact p=%.4f  min attainable p=%.4f'
                  % (label, g, l, g + l, p, minp))

        # The divergence tell: secondary metrics up while the primary falls.
        up = ((cp and sum(1 for t in tasks if cp[t]['comp'] > 0)
               >= sum(1 for t in tasks if base[t]['comp'] > 0))
              and (sum(1 for t in tasks if cp[t]['pass0'])
                   >= sum(1 for t in tasks if base[t]['pass0'])))
        down = (sum(1 for t in tasks if cp[t]['pass'] > 0)
                < sum(1 for t in tasks if base[t]['pass'] > 0))
        print()
        if up and down:
            print('  *** DIVERGENCE: compile@10 and pass@1 flat-or-up while pass@10 falls.')
            print('      This is the entropy-collapse signature, not progress.')
        else:
            print('  no divergence signature')
    return 0


if __name__ == '__main__':
    sys.exit(main())
