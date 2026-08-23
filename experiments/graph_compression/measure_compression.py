"""
Measure the compact serialization against the REAL Qwen3-8B tokenizer.

Compares, per row:
  RAW       = row['assembly'] verbatim (what we currently feed/truncate)
  CLEAN     = assembly with GDB header + per-line addresses stripped (no grammar)
  COMPACT   = serialize_compact(..., value_slots=False)   [representative]
  COMPACT+VS= serialize_compact(..., value_slots=True)    [SSA-proxy renaming]

Metrics: real Qwen-BPE tokens, custom-tokenizer floor (whitespace+comma field
count), tok/instruction, fit rates at 3K/6K/9K/12K. Stratified <200 vs >=200.
"""
import json, os, sys, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serialize_compact import (serialize_compact, clean_assembly, instr_count,
                               clean_sp, serialize_ccfg)

QWEN_TOK = r'C:\Users\Raafat Abualazm\.cache\huggingface\hub\models--Qwen--Qwen3-8B\snapshots\b968826d9c46dd6066d109eabc6255188de91218\tokenizer.json'
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'testing', 'grpo_data_graphv2.jsonl')
BUDGETS = [3072, 6144, 9216, 12288]


def custom_floor(text):
    """Optimistic custom-tokenizer floor: one token per whitespace/comma field
    (each opcode, operand, memref, and structural marker = one vocab entry)."""
    n = 0
    for ln in text.splitlines():
        for field in ln.replace(',', ' ').split():
            n += 1
    return n


def load_tok():
    import warnings; warnings.filterwarnings('ignore')
    from tokenizers import Tokenizer
    return Tokenizer.from_file(QWEN_TOK)


def qwen_count(tok, text):
    # chunk to avoid any internal length caps; encode is fine for these sizes
    return len(tok.encode(text).ids)


def stratum_report(name, rows_metrics, key):
    """rows_metrics: list of dicts with keys qwen/floor/instr for one variant."""
    q = [m['qwen'] for m in rows_metrics]
    f = [m['floor'] for m in rows_metrics]
    instr = [m['instr'] for m in rows_metrics]
    tot_instr = sum(instr) or 1
    def fit(vals, b): return 100.0 * sum(1 for v in vals if v <= b) / len(vals)
    line = '  %-11s n=%-3d  mean_instr=%-4d | QWEN mean=%-6d max=%-6d tok/instr=%-4.1f  fit%s' % (
        name, len(rows_metrics), int(st.mean(instr)),
        int(st.mean(q)), max(q), sum(q) / tot_instr,
        ' '.join('%d:%d%%' % (b, fit(q, b)) for b in BUDGETS))
    print(line)
    line2 = '  %-11s %28s | FLOOR mean=%-6d max=%-6d tok/instr=%-4.1f  fit%s' % (
        '', '',
        int(st.mean(f)), max(f), sum(f) / tot_instr,
        ' '.join('%d:%d%%' % (b, fit(f, b)) for b in BUDGETS))
    print(line2)


def main():
    tok = load_tok()
    with open(DATA, encoding='utf-8') as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    # keep only rows with a cfg (the graph path)
    rows = [r for r in rows if r.get('cfg')]

    variants = {
        'RAW': lambda r: r.get('assembly') or '',
        'CLEAN': clean_assembly,
        'CLEANSP': clean_sp,
        'CCFG': lambda r: serialize_ccfg(r, macros=False),
        'CCFG+M': lambda r: serialize_ccfg(r, macros=True),
        'COMPACT': lambda r: serialize_compact(r, with_dataflow=True, value_slots=False),
        'COMPACT+VS': lambda r: serialize_compact(r, with_dataflow=True, value_slots=True),
        'COMPACT-noDF': lambda r: serialize_compact(r, with_dataflow=False, value_slots=False),
    }

    # per-row metrics per variant
    data = {v: {'lt200': [], 'ge200': [], 'all': []} for v in variants}
    for r in rows:
        ic = instr_count(r)
        strat = 'ge200' if ic >= 200 else 'lt200'
        for v, fn in variants.items():
            txt = fn(r)
            m = {'qwen': qwen_count(tok, txt), 'floor': custom_floor(txt), 'instr': ic, 'task': r.get('task_id')}
            data[v][strat].append(m)
            data[v]['all'].append(m)

    n_ge = len(data['RAW']['ge200']); n_lt = len(data['RAW']['lt200'])
    print('=' * 100)
    print('COMPACT SERIALIZATION vs REAL Qwen3-8B BPE   (n=%d rows: %d <200 instr, %d >=200 instr)' % (len(rows), n_lt, n_ge))
    print('data: data/testing/grpo_data_graphv2.jsonl   tokenizer: Qwen/Qwen3-8B (real)')
    print('=' * 100)
    for v in ['RAW', 'CLEAN', 'CLEANSP', 'CCFG', 'CCFG+M', 'COMPACT-noDF', 'COMPACT', 'COMPACT+VS']:
        print('\n[%s]' % v)
        for strat, label in [('lt200', '<200'), ('ge200', '>=200'), ('all', 'ALL')]:
            if data[v][strat]:
                stratum_report(label, data[v][strat], strat)

    # compression ratios vs RAW, on >=200 (the stratum that matters)
    print('\n' + '=' * 100)
    print('COMPRESSION RATIO vs RAW assembly (>=200 stratum, real Qwen-BPE mean tokens)')
    print('=' * 100)
    raw_mean = st.mean([m['qwen'] for m in data['RAW']['ge200']])
    for v in ['CLEAN', 'CLEANSP', 'CCFG', 'CCFG+M', 'COMPACT-noDF', 'COMPACT', 'COMPACT+VS']:
        vm = st.mean([m['qwen'] for m in data[v]['ge200']])
        print('  %-13s %6d tok  (%.2fx of RAW, %.0f%% smaller)' % (v, int(vm), vm / raw_mean, 100 * (1 - vm / raw_mean)))

    # worst-case function (max instr)
    print('\n' + '=' * 100)
    print('WORST-CASE FUNCTION (max instruction count)')
    print('=' * 100)
    worst = max(range(len(rows)), key=lambda i: instr_count(rows[i]))
    wr = rows[worst]; wic = instr_count(wr)
    print('  task=%s  instr=%d' % (wr.get('task_id'), wic))
    for v in ['RAW', 'CLEAN', 'CLEANSP', 'CCFG', 'CCFG+M', 'COMPACT']:
        txt = variants[v](wr)
        print('    %-13s QWEN=%-6d (%.1f tok/instr)   FLOOR=%-6d (%.1f tok/instr)' % (
            v, qwen_count(tok, txt), qwen_count(tok, txt) / wic, custom_floor(txt), custom_floor(txt) / wic))

    # sample serialization
    print('\n' + '=' * 100)
    print('SAMPLE COMPACT SERIALIZATION (first row, first ~18 lines)')
    print('=' * 100)
    samp = serialize_ccfg(rows[0], macros=True)
    for ln in samp.splitlines()[:18]:
        print('  ' + ln)


if __name__ == '__main__':
    main()
