"""
Compact CFG+DFG program serialization ("compression algorithm").

This is the TEXT-native serializer that replaces the (dropped) graph encoder /
GNN / soft-prefix. It turns a grpo_data_graphv2 row into a linearized string
that folds CFG structure into block headers, annotates block-level dataflow,
and normalizes instructions representatively (near-lossless) so the whole
function fits the decoder context window.

Design constraints (measured against data/testing/grpo_data_graphv2.jsonl):
  - `edges` dataflow is BLOCK-level only (source/target are block ids). There is
    no per-instruction def/use in the JSONL, so we CANNOT emit true SSA value
    references. We emit block-level dataflow annotations instead.
  - Representative normalization only (the balanced kind, not the aggressive
    register->REG / imm->IMM collapse that was shown to be a mirage):
      * strip GDB header + per-line absolute addresses (pure noise)
      * collapse `QWORD PTR [r14+0x38]` -> `[r14+0x38]q` (kills Qwen fragmentation,
        loses nothing: width kept as suffix)
      * replace >=5-hex absolute code addresses (branch/call targets) with `@a`
        (redundant: the successor is already in the block header)
      * KEEP real registers, immediates, struct offsets (dataflow + semantics)
  - Optional `value_slots=True` renames registers to stable per-function slots
    (rax->v0 everywhere) as an SSA *proxy*: preserves def-use linkage, shrinks
    the custom-tokenizer vocabulary. Off by default (keeps ABI priors).
"""
import re

_SIZE_PTR = re.compile(r'\b(QWORD|DWORD|WORD|BYTE|XMMWORD|YMMWORD|TBYTE)\s+PTR\s+\[([^\]]+)\]')
_ABS_ADDR = re.compile(r'0x[0-9a-fA-F]{5,}')
# GDB comment on a branch to a self-relative offset: '<hasCloseElements+465>' -> drop
# (redundant with block successors). But KEEP a bare callee name like '<memmove>'
# (semantic: tells you what runtime function is called).
_SELF_OFF_COMMENT = re.compile(r'\s*<[^>]*\+[^>]*>')
_WS = re.compile(r'\s+')
_REG = re.compile(r'\b(r[a-z0-9]+|e[a-z]{2}|[abcd][xlh]|[sd]il|[sb]pl|xmm[0-9]+|[re]?(?:si|di|sp|bp))\b')

_SIZE_SUFFIX = {'QWORD': 'q', 'DWORD': 'd', 'WORD': 'w', 'BYTE': 'b',
                'XMMWORD': 'x', 'YMMWORD': 'y', 'TBYTE': 't'}


def normalize_instruction(ins, reg_map=None):
    """Representative, near-lossless instruction normalization.

    reg_map (optional dict) enables stable value-slot renaming (SSA proxy).
    """
    ins = _WS.sub(' ', ins.strip())
    # drop redundant self-relative branch comments '<func+off>', keep bare '<callee>'
    ins = _SELF_OFF_COMMENT.sub('', ins)
    # collapse verbose size-ptr memory operands
    ins = _SIZE_PTR.sub(lambda m: '[' + m.group(2) + ']' + _SIZE_SUFFIX.get(m.group(1), 'q'), ins)
    # absolute code addresses (branch/call targets) -> @a  (successor is in header)
    ins = _ABS_ADDR.sub('@a', ins)
    if reg_map is not None:
        def _rr(m):
            r = m.group(0)
            if r not in reg_map:
                reg_map[r] = 'v%d' % len(reg_map)
            return reg_map[r]
        ins = _REG.sub(_rr, ins)
    return ins


def serialize_compact(row, with_dataflow=True, value_slots=False):
    """Serialize one row to the compact CFG+DFG grammar. Returns a string.

    Grammar:
        B{id}>{succ,succ}         block header, CFG successors folded in
        <normalized instruction>  one per line
        ...
        DF {src>tgt,tgt; ...}     (optional) block-level dataflow summary
    """
    cfg = row.get('cfg') or []
    reg_map = {} if value_slots else None
    lines = []
    for b in cfg:
        succ = ','.join(str(s) for s in (b.get('successors') or []))
        lines.append('B%s>%s' % (b.get('id'), succ))
        for ins in (b.get('instructions') or []):
            lines.append(normalize_instruction(ins, reg_map))
    body = '\n'.join(lines)

    if with_dataflow:
        df = {}
        for e in (row.get('edges') or []):
            if e.get('edge_type') == 'dataflow':
                df.setdefault(e['source'], []).append(e['target'])
        if df:
            parts = ['%d>%s' % (s, ','.join(str(t) for t in ts)) for s, ts in sorted(df.items())]
            body += '\nDF ' + ';'.join(parts)
    return body


def clean_assembly(row):
    """Baseline B: raw assembly with only GDB noise stripped (header + per-line
    absolute addresses), NO grammar compaction. Isolates 'strip noise' from
    'compact grammar'."""
    asm = row.get('assembly') or ''
    out = []
    for ln in asm.splitlines():
        # GDB body lines look like: '   0x0000...90bfc <+0>:\tpush   rbp'
        m = re.match(r'\s*0x[0-9a-fA-F]+\s+<\+\d+>:\s*(.*)$', ln)
        if m:
            txt = _SELF_OFF_COMMENT.sub('', m.group(1).strip())
            out.append(_WS.sub(' ', txt.strip()))
    return '\n'.join(out)


def instr_count(row):
    return sum(len(b.get('instructions') or []) for b in (row.get('cfg') or []))


# --- refined variants: "compress CLEAN even further" -------------------------

_DROP_Q = re.compile(r'\]q\b')          # QWORD is the 64-bit default width
_JUMP_AT = re.compile(r'^(j\w+) @a$')   # branch to abs addr; target redundant w/ headers

# Dart AOT runtime idioms (verified frequencies on grpo_data_graphv2: 319/159/217/313
# occurrences out of 18,359 instructions). Exact-sequence macros, applied AFTER
# normalization + q-default + branch-target drop. Lossless given fixed expansion.
_MACROS = [
    (('cmp rsp,[r14+0x38]', 'jbe'), 'chk'),        # CheckStackOverflow fast path
    (('push rbp', 'mov rbp,rsp'), 'pro'),           # prologue
    (('mov rsp,rbp', 'pop rbp', 'ret'), 'epi'),     # epilogue
    (('call [r14+0x238]', 'jmp'), 'chkslow'),       # stack-overflow slow path stub
]


def _refine_line(ins):
    """q-default width + drop redundant branch targets (headers carry successors;
    successors[0] is the TAKEN target — verified 1776/1776 conditional blocks)."""
    ins = _DROP_Q.sub(']', ins)
    return _JUMP_AT.sub(r'\1', ins)


def _apply_macros(lines):
    out, i = [], 0
    while i < len(lines):
        hit = None
        for seq, name in _MACROS:
            if tuple(lines[i:i + len(seq)]) == seq:
                hit = (name, len(seq))
                break
        if hit:
            out.append(hit[0]); i += hit[1]
        else:
            out.append(lines[i]); i += 1
    return out


def clean_sp(row):
    """CLEAN + size-ptr collapse + q-default. NO headers, so branch targets are
    KEPT (they are the only control-flow info in a linear listing)."""
    out = []
    for ln in clean_assembly(row).splitlines():
        ln = _SIZE_PTR.sub(lambda m: '[' + m.group(2) + ']' + _SIZE_SUFFIX.get(m.group(1), 'q'), ln)
        out.append(_DROP_Q.sub(']', ln))
    return '\n'.join(out)


def serialize_ccfg(row, macros=False):
    """CLEAN + compact CFG: block headers (successors folded, taken-target first),
    size-ptr collapse, q-default, branch targets dropped, optional runtime macros."""
    lines = []
    for b in (row.get('cfg') or []):
        succ = ','.join(str(s) for s in (b.get('successors') or []))
        body = [_refine_line(normalize_instruction(i)) for i in (b.get('instructions') or [])]
        if macros:
            body = _apply_macros(body)
        lines.append('B%s>%s' % (b.get('id'), succ))
        lines.extend(body)
    return '\n'.join(lines)
