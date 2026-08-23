#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, concurrent.futures as cf, dataclasses, gzip, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from typing import Any

ROOT = Path('/mnt/data')
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT/'master_dart_cfg_dfg/master_dart_cfg_dfg_train.jsonl'
DEFAULT_OUT = ROOT/'scrubbed_master_v2'
DART = ROOT/'dart_sdk_root/usr/lib/dart/bin/dart'
AOT = ROOT/'dart_sdk_root/usr/lib/dart/bin/dartaotruntime'
EXTRACTORS = ROOT/'prior_extractors'
if not EXTRACTORS.exists():
    # Make the released builder and its safety tests self-contained. Production
    # builds still prefer the pinned /mnt/data copy when it is present.
    EXTRACTORS = HERE/'extractors'
sys.path.insert(0, str(EXTRACTORS))
from cfg_extractor import AssemblyCFGExtractor
from dfg_extractor import build_cross_block_dfg

from tree_sitter import Language, Parser
import tree_sitter_dart
DART_LANG = Language(tree_sitter_dart.language())

SCHEMA = 'dart-signature-scrubbed-v1'
GRAPH_SCHEMA = 'antigravity-graph-v2.1'
MAX_BLOCK_INSTRS = 20
BUILD_VERSION = 'scrubbed-master-human-eval-v2'

# This is deliberately a small model-input contract. Identifiers, hashes and
# build provenance belong in the compile ledger/manifest, never in a row that a
# trainer can accidentally serialize into a prompt.
PUBLIC_MODEL_FIELDS = (
    'lang', 'function', 'camel_case_function_name', 'python_function_name',
    'dart_function_signature', 'prompt_signature_mode', 'assembly', 'cfg',
    'edges',
)

# GNU objdump mnemonics observed in Dart AOT output plus common x86-64 forms.
# Prefixes are consumed separately. Fail closed on anything outside this set:
# in v2, context-insensitive name replacement silently changed `add` to
# `local_0`, and the graph extractor then treated corrupted text as assembly.
X86_INSTRUCTION_PREFIXES = {
    'addr16', 'addr32', 'bnd', 'cs', 'data16', 'ds', 'es', 'fs', 'gs',
    'lock', 'rep', 'repe', 'repne', 'rex', 'rex.w', 'ss',
}
ALLOWED_X86_MNEMONICS = {
    'adc', 'add', 'addpd', 'addps', 'addsd', 'and', 'andnpd', 'andpd',
    'bts', 'call', 'cdq', 'cmova', 'cmovae', 'cmovb', 'cmovbe', 'cmove',
    'cmovg', 'cmovge', 'cmovl', 'cmovle', 'cmovne', 'cmp', 'comisd',
    'cqo', 'cvtsd2ss', 'cvtss2sd', 'cvtsi2sd', 'cvttsd2si', 'dec',
    'div', 'divsd', 'idiv', 'imul', 'inc', 'int3', 'ja', 'jae', 'jb',
    'jbe', 'je', 'jg', 'jge', 'jl', 'jle', 'jmp', 'jne', 'jno', 'jnp',
    'jo', 'jp', 'jrcxz', 'js', 'lea', 'leave', 'mov', 'movabs', 'movaps',
    'movd', 'movmskpd', 'movq', 'movsd', 'movss', 'movsx', 'movsxd',
    'movups', 'movzx', 'mul', 'mulpd', 'mulps', 'mulsd', 'neg', 'nop',
    'not', 'or', 'orpd', 'pop', 'push', 'pxor', 'ret', 'rol', 'ror',
    'roundsd', 'sar', 'sbb', 'seta', 'setae', 'setb', 'setbe', 'sete',
    'setg', 'setge', 'setl', 'setle', 'setne', 'shl', 'shr', 'shufps',
    'sqrtsd', 'sub', 'subsd', 'test', 'ud2', 'xadd', 'xchg', 'xor',
    'xorpd', 'xorps',
}


def sha256_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def sha256_text(text: str) -> str: return sha256_bytes(text.encode('utf-8'))
def file_sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def run(cmd:list[str], *, cwd:Path, input_text:str|None=None, timeout:float=20, env:dict|None=None):
    e=os.environ.copy(); e.update({'DART_SUPPRESS_ANALYTICS':'1','PUB_ENVIRONMENT':'chatgpt.dataset_builder'})
    if env: e.update(env)
    t=time.monotonic()
    try:
        p=subprocess.run(cmd,cwd=cwd,input=input_text,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,env=e)
        return {'ok':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'timeout':False,'elapsed_s':round(time.monotonic()-t,4)}
    except subprocess.TimeoutExpired as x:
        return {'ok':False,'returncode':None,'stdout':x.stdout or '','stderr':x.stderr or '','timeout':True,'elapsed_s':round(time.monotonic()-t,4)}
    except Exception as x:
        return {'ok':False,'returncode':None,'stdout':'','stderr':repr(x),'timeout':False,'elapsed_s':round(time.monotonic()-t,4)}

def parser_new(): return Parser(DART_LANG)

def iter_nodes(node):
    yield node
    for c in node.children: yield from iter_nodes(c)

@dataclasses.dataclass
class FuncDecl:
    name:str; sig_start:int; sig_end:int; body_end:int; full_start:int; identifier_start:int; identifier_end:int; signature:str; params:str; async_kw:bool


def functions_from_source(source:str, parser:Parser) -> list[FuncDecl]:
    b=source.encode(); root=parser.parse(b).root_node; cs=root.children; out=[]
    for i,n in enumerate(cs):
        if n.type!='function_signature': continue
        body=cs[i+1] if i+1<len(cs) and cs[i+1].type=='function_body' else None
        if body is None: continue
        ids=[c for c in n.children if c.type=='identifier']
        if not ids: continue
        idn=ids[-1]; name=b[idn.start_byte:idn.end_byte].decode('utf-8','replace')
        full=n.start_byte; j=i-1
        while j>=0 and cs[j].type=='annotation': full=cs[j].start_byte; j-=1
        sig=b[n.start_byte:n.end_byte].decode('utf-8','replace')
        pl=[c for c in n.children if c.type=='formal_parameter_list']
        params=b[pl[0].start_byte:pl[0].end_byte].decode() if pl else '()'
        bodytxt=b[body.start_byte:body.end_byte].decode('utf-8','replace')
        out.append(FuncDecl(name,n.start_byte,n.end_byte,body.end_byte,full,idn.start_byte,idn.end_byte,sig,params,bodytxt.lstrip().startswith('async')))
    return out


def apply_edits(source:str, edits:list[tuple[int,int,str]]) -> str:
    b=source.encode('utf-8')
    # remove exact duplicates and apply descending; overlapping outer removals win
    uniq=[]; seen=set()
    for e in edits:
        if e not in seen: uniq.append(e);seen.add(e)
    uniq.sort(key=lambda x:(x[0],x[1]), reverse=True)
    last_start=len(b)+1
    for s,e,r in uniq:
        if e>last_start: # overlaps an already-applied later edit
            continue
        b=b[:s]+r.encode()+b[e:]
        last_start=s
    return b.decode('utf-8','replace')


def strip_and_transform(source:str, target:str, new_name:str, *, remove_demo_main:bool=True) -> tuple[str,str,dict]:
    parser=parser_new(); b=source.encode(); tree=parser.parse(b); funcs=functions_from_source(source,parser)
    candidates=[f for f in funcs if f.name==target]
    if not candidates: raise ValueError(f'target_declaration_not_found:{target}')
    f=candidates[0]
    removed=[]
    if remove_demo_main and target!='main':
        removed=[x for x in funcs if x.name=='main']
    remove_ranges=[(x.full_start,x.body_end) for x in removed]
    def inside_removed(s,e): return any(s>=a and e<=z for a,z in remove_ranges)
    edits=[]
    for a,z in remove_ranges: edits.append((a,z,''))
    # Remove all comments: both leakage control and source normalization.
    for n in iter_nodes(tree.root_node):
        if n.type=='comment' and not inside_removed(n.start_byte,n.end_byte): edits.append((n.start_byte,n.end_byte,''))
    # Rename identifier tokens exactly matching target (declaration, recursion, tearoffs).
    for n in iter_nodes(tree.root_node):
        if n.type=='identifier' and not inside_removed(n.start_byte,n.end_byte):
            txt=b[n.start_byte:n.end_byte].decode('utf-8','replace')
            if txt==target: edits.append((n.start_byte,n.end_byte,new_name))
    # Pragmas harden symbol retention and prevent inlining.
    edits.append((f.full_start,f.full_start,"@pragma('vm:never-inline')\n@pragma('vm:entry-point')\n"))
    out=apply_edits(source,edits)
    out=re.sub(r"(?:@pragma\('vm:entry-point'\)\s*){2,}", "@pragma('vm:entry-point')\n", out)
    out=re.sub(r"(?:@pragma\('vm:never-inline'\)\s*){2,}", "@pragma('vm:never-inline')\n", out)
    out=re.sub(r'\n[ \t]*\n[ \t]*\n+', '\n\n', out).strip()+"\n"
    # A semantic target name left as a word is normally in a string literal.
    leftover=bool(re.search(rf'(?<![A-Za-z0-9_$]){re.escape(target)}(?![A-Za-z0-9_$])',out)) if target not in ('main','candidate') else False
    sig=f.signature
    # replace only declaration identifier occurrence
    rel=f.identifier_start-f.sig_start
    sig_new=sig[:rel]+new_name+sig[rel+len(target):]
    meta={'original_signature':sig,'new_signature':sig_new,'params':f.params,'async':f.async_kw,'removed_main_count':len(removed),'semantic_name_leftover':leftover}
    return out,sig_new,meta


def add_imports(source:str, imports:list[str]) -> str:
    needed=[]
    for uri in imports:
        if not re.search(rf"import\s+['\"]{re.escape(uri)}['\"]",source): needed.append(f"import '{uri}';")
    if not needed: return source
    # Library declarations must precede imports.
    m=re.match(r'\s*library\s+[^;]+;\s*',source)
    insert=m.end() if m else 0
    return source[:insert]+'\n'.join(needed)+'\n'+source[insert:]


def rewrite_rules(source:str, rules:list[dict]) -> str:
    for r in rules or []:
        if r.get('pattern') is not None: source=source.replace(str(r['pattern']),str(r.get('replacement','')))
    return source


def replace_identifier_text(text:str, old:str, new:str) -> str:
    return re.sub(rf'(?<![A-Za-z0-9_$]){re.escape(old)}(?![A-Za-z0-9_$])',new,text)


def harness_main_capture(call_expr:str, expected_stdout:str) -> str:
    expected=json.dumps(expected_stdout)
    return f'''\nFuture<void> main() async {{
  final _captured = <String>[];
  await runZoned(
    () async {{ await Future.sync(() => {call_expr}); }},
    zoneSpecification: ZoneSpecification(
      print: (self, parent, zone, line) {{ _captured.add(line); }},
    ),
  );
  final _actual = _captured.isEmpty ? '' : '${{_captured.join('\\n')}}\\n';
  const _expected = {expected};
  if (_actual != _expected) {{
    throw StateError('stdout mismatch\\nEXPECTED:\\n$_expected\\nACTUAL:\\n$_actual');
  }}
}}
'''

NORM_HELPERS = r'''
String _agNorm(dynamic v) {
  if (v == null) return 'null';
  if (v is bool || v is num || v is String) return jsonEncode(v);
  if (v is List) return '[${v.map(_agNorm).join(',')}]';
  if (v is Set) { final x=v.map(_agNorm).toList()..sort(); return '{${x.join(',')}}'; }
  if (v is Map) {
    final x=v.entries.map((e)=>'${_agNorm(e.key)}:${_agNorm(e.value)}').toList()..sort();
    return '{${x.join(',')}}';
  }
  return 'object:${v.runtimeType}:${v.toString()}';
}
Future<String> _agEval(FutureOr<dynamic> Function() body) async {
  try { final v=await Future.sync(body); return 'ok:${_agNorm(v)}'; }
  catch (e) { return 'err:${e.runtimeType}:${e.toString()}'; }
}
'''

def reference_probe(calls:list[str]) -> str:
    lines='\n'.join([f"  print(await _agEval(() => {c}));" for c in calls])
    return NORM_HELPERS+f"\nFuture<void> main() async {{\n{lines}\n}}\n"

def expected_harness(calls:list[str], expected:list[str], old_name:str, new_name:str) -> str:
    tests=[]
    for i,(c,e) in enumerate(zip(calls,expected)):
        c2=replace_identifier_text(c,old_name,new_name)
        tests.append(f"  final _v{i}=await _agEval(() => {c2});\n  if (_v{i} != {json.dumps(e)}) throw StateError('case {i} failed: $_v{i}');")
    return NORM_HELPERS+"\nFuture<void> main() async {\n"+'\n'.join(tests)+"\n}\n"


def clean_diag(s:str, temp:Path) -> str:
    s=s.replace(str(temp),'<workdir>')
    return s[-6000:]


def _append_unique(items:list[str], value:str) -> None:
    if value and value not in items:
        items.append(value)


def source_symbol_names(source:str) -> dict[str,list[str]]:
    """Return source-declared callable and type names in stable source order.

    These names are used only to sanitize disassembler *symbol annotations*.
    They must never be replaced across raw instruction text because legal Dart
    helper names such as ``add`` and ``sub`` are also ISA mnemonics.
    """
    p=parser_new(); b=source.encode(); functions=[]; types=[]
    for n in iter_nodes(p.parse(b).root_node):
        if n.type in ('function_signature','method_signature'):
            ids=[c for c in n.children if c.type=='identifier']
            if ids:
                name=b[ids[-1].start_byte:ids[-1].end_byte].decode('utf-8','replace')
                _append_unique(functions,name)
        elif n.type in {
            'class_definition', 'enum_declaration', 'extension_declaration',
            'mixin_declaration', 'type_alias',
        }:
            named=n.child_by_field_name('name')
            candidates=[named] if named is not None else [
                c for c in n.children if c.type in ('identifier','type_identifier')
            ]
            if candidates:
                name=b[candidates[0].start_byte:candidates[0].end_byte].decode('utf-8','replace')
                _append_unique(types,name)
    return {'functions':functions,'types':types}


def source_function_names(source:str) -> list[str]:
    """Compatibility wrapper for callers that only need callable names."""
    return source_symbol_names(source)['functions']


def _neutral_symbol_maps(source_symbols:dict[str,list[str]]|list[str], target_symbol:str):
    if isinstance(source_symbols,list):
        source_symbols={'functions':source_symbols,'types':[]}
    type_map={}
    for name in source_symbols.get('types',[]):
        if name not in (target_symbol,'candidate','main') and len(name)>=2 and name not in type_map:
            type_map[name]=f'type_{len(type_map)}'
    function_map={}
    for name in source_symbols.get('functions',[]):
        if name in (target_symbol,'candidate','main') or name in type_map or len(name)<2:
            continue
        if name not in function_map:
            function_map[name]=f'local_{len(function_map)}'
    return type_map,function_map


def scrub_objdump_instruction(
    instruction:str,
    target_symbol:str,
    source_symbols:dict[str,list[str]]|list[str],
) -> str:
    """Normalize an objdump instruction without ever rewriting the opcode.

    Source names occur in ``<symbol annotations>``. Restricting substitutions
    to that grammar boundary both prevents semantic-name leakage and preserves
    machine instructions whose mnemonic happens to equal a Dart helper name.
    """
    instruction=re.sub(
        r'(?<=\s)([0-9a-fA-F]{4,})(?=\s+<)',
        lambda m:'0x'+m.group(1), instruction,
    )
    type_map,function_map=_neutral_symbol_maps(source_symbols,target_symbol)

    def scrub_label(match:re.Match) -> str:
        label=replace_identifier_text(match.group(1),target_symbol,'candidate')
        for old,new in type_map.items():
            label=replace_identifier_text(label,old,new)
        for old,new in function_map.items():
            label=replace_identifier_text(label,old,new)
        return '<'+label+'>'

    scrubbed=re.sub(r'<([^<>]*)>',scrub_label,instruction)
    validate_instruction_mnemonic(scrubbed)
    return scrubbed


def instruction_mnemonic(instruction:str) -> str:
    fields=instruction.strip().lower().split()
    while fields and fields[0] in X86_INSTRUCTION_PREFIXES:
        fields.pop(0)
    return fields[0] if fields else ''


def validate_instruction_mnemonic(instruction:str) -> None:
    mnemonic=instruction_mnemonic(instruction)
    if mnemonic not in ALLOWED_X86_MNEMONICS:
        raise ValueError(f'unknown_or_corrupted_mnemonic:{mnemonic}:{instruction}')


def validate_cfg_mnemonics(cfg:list[dict]) -> None:
    for block in cfg:
        for instruction in block.get('instructions') or []:
            validate_instruction_mnemonic(str(instruction))


def symbol_residue(assembly:str, source_symbols:dict[str,list[str]]|list[str]) -> list[str]:
    """Find source-declared identifiers remaining inside symbol annotations."""
    if isinstance(source_symbols,list):
        names=source_symbols
    else:
        names=list(source_symbols.get('types',[]))+list(source_symbols.get('functions',[]))
    labels='\n'.join(re.findall(r'<([^<>]*)>',assembly))
    return sorted({
        name for name in names
        if name not in ('candidate','main') and len(name)>=2
        and re.search(rf'(?<![A-Za-z0-9_$]){re.escape(name)}(?![A-Za-z0-9_$])',labels)
    })


def model_facing_row(common:dict) -> dict:
    """Construct the only top-level fields permitted in model-facing JSONL."""
    missing=[key for key in PUBLIC_MODEL_FIELDS if key not in common]
    if missing:
        raise ValueError(f'missing_model_fields:{missing}')
    return {key:common[key] for key in PUBLIC_MODEL_FIELDS}


def format_scrubbed_assembly(
    instructions:list[tuple[int,str]],
    target_symbol:str,
    source_symbols:dict[str,list[str]]|list[str],
) -> str:
    if not instructions:
        raise ValueError('no_disassembled_instructions')
    base=instructions[0][0]
    lines=['All functions matching regular expression "candidate":','',
           'Dump of assembler code for function candidate:']
    for address,instruction in instructions:
        scrubbed=scrub_objdump_instruction(instruction,target_symbol,source_symbols)
        lines.append(f'   0x{address:016x} <+{address-base}>:\t{scrubbed}')
    lines.append('End of assembler dump.')
    assembly='\n'.join(lines)+'\n'
    residues=symbol_residue(assembly,source_symbols)
    if residues:
        raise ValueError(f'user_symbol_residue:{residues}')
    return assembly


def extract_symbol_assembly(
    aot_path:Path,
    symbol:str,
    neutral_file:str,
    source_symbols:dict[str,list[str]]|list[str],
    *,
    readelf:str='readelf',
    objdump:str='objdump',
) -> tuple[str,list[str]]:
    rr=run([readelf,'-sW',str(aot_path)],cwd=aot_path.parent,timeout=20)
    if not rr['ok']: raise RuntimeError('readelf_failed:'+rr['stderr'])
    syms=[]
    for line in rr['stdout'].splitlines():
        parts=line.split()
        if len(parts)>=8 and parts[3]=='FUNC' and parts[-1]==symbol:
            try: syms.append((int(parts[1],16),int(parts[2]),parts[-1]))
            except Exception: pass
    if not syms: raise RuntimeError('candidate_symbol_not_found')
    # Largest is optimized body; ties prefer lower address.
    syms.sort(key=lambda x:(-x[1],x[0])); address,size,_=syms[0]
    stop=address+max(size,1)
    od=run([objdump,'-d','-Mintel','--no-show-raw-insn',f'--start-address=0x{address:x}',f'--stop-address=0x{stop:x}',str(aot_path)],cwd=aot_path.parent,timeout=30)
    if not od['ok']: raise RuntimeError('objdump_failed:'+od['stderr'])
    instr=[]
    rx=re.compile(r'^\s*([0-9a-fA-F]+):\s+(.+?)\s*$')
    for line in od['stdout'].splitlines():
        m=rx.match(line)
        if not m: continue
        addr=int(m.group(1),16); ins=m.group(2)
        if ins.startswith('.byte') or ins.startswith('(bad)'): continue
        instr.append((addr,ins))
    if not instr: raise RuntimeError('no_disassembled_instructions')
    assembly=format_scrubbed_assembly(instr,symbol,source_symbols)
    return assembly,[f'0x{address:x}']


def graph_from_assembly(assembly:str, entry_addresses:list[str]):
    old=os.environ.get('GRAPH_MAX_BLOCK_INSTRS'); os.environ['GRAPH_MAX_BLOCK_INSTRS']=str(MAX_BLOCK_INSTRS)
    try:
        blocks, control, integ=AssemblyCFGExtractor(assembly).build_blocks()
    finally:
        if old is None: os.environ.pop('GRAPH_MAX_BLOCK_INSTRS',None)
        else: os.environ['GRAPH_MAX_BLOCK_INSTRS']=old
    cfg=[]
    for b in blocks:
        d=dataclasses.asdict(b)
        cfg.append(d)
    ce=[dataclasses.asdict(e) for e in control]
    dfg=build_cross_block_dfg(cfg,ce,max_edges=100000)
    edges=ce+dfg
    n=len(cfg)
    all_range=all(isinstance(e.get('source'),int) and isinstance(e.get('target'),int) and 0<=e['source']<n and 0<=e['target']<n for e in edges)
    integrity={
      'isolated_nodes': integ.get('isolated_nodes',[]),
      'isolated_nonentry_nodes':[x for x in integ.get('isolated_nodes',[]) if x!=0],
      'unreachable_nodes':integ.get('unreachable_nodes',[]),
      'entry_nodes':[0] if cfg else [],'valid':bool(cfg and all_range and all(b.get('instructions') for b in cfg)),
      'pruned_unreachable_block_count':0,'pruned_unreachable_block_starts':[],
      'networkx_available':integ.get('valid') is not None,
      'entry_address':entry_addresses[0] if entry_addresses else None,
      'entry_addresses':entry_addresses,'entry_block':0 if cfg else None,'entry_blocks':[0] if cfg else [],
      'requested_entry_address_count':len(entry_addresses),'resolved_entry_address_count':len(entry_addresses) if cfg else 0,
      'unresolved_entry_addresses':[],'has_entry':bool(cfg),'all_edges_in_range':all_range,
      'all_blocks_nonempty':all(bool(b.get('instructions')) for b in cfg),
      'parsed_instruction_count':sum(len(b.get('instructions') or []) for b in cfg),
      'candidate_line_count':sum(len(b.get('instructions') or []) for b in cfg),
      'rejected_symbol_line_count':0,'rejected_non_instruction_count':0,'duplicate_address_count':0,
      'unresolved_direct_branch_count':0,'external_direct_branch_count':0,'indirect_branch_count':0,
      'internal_direct_call_count':0,'external_direct_call_count':0,'indirect_call_count':0,
      'unknown_branch_mnemonics':[],'graph_schema_version':GRAPH_SCHEMA,
      'cfg_edge_count':len(ce),'dataflow_edge_count':len(dfg),'max_block_instrs':MAX_BLOCK_INSTRS,
      'max_dataflow_edges':0,'symbol_entry_addresses':entry_addresses,
    }
    if not integrity['valid']: raise RuntimeError('invalid_graph')
    return cfg,edges,integrity


def canonical_source_hash(source:str)->str:
    # Source already comment-scrubbed. Ignore whitespace and neutral/unique name.
    s=re.sub(r'\s+','',source)
    s=re.sub(r'c_[0-9a-f]{12}','candidate',s)
    return sha256_text(s)

def canonical_asm_hash(asm:str)->str:
    s=asm.lower(); s=re.sub(r'file:///[^\n]+','file:///<neutral>',s);s=re.sub(r'0x[0-9a-f]+','<addr>',s);s=re.sub(r'<\+\d+>','<+off>',s);s=re.sub(r'\s+',' ',s)
    return sha256_text(s)


def process_row(item:tuple[int,dict], input_sha:str, dart_version:str) -> dict:
    idx,row=item; t0=time.monotonic(); rid=str(row.get('id',idx)); target=str(row.get('function') or 'main'); source=str(row.get('dart_source') or '')
    unique='c_'+sha256_text(rid)[:12]
    rules=row.get('tests',{}).get('rewrite_rules') or []
    source=rewrite_rules(source,rules)
    temp=Path(tempfile.mkdtemp(prefix=f'ag_scrub_{idx:05d}_',dir='/tmp'))
    try:
        if not source.strip(): raise ValueError('empty_source')
        if re.search(r'^\s*(part\s+of|part\s+)',source,re.M): raise ValueError('part_directive_unsupported')
        # Strict HumanEval conversion cannot directly unit-test process-global IO.
        if target=='main' and re.search(r'\b(stdin|stdout|stderr)\b|\bexit\s*\(|\bProcess\s*\.',source):
            raise ValueError('main_requires_process_global_io')
        compile_source, unique_sig, meta=strip_and_transform(source,target,unique,remove_demo_main=True)
        if meta['semantic_name_leftover']: raise ValueError('semantic_function_name_in_literal')
        neutral_source=replace_identifier_text(compile_source,unique,'candidate')
        neutral_sig=replace_identifier_text(unique_sig,unique,'candidate')
        source_symbols=source_symbol_names(compile_source)
        tests=row.get('tests') or {}; kind=tests.get('kind')
        neutral_harness=''; unique_harness=''; oracle={}
        if kind=='dart_harness':
            h=str(tests.get('harness') or '')
            if not h.strip(): raise ValueError('empty_provided_harness')
            unique_harness=replace_identifier_text(h,target,unique)
            neutral_harness=replace_identifier_text(h,target,'candidate')
        elif kind=='differential_function':
            calls=[str(c.get('call')) for c in tests.get('cases',[]) if isinstance(c,dict) and c.get('call')]
            if not calls: raise ValueError('no_function_cases')
            # Build reference source without its demo main, keeping semantic target.
            ref_source,_,_=strip_and_transform(source,target,target,remove_demo_main=True)
            ref_program=add_imports(ref_source,['dart:async','dart:convert'])+reference_probe(calls)
            (temp/'reference.dart').write_text(ref_program)
            rr=run([str(DART),'run','reference.dart'],cwd=temp,timeout=12)
            if not rr['ok']: raise RuntimeError('reference_oracle_failed:'+clean_diag(rr['stderr'],temp))
            expected=rr['stdout'].replace('\r\n','\n').splitlines()
            if len(expected)!=len(calls): raise RuntimeError(f'oracle_line_count:{len(expected)}!={len(calls)}')
            unique_harness=expected_harness(calls,expected,target,unique)
            neutral_harness=expected_harness(calls,expected,target,'candidate')
            oracle={'case_count':len(calls),'expected_sha256':sha256_text('\n'.join(expected))}
        elif kind=='differential_program' or target=='main':
            # Original executable is the oracle, with the same deterministic rewrites.
            (temp/'reference.dart').write_text(source)
            rr=run([str(DART),'run','reference.dart'],cwd=temp,input_text='',timeout=12)
            if not rr['ok'] or rr['stderr']:
                raise RuntimeError('reference_program_failed:'+clean_diag(rr['stderr'] or rr['stdout'],temp))
            expected=rr['stdout'].replace('\r\n','\n')
            params=meta.get('params','()')
            call=f'{unique}(const <String>[])' if params.strip()!='()' else f'{unique}()'
            ncall='candidate(const <String>[])' if params.strip()!='()' else 'candidate()'
            unique_harness=harness_main_capture(call,expected)
            neutral_harness=harness_main_capture(ncall,expected)
            oracle={'case_count':1,'expected_stdout_sha256':sha256_text(expected)}
        else:
            raise ValueError(f'unsupported_test_kind:{kind}')

        program=add_imports(compile_source,['dart:async','dart:convert'])+'\n'+unique_harness
        (temp/'program.dart').write_text(program)
        jit=run([str(DART),'run','program.dart'],cwd=temp,timeout=15)
        if not jit['ok']: raise RuntimeError('jit_test_failed:'+clean_diag(jit['stderr'] or jit['stdout'],temp))
        aot_path=temp/'program.aot'
        comp=run([str(DART),'compile','aot-snapshot','program.dart','-o',str(aot_path)],cwd=temp,timeout=60)
        if not comp['ok']: raise RuntimeError('aot_compile_failed:'+clean_diag(comp['stderr'] or comp['stdout'],temp))
        aotr=run([str(AOT),str(aot_path)],cwd=temp,timeout=15)
        if not aotr['ok']: raise RuntimeError('aot_test_failed:'+clean_diag(aotr['stderr'] or aotr['stdout'],temp))
        neutral_id='sigless_'+sha256_text(canonical_source_hash(neutral_source)+rid)[:12]
        neutral_file=neutral_id+'.dart'
        assembly,entry=extract_symbol_assembly(aot_path,unique,neutral_file,source_symbols)
        cfg,edges,integrity=graph_from_assembly(assembly,entry)
        validate_cfg_mnemonics(cfg)
        asm_sha=sha256_text(assembly)
        graph_v2={'schema':GRAPH_SCHEMA,'assembly_sha256':asm_sha,'extractor_sha256':sha256_bytes((EXTRACTORS/'cfg_extractor.py').read_bytes()+(EXTRACTORS/'dfg_extractor.py').read_bytes()),'max_block_instrs':MAX_BLOCK_INSTRS,'max_dataflow_edges':0,'symbol_entry_addresses':entry}
        protocol={
          'schema':SCHEMA,'benchmark_kind':'training','fresh_holdout':False,'input_sha256':input_sha,'freeze_manifest_sha256':None,
          'original_source_sha256':sha256_text(source),'semantic_function_name_sha256':sha256_text(target),'neutral_target_name':'candidate',
          'prompt_exposes':['assembly_or_graph','neutral_target_name'],
          'prompt_withholds':['return_type','parameter_types','parameter_names','reference_source','tests','semantic_function_name'],
          'assembly_build':{'function':'candidate','blocks':len(cfg),'cfg_edges':sum(e.get('edge_type')!='dataflow' for e in edges),'parsed_instructions':integrity['parsed_instruction_count'],'external_direct_branches':integrity['external_direct_branch_count'],'pruned_unreachable_blocks':0,'dart_sdk':dart_version,'jit_tests':'passed','aot_compile':'passed','aot_tests':'passed','build_version':BUILD_VERSION}
        }
        common={'lang':'Dart','function':'candidate','camel_case_function_name':'candidate','python_function_name':'','dart_function_signature':'','prompt_signature_mode':'name_only','assembly':assembly,'cfg':cfg,'edges':edges}
        public=model_facing_row(common)
        private={'dart_source':neutral_source,**public,'evaluation_only_dart_function_signature':neutral_sig,'tests':neutral_harness}
        ledger={'status':'retained','source_index':idx,'source_id_sha256':sha256_text(rid),'neutral_id':neutral_id,'original_function_sha256':sha256_text(target),'test_kind':kind,'main_transformed':target=='main','removed_demo_main_count':meta['removed_main_count'],'source_sha256':sha256_text(neutral_source),'canonical_source_sha256':canonical_source_hash(neutral_source),'assembly_sha256':asm_sha,'canonical_assembly_sha256':canonical_asm_hash(assembly),'test_harness_sha256':sha256_text(neutral_harness),'oracle':oracle,'dart_sdk':dart_version,'jit':jit,'aot_compile':comp,'aot_run':aotr,'elapsed_s':round(time.monotonic()-t0,3),'provenance':{'neutral_file':neutral_file,'benchmark_protocol':protocol,'graph_v2':graph_v2,'integrity':integrity}}
        # discard stdout/stderr noise from successful ledger
        for k in ('jit','aot_compile','aot_run'):
            ledger[k]={x:ledger[k][x] for x in ('ok','returncode','timeout','elapsed_s')}
        return {'ok':True,'public':public,'private':private,'ledger':ledger}
    except Exception as e:
        return {'ok':False,'reject':{'status':'rejected','source_index':idx,'source_id_sha256':sha256_text(rid),'original_function_sha256':sha256_text(target),'reason':str(e).split(':',1)[0],'diagnostic':str(e)[-6000:],'traceback':traceback.format_exc(limit=2)[-3000:],'elapsed_s':round(time.monotonic()-t0,3)}}
    finally:
        shutil.rmtree(temp,ignore_errors=True)


def load_holdout_hashes(paths:list[Path]):
    sh=set();ah=set()
    for p in paths:
        if not p.exists(): continue
        with p.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line)
                s=r.get('dart_source')
                if isinstance(s,str) and s.strip():
                    # Remove comments and normalize, but tolerate parse failures.
                    try:
                        t,_,_=strip_and_transform(s,str(r.get('function') or 'candidate'),'candidate',remove_demo_main=True)
                    except Exception: t=s
                    sh.add(canonical_source_hash(t))
                a=r.get('assembly')
                if isinstance(a,str) and a.strip(): ah.add(canonical_asm_hash(a))
    return sh,ah


def jsonl_write(path:Path, rows:list[dict]):
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def gzip_copy(src:Path,dst:Path):
    with src.open('rb') as fi,gzip.open(dst,'wb',compresslevel=6) as fo: shutil.copyfileobj(fi,fo)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=DEFAULT_INPUT);ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if not DART.exists() or not AOT.exists(): raise SystemExit('Real Dart SDK missing')
    dv=run([str(DART),'--version'],cwd=out,timeout=10);dart_version=(dv['stderr'] or dv['stdout']).strip()
    input_sha=file_sha256(args.input)
    rows=[]
    with args.input.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if line.strip(): rows.append((i,json.loads(line)))
            if args.limit and len(rows)>=args.limit: break
    raw_results=out/'raw_results.jsonl'
    completed=set(); retained=[]; rejects=[]; ledgers=[]
    if args.resume and raw_results.exists():
        for line in raw_results.open(encoding='utf-8'):
            if not line.strip():continue
            x=json.loads(line); completed.add(x.get('source_index'))
            if x.get('ok'): retained.append((x['public'],x['private'],x['ledger']));ledgers.append(x['ledger'])
            else: rejects.append(x['reject'])
    todo=[x for x in rows if x[0] not in completed]
    mode='a' if args.resume else 'w'
    print(json.dumps({'rows':len(rows),'todo':len(todo),'workers':args.workers,'dart':dart_version}),flush=True)
    with raw_results.open(mode,encoding='utf-8') as rf, cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(process_row,item,input_sha,dart_version):item[0] for item in todo}
        done=0
        for fut in cf.as_completed(futs):
            idx=futs[fut]
            try: res=fut.result()
            except Exception as e: res={'ok':False,'reject':{'status':'rejected','source_index':idx,'reason':'worker_exception','diagnostic':repr(e)}}
            x={'source_index':idx,**res};rf.write(json.dumps(x,ensure_ascii=False)+'\n');rf.flush();os.fsync(rf.fileno())
            if res.get('ok'): retained.append((res['public'],res['private'],res['ledger']));ledgers.append(res['ledger'])
            else: rejects.append(res['reject'])
            done+=1
            if done%10==0 or done==len(todo): print(json.dumps({'completed_now':done,'remaining':len(todo)-done,'retained_total':len(retained),'rejected_total':len(rejects)}),flush=True)
    # Stable ordering and strict post-build dedup/leakage removal.
    retained.sort(key=lambda x:x[2]['source_index'])
    hold_src,hold_asm=load_holdout_hashes([ROOT/'grpo_data_graphv2_signature_scrubbed_private.jsonl',ROOT/'master_dart_cfg_dfg/master_dart_cfg_dfg_heldout.jsonl'])
    seen_src={};seen_asm={};final=[]
    for pub,priv,led in retained:
        cs=led['canonical_source_sha256'];ca=led['canonical_assembly_sha256']
        reason=None
        if cs in hold_src: reason='heldout_source_overlap'
        elif ca in hold_asm: reason='heldout_assembly_overlap'
        elif cs in seen_src: reason='duplicate_transformed_source'
        elif ca in seen_asm: reason='duplicate_transformed_assembly'
        if reason:
            rejects.append({'status':'rejected_postbuild','source_index':led['source_index'],'source_id_sha256':led['source_id_sha256'],'neutral_id':led['neutral_id'],'reason':reason,'duplicate_of':seen_src.get(cs) or seen_asm.get(ca)})
        else:
            seen_src[cs]=led['neutral_id'];seen_asm[ca]=led['neutral_id'];final.append((pub,priv,led))
    publics=[x[0] for x in final];privates=[x[1] for x in final];final_ledgers=[x[2] for x in final]
    public_path=out/'master_dart_graphv2_signature_scrubbed_public.jsonl'; private_path=out/'master_dart_graphv2_signature_scrubbed_private.jsonl'; ledger_path=out/'master_dart_graphv2_compile_ledger.jsonl'; reject_path=out/'master_dart_graphv2_quarantine.jsonl'
    jsonl_write(public_path,publics);jsonl_write(private_path,privates);jsonl_write(ledger_path,final_ledgers);jsonl_write(reject_path,sorted(rejects,key=lambda x:x.get('source_index',10**9)))
    gzip_copy(public_path,public_path.with_suffix('.jsonl.gz'));gzip_copy(private_path,private_path.with_suffix('.jsonl.gz'))
    # Audits.
    pub_bad=[]
    for i,r in enumerate(publics):
        missing=set(PUBLIC_MODEL_FIELDS)-set(r)
        extra=set(r)-set(PUBLIC_MODEL_FIELDS)
        if missing or extra: pub_bad.append([i,{'missing':sorted(missing),'extra':sorted(extra)}])
        if r.get('function')!='candidate' or r.get('dart_function_signature')!='': pub_bad.append([i,'identity'])
        if 'static void candidate(void)' in r.get('assembly','') or 'File file:///' in r.get('assembly',''):
            pub_bad.append([i,'synthetic_header_or_file_id'])
        try: validate_cfg_mnemonics(r.get('cfg') or [])
        except Exception as exc: pub_bad.append([i,str(exc)])
        residues=symbol_residue(r.get('assembly',''),source_symbol_names(privates[i]['dart_source']))
        if residues: pub_bad.append([i,{'user_symbol_residue':residues}])
    private_main_logic=sum(1 for r in privates if re.search(r'(?m)^\s*(?:Future\s*<\s*void\s*>|void|dynamic)?\s*main\s*\(',r['dart_source']))
    manifest={'schema':'scrubbed-master-build-manifest-v2','build_version':BUILD_VERSION,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'input':{'path':str(args.input),'sha256':input_sha,'rows':len(rows)},'dart_sdk':dart_version,'requirements':{'real_dart_jit':True,'real_dart_aot_compile':True,'real_dartaotruntime':True,'main_logic_moved_to_candidate':True,'public_signature_scrubbed':True,'tests_hidden_from_public':True},'counts':{'processed':len(rows),'runtime_retained_before_dedup':len(retained),'final_retained':len(final),'quarantined':len(rejects),'main_rows_retained':sum(x[2]['main_transformed'] for x in final),'provided_harness_rows':sum(x[2]['test_kind']=='dart_harness' for x in final),'generated_function_test_rows':sum(x[2]['test_kind']=='differential_function' for x in final),'generated_main_test_rows':sum(x[2]['main_transformed'] for x in final)},'audit':{'public_forbidden_field_violations':pub_bad,'private_sources_with_top_level_main':private_main_logic,'unique_canonical_sources':len(seen_src),'unique_canonical_assemblies':len(seen_asm),'heldout_source_hashes':len(hold_src),'heldout_assembly_hashes':len(hold_asm),'passed':not pub_bad and private_main_logic==0 and len(final)==len(seen_src)==len(seen_asm)},'outputs':{}}
    for p in [public_path,private_path,public_path.with_suffix('.jsonl.gz'),private_path.with_suffix('.jsonl.gz'),ledger_path,reject_path,raw_results]:
        manifest['outputs'][p.name]={'sha256':file_sha256(p),'size_bytes':p.stat().st_size}
    manpath=out/'master_dart_graphv2_scrubbed_manifest.json';manpath.write_text(json.dumps(manifest,indent=2))
    readme=out/'README.md';readme.write_text(f'''# Scrubbed HumanEval-style Dart CFG+DFG training set\n\n- Real compiler: `{dart_version}`\n- Final retained rows: **{len(final)}**\n- Quarantined rows: **{len(rejects)}**\n- `main` rows converted to callable `candidate`: **{manifest['counts']['main_rows_retained']}**\n\nThe public file is a strict model-input allowlist containing only neutral `candidate` assembly/CFG/DFG and their minimal contract fields. Identifiers, filenames, hashes, build provenance, source, tests, signatures, and expected outputs are excluded.\n\nThe private file adds the withheld neutralized source, evaluation-only signature, and Dart test harness to the same allowlisted input. Every retained row passed `dart run`, `dart compile aot-snapshot`, and `dartaotruntime` with the harness. Per-row identifiers, hashes, graph integrity, and build provenance are retained only in the compile ledger. AOT disassembly and graphs were regenerated from the rewritten callable function; old `main` assembly was never reused.\n\nRows that require process-global stdin/stdout/stderr/exit, time out, fail either compiler mode, fail tests, produce invalid graphs, duplicate another transformed row, or overlap the held-out set are listed in `master_dart_graphv2_quarantine.jsonl`.\n''')
    # checksums and bundle
    checks=out/'SHA256SUMS.txt'; files=[p for p in out.iterdir() if p.is_file() and p.name not in ('SHA256SUMS.txt','scrubbed_master_v2_bundle.zip')]
    checks.write_text(''.join(f'{file_sha256(p)}  {p.name}\n' for p in sorted(files)))
    zip_path=ROOT/'scrubbed_master_v2_bundle.zip'
    if zip_path.exists(): zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix('')),'zip',root_dir=out)
    (ROOT/'scrubbed_master_v2_bundle.zip.sha256').write_text(f'{file_sha256(zip_path)}  {zip_path.name}\n')
    print(json.dumps(manifest['counts'],indent=2));print(json.dumps(manifest['audit'],indent=2));print(zip_path)

if __name__=='__main__': main()
