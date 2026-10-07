#!/usr/bin/env python3
"""Independent, fail-closed audit for the scrubbed master release."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import build_scrubbed_dataset as core


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def load(path:Path)->list[dict]:
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(directory:Path, dart:Path, aot:Path, *, skip_runtime:bool=False)->dict:
    pub_path=directory/'master_dart_graphv2_signature_scrubbed_public.jsonl'
    pri_path=directory/'master_dart_graphv2_signature_scrubbed_private.jsonl'
    led_path=directory/'master_dart_graphv2_compile_ledger.jsonl'
    quarantine_path=directory/'master_dart_graphv2_quarantine.jsonl'
    pub,pri,led,quarantine=map(load,[pub_path,pri_path,led_path,quarantine_path])
    problems=[];graphs={'blocks':0,'cfg_edges':0,'dfg_edges':0}
    expected_public=set(core.PUBLIC_MODEL_FIELDS)
    expected_private=expected_public|{'dart_source','evaluation_only_dart_function_signature','tests'}
    if not(len(pub)==len(pri)==len(led)):
        problems.append('count_mismatch')
    source_hashes=set();assembly_hashes=set();main_count=0
    for index,(public,private,ledger) in enumerate(zip(pub,pri,led)):
        if set(public)!=expected_public:
            problems.append(['public_allowlist',index,sorted(set(public)^expected_public)])
        if set(private)!=expected_private:
            problems.append(['private_allowlist',index,sorted(set(private)^expected_private)])
        if any(public.get(key)!=private.get(key) for key in core.PUBLIC_MODEL_FIELDS):
            problems.append(['public_private_input_mismatch',index])
        if public.get('function')!='candidate' or public.get('dart_function_signature')!='':
            problems.append(['identity',index])
        assembly=str(public.get('assembly') or '')
        if 'static void candidate(void)' in assembly:
            problems.append(['synthetic_signature_instruction',index])
        if 'File file:///' in assembly or re.search(r'\bsigless_[0-9a-f]+\b',assembly):
            problems.append(['model_facing_file_or_task_id',index])
        if hashlib.sha256(assembly.encode()).hexdigest()!=ledger.get('assembly_sha256'):
            problems.append(['assembly_hash',index])
        if hashlib.sha256(str(private.get('dart_source') or '').encode()).hexdigest()!=ledger.get('source_sha256'):
            problems.append(['source_hash',index])
        try:
            core.validate_cfg_mnemonics(public.get('cfg') or [])
        except Exception as exc:
            problems.append(['mnemonic',index,str(exc)])
        symbols=core.source_symbol_names(str(private.get('dart_source') or ''))
        residues=core.symbol_residue(assembly,symbols)
        if residues:
            problems.append(['user_symbol_residue',index,residues])
        if re.search(r'(?m)^\s*(?:Future\s*<\s*void\s*>|void|dynamic)?\s*main\s*\(',str(private.get('dart_source') or '')):
            main_count+=1
        canonical_source=ledger.get('canonical_source_sha256')
        canonical_assembly=ledger.get('canonical_assembly_sha256')
        if canonical_source in source_hashes:problems.append(['source_dup',index])
        if canonical_assembly in assembly_hashes:problems.append(['assembly_dup',index])
        source_hashes.add(canonical_source);assembly_hashes.add(canonical_assembly)
        block_count=len(public.get('cfg') or []);graphs['blocks']+=block_count;seen=set()
        for edge in public.get('edges') or []:
            if not(0<=edge.get('source',-1)<block_count and 0<=edge.get('target',-1)<block_count):
                problems.append(['edge_range',index,edge])
            key=(edge.get('source'),edge.get('target'),edge.get('edge_type'))
            if key in seen:problems.append(['edge_dup',index,key])
            seen.add(key)
            family='dfg_edges' if edge.get('edge_type')=='dataflow' else 'cfg_edges'
            graphs[family]+=1
    if main_count:problems.append({'top_level_main':main_count})

    for path in [pub_path,pri_path]:
        compressed=path.with_suffix('.jsonl.gz')
        if compressed.exists():
            with path.open('rb') as raw,gzip.open(compressed,'rb') as packed:
                if hashlib.sha256(raw.read()).digest()!=hashlib.sha256(packed.read()).digest():
                    problems.append(['gzip_mismatch',path.name])

    samples=[]
    if not skip_runtime:
        if not dart.exists() or not aot.exists():
            problems.append(['runtime_missing',str(dart),str(aot)])
        else:
            indices=[]
            for predicate in [lambda row:row['main_transformed'],lambda row:row['test_kind']=='dart_harness']:
                for index,row in enumerate(led):
                    if predicate(row):indices.append(index);break
            if pri:indices.append(len(pri)-1)
            for index in dict.fromkeys(indices):
                row=pri[index]
                with tempfile.TemporaryDirectory() as temp:
                    source=Path(temp)/'test.dart'
                    source.write_text("import 'dart:async';\nimport 'dart:convert';\n"+row['dart_source']+'\n'+row['tests'])
                    jit=subprocess.run([str(dart),'run',str(source)],text=True,capture_output=True,timeout=30)
                    snapshot=Path(temp)/'test.aot'
                    compile_result=subprocess.run([str(dart),'compile','aot-snapshot',str(source),'-o',str(snapshot)],text=True,capture_output=True,timeout=90)
                    run_result=subprocess.run([str(aot),str(snapshot)],text=True,capture_output=True,timeout=30) if compile_result.returncode==0 else None
                    passed=jit.returncode==0 and compile_result.returncode==0 and run_result is not None and run_result.returncode==0
                    samples.append({'index':index,'neutral_id':led[index].get('neutral_id'),'jit':jit.returncode,'aot_compile':compile_result.returncode,'aot_run':None if run_result is None else run_result.returncode,'passed':passed})
                    if not passed:problems.append(['fresh_compile_sample',index])

    reasons={}
    for row in quarantine:
        reasons[row['reason']]=reasons.get(row['reason'],0)+1
    return {
        'schema':'scrubbed-release-audit-v2',
        'counts':{'public':len(pub),'private':len(pri),'ledger':len(led),'quarantine':len(quarantine)},
        'model_allowlist':list(core.PUBLIC_MODEL_FIELDS),
        'graphs':graphs,
        'quarantine_reasons':dict(sorted(reasons.items(),key=lambda item:(-item[1],item[0]))),
        'fresh_compile_samples':samples,
        'sha256':{path.name:sha(path) for path in [pub_path,pri_path,led_path,quarantine_path]},
        'problems':problems,
        'passed':not problems,
    }


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=Path('/mnt/data/scrubbed_master_v2'))
    parser.add_argument('--dart',type=Path,default=Path('/mnt/data/dart_sdk_root/usr/lib/dart/bin/dart'))
    parser.add_argument('--aot-runtime',type=Path,default=Path('/mnt/data/dart_sdk_root/usr/lib/dart/bin/dartaotruntime'))
    parser.add_argument('--skip-runtime',action='store_true')
    args=parser.parse_args()
    result=audit(args.directory,args.dart,args.aot_runtime,skip_runtime=args.skip_runtime)
    output=args.directory/'master_dart_graphv2_independent_audit.json'
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
