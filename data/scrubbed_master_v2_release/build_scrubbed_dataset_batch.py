#!/usr/bin/env python3
from __future__ import annotations
import argparse, base64, concurrent.futures as cf, gzip, hashlib, json, os, re, shutil, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from typing import Any
sys.path.insert(0,'/mnt/data/scrubbed_master_v2')
import build_scrubbed_dataset as core

ROOT=Path('/mnt/data'); DEFAULT_INPUT=ROOT/'master_dart_cfg_dfg/master_dart_cfg_dfg_train.jsonl'; DEFAULT_OUT=ROOT/'scrubbed_master_v2'
DART=core.DART; AOT=core.AOT


def prepare_row(item:tuple[int,dict], input_sha:str, dart_version:str)->dict:
    idx,row=item; t0=time.monotonic(); rid=str(row.get('id',idx)); target=str(row.get('function') or 'main'); source=str(row.get('dart_source') or '')
    unique='c_'+core.sha256_text(rid)[:12]; temp=Path(tempfile.mkdtemp(prefix=f'ag_prep_{idx:05d}_',dir='/tmp'))
    try:
        source=core.rewrite_rules(source,(row.get('tests') or {}).get('rewrite_rules') or [])
        if not source.strip(): raise ValueError('empty_source')
        if re.search(r'^\s*(part\s+of|part\s+)',source,re.M): raise ValueError('part_directive_unsupported')
        if target=='main' and re.search(r'\b(stdin|stdout|stderr)\b|\bexit\s*\(|\bProcess\s*\.',source): raise ValueError('main_requires_process_global_io')
        compile_source,unique_sig,meta=core.strip_and_transform(source,target,unique,remove_demo_main=True)
        if meta['semantic_name_leftover']: raise ValueError('semantic_function_name_in_literal')
        neutral_source=core.replace_identifier_text(compile_source,unique,'candidate')
        neutral_sig=core.replace_identifier_text(unique_sig,unique,'candidate')
        source_symbols=core.source_symbol_names(compile_source)
        tests=row.get('tests') or {}; kind=tests.get('kind'); oracle={}
        if kind=='dart_harness':
            h=str(tests.get('harness') or '')
            if not h.strip(): raise ValueError('empty_provided_harness')
            uh=core.replace_identifier_text(h,target,unique); nh=core.replace_identifier_text(h,target,'candidate')
        elif kind=='differential_function':
            calls=[str(c.get('call')) for c in tests.get('cases',[]) if isinstance(c,dict) and c.get('call')]
            if not calls: raise ValueError('no_function_cases')
            ref_source,_,_=core.strip_and_transform(source,target,target,remove_demo_main=True)
            ref_program=core.add_imports(ref_source,['dart:async','dart:convert'])+core.reference_probe(calls)
            (temp/'reference.dart').write_text(ref_program)
            rr=core.run([str(DART),'run','reference.dart'],cwd=temp,timeout=15)
            if not rr['ok']: raise RuntimeError('reference_oracle_failed:'+core.clean_diag(rr['stderr'] or rr['stdout'],temp))
            expected=rr['stdout'].replace('\r\n','\n').splitlines()
            if len(expected)!=len(calls): raise RuntimeError(f'oracle_line_count:{len(expected)}!={len(calls)}')
            uh=core.expected_harness(calls,expected,target,unique);nh=core.expected_harness(calls,expected,target,'candidate')
            oracle={'case_count':len(calls),'expected_sha256':core.sha256_text('\n'.join(expected))}
        elif kind=='differential_program' or target=='main':
            (temp/'reference.dart').write_text(source)
            rr=core.run([str(DART),'run','reference.dart'],cwd=temp,input_text='',timeout=15)
            if not rr['ok'] or rr['stderr']: raise RuntimeError('reference_program_failed:'+core.clean_diag(rr['stderr'] or rr['stdout'],temp))
            expected=rr['stdout'].replace('\r\n','\n');params=meta.get('params','()')
            call=f'{unique}(const <String>[])' if params.strip()!='()' else f'{unique}()';ncall='candidate(const <String>[])' if params.strip()!='()' else 'candidate()'
            uh=core.harness_main_capture(call,expected);nh=core.harness_main_capture(ncall,expected)
            oracle={'case_count':1,'expected_stdout_sha256':core.sha256_text(expected)}
        else: raise ValueError(f'unsupported_test_kind:{kind}')
        program=core.add_imports(compile_source,['dart:async','dart:convert'])+'\n'+uh
        neutral_id='sigless_'+core.sha256_text(core.canonical_source_hash(neutral_source)+rid)[:12]
        return {'ok':True,'source_index':idx,'source_id_sha256':core.sha256_text(rid),'target_sha256':core.sha256_text(target),'target':target,'unique':unique,'neutral_id':neutral_id,'neutral_file':neutral_id+'.dart','compile_program':program,'neutral_source':neutral_source,'neutral_signature':neutral_sig,'neutral_harness':nh,'source_symbols':source_symbols,'test_kind':kind,'main_transformed':target=='main','removed_demo_main_count':meta['removed_main_count'],'original_source_sha256':core.sha256_text(source),'canonical_source_sha256':core.canonical_source_hash(neutral_source),'oracle':oracle,'prepare_elapsed_s':round(time.monotonic()-t0,3)}
    except Exception as e:
        return {'ok':False,'source_index':idx,'reject':{'status':'rejected','source_index':idx,'source_id_sha256':core.sha256_text(rid),'original_function_sha256':core.sha256_text(target),'reason':str(e).split(':',1)[0],'diagnostic':str(e)[-6000:],'traceback':traceback.format_exc(limit=2)[-2500:],'elapsed_s':round(time.monotonic()-t0,3)}}
    finally: shutil.rmtree(temp,ignore_errors=True)


def root_program(items:list[dict])->str:
    im=[];calls=[]
    for j,x in enumerate(items):
        alias=f'r{j}'; fn=Path(x['lib_path']).name
        im.append(f"import '{fn}' as {alias};")
        calls.append(f"  try {{ await Future.sync(() => {alias}.main()); print('AG|{x['source_index']}|PASS'); }} catch (e, st) {{ print('AG|{x['source_index']}|FAIL|${{base64Url.encode(utf8.encode('$e\\n$st'))}}'); }}")
    return "import 'dart:async';\nimport 'dart:convert';\n"+'\n'.join(im)+"\nFuture<void> main() async {\n"+'\n'.join(calls)+"\n}\n"

def statuses(output:str)->dict[int,tuple[str,str]]:
    d={}
    for line in output.replace('\r\n','\n').splitlines():
        if not line.startswith('AG|'):continue
        p=line.split('|',3)
        if len(p)>=3:
            diag=''
            if len(p)==4:
                try: diag=base64.urlsafe_b64decode(p[3]+'='*(-len(p[3])%4)).decode('utf-8','replace')
                except Exception:diag=p[3]
            d[int(p[1])]=(p[2],diag[-5000:])
    return d

def symbol_table(aot:Path)->dict[str,list[tuple[int,int]]]:
    rr=core.run(['readelf','-sW',str(aot)],cwd=aot.parent,timeout=30)
    if not rr['ok']: raise RuntimeError('readelf_failed')
    d={}
    for line in rr['stdout'].splitlines():
        p=line.split()
        if len(p)>=8 and p[3]=='FUNC':
            try:d.setdefault(p[-1],[]).append((int(p[1],16),int(p[2])))
            except:pass
    return d

def extract(aot:Path,symbol:str,neutral_file:str,source_symbols:dict[str,list[str]],symtab:dict)->tuple[str,list[str]]:
    syms=symtab.get(symbol) or []
    if not syms: raise RuntimeError('candidate_symbol_not_found')
    address,size=sorted(syms,key=lambda x:(-x[1],x[0]))[0];stop=address+max(1,size)
    od=core.run(['objdump','-d','-Mintel','--no-show-raw-insn',f'--start-address=0x{address:x}',f'--stop-address=0x{stop:x}',str(aot)],cwd=aot.parent,timeout=30)
    if not od['ok']:raise RuntimeError('objdump_failed')
    rx=re.compile(r'^\s*([0-9a-fA-F]+):\s+(.+?)\s*$');ins=[]
    for line in od['stdout'].splitlines():
        m=rx.match(line)
        if m and not m.group(2).startswith(('.byte','(bad)')):ins.append((int(m.group(1),16),m.group(2)))
    if not ins:raise RuntimeError('no_disassembled_instructions')
    assembly=core.format_scrubbed_assembly(ins,symbol,source_symbols)
    return assembly,[f'0x{address:x}']


def compile_group(items:list[dict], workroot:Path, group_id:str, dart_version:str, input_sha:str, depth:int=0)->tuple[list[dict],list[dict]]:
    if not items:return [],[]
    wd=workroot/f'{group_id}_d{depth}_{items[0]["source_index"]}_{len(items)}';shutil.rmtree(wd,ignore_errors=True);wd.mkdir(parents=True)
    for x in items:
        lp=wd/f'row_{x["source_index"]}.dart';lp.write_text(x['compile_program']);x['lib_path']=str(lp)
    root=wd/'root.dart';root.write_text(root_program(items))
    # JIT validates all test harnesses through the real frontend/runtime.
    jit=core.run([str(DART),'run','root.dart'],cwd=wd,timeout=max(45,4*len(items)))
    if not jit['ok']:
        if len(items)>1:
            mid=len(items)//2;a1,r1=compile_group(items[:mid],workroot,group_id+'a',dart_version,input_sha,depth+1);a2,r2=compile_group(items[mid:],workroot,group_id+'b',dart_version,input_sha,depth+1);return a1+a2,r1+r2
        x=items[0];return [],[{'status':'rejected','source_index':x['source_index'],'source_id_sha256':x['source_id_sha256'],'reason':'jit_batch_failure','diagnostic':(jit['stderr'] or jit['stdout'])[-6000:]}]
    js=statuses(jit['stdout']);passed=[];reject=[]
    for x in items:
        st,diag=js.get(x['source_index'],('MISSING',''))
        if st=='PASS':passed.append(x)
        else:reject.append({'status':'rejected','source_index':x['source_index'],'source_id_sha256':x['source_id_sha256'],'reason':'jit_test_failed' if st=='FAIL' else 'jit_status_missing','diagnostic':diag or jit['stdout'][-3000:]})
    if not passed:return [],reject
    # Rebuild root with JIT-pass rows only.
    if len(passed)!=len(items):
        shutil.rmtree(wd,ignore_errors=True)
        ok2,r2=compile_group(passed,workroot,group_id+'p',dart_version,input_sha,depth+1);return ok2,reject+r2
    aot=wd/'batch.aot';comp=core.run([str(DART),'compile','aot-snapshot','root.dart','-o',str(aot)],cwd=wd,timeout=max(120,5*len(items)))
    if not comp['ok']:
        if len(items)>1:
            mid=len(items)//2;a1,r1=compile_group(items[:mid],workroot,group_id+'c',dart_version,input_sha,depth+1);a2,r2=compile_group(items[mid:],workroot,group_id+'d',dart_version,input_sha,depth+1);return a1+a2,reject+r1+r2
        x=items[0];return [],reject+[{'status':'rejected','source_index':x['source_index'],'source_id_sha256':x['source_id_sha256'],'reason':'aot_compile_failed','diagnostic':(comp['stderr'] or comp['stdout'])[-6000:]}]
    ar=core.run([str(AOT),str(aot)],cwd=wd,timeout=max(45,4*len(items)))
    if not ar['ok']:
        if len(items)>1:
            mid=len(items)//2;a1,r1=compile_group(items[:mid],workroot,group_id+'e',dart_version,input_sha,depth+1);a2,r2=compile_group(items[mid:],workroot,group_id+'f',dart_version,input_sha,depth+1);return a1+a2,reject+r1+r2
        x=items[0];return [],reject+[{'status':'rejected','source_index':x['source_index'],'source_id_sha256':x['source_id_sha256'],'reason':'aot_runtime_failure','diagnostic':(ar['stderr'] or ar['stdout'])[-6000:]}]
    ast=statuses(ar['stdout']);aotpass=[]
    for x in items:
        st,diag=ast.get(x['source_index'],('MISSING',''))
        if st=='PASS':aotpass.append(x)
        else:reject.append({'status':'rejected','source_index':x['source_index'],'source_id_sha256':x['source_id_sha256'],'reason':'aot_test_failed' if st=='FAIL' else 'aot_status_missing','diagnostic':diag or ar['stdout'][-3000:]})
    if not aotpass:return [],reject
    syms=symbol_table(aot);out=[]
    for x in aotpass:
        try:
            asm,entry=extract(aot,x['unique'],x['neutral_file'],x['source_symbols'],syms);cfg,edges,integ=core.graph_from_assembly(asm,entry);core.validate_cfg_mnemonics(cfg);asmsha=core.sha256_text(asm)
            gv={'schema':core.GRAPH_SCHEMA,'assembly_sha256':asmsha,'extractor_sha256':core.sha256_bytes((core.EXTRACTORS/'cfg_extractor.py').read_bytes()+(core.EXTRACTORS/'dfg_extractor.py').read_bytes()),'max_block_instrs':core.MAX_BLOCK_INSTRS,'max_dataflow_edges':0,'symbol_entry_addresses':entry}
            protocol={'schema':core.SCHEMA,'benchmark_kind':'training','fresh_holdout':False,'input_sha256':input_sha,'freeze_manifest_sha256':None,'original_source_sha256':x['original_source_sha256'],'semantic_function_name_sha256':x['target_sha256'],'neutral_target_name':'candidate','prompt_exposes':['assembly_or_graph','neutral_target_name'],'prompt_withholds':['return_type','parameter_types','parameter_names','reference_source','tests','semantic_function_name'],'assembly_build':{'function':'candidate','blocks':len(cfg),'cfg_edges':sum(e.get('edge_type')!='dataflow' for e in edges),'parsed_instructions':integ['parsed_instruction_count'],'external_direct_branches':integ['external_direct_branch_count'],'pruned_unreachable_blocks':0,'dart_sdk':dart_version,'jit_tests':'passed','aot_compile':'passed','aot_tests':'passed','build_version':core.BUILD_VERSION,'batch_size':len(items)}}
            common={'lang':'Dart','function':'candidate','camel_case_function_name':'candidate','python_function_name':'','dart_function_signature':'','prompt_signature_mode':'name_only','assembly':asm,'cfg':cfg,'edges':edges}
            pub=core.model_facing_row(common);priv={'dart_source':x['neutral_source'],**pub,'evaluation_only_dart_function_signature':x['neutral_signature'],'tests':x['neutral_harness']}
            led={'status':'retained','source_index':x['source_index'],'source_id_sha256':x['source_id_sha256'],'neutral_id':x['neutral_id'],'original_function_sha256':x['target_sha256'],'test_kind':x['test_kind'],'main_transformed':x['main_transformed'],'removed_demo_main_count':x['removed_demo_main_count'],'source_sha256':core.sha256_text(x['neutral_source']),'canonical_source_sha256':x['canonical_source_sha256'],'assembly_sha256':asmsha,'canonical_assembly_sha256':core.canonical_asm_hash(asm),'test_harness_sha256':core.sha256_text(x['neutral_harness']),'oracle':x['oracle'],'dart_sdk':dart_version,'jit':{'ok':True},'aot_compile':{'ok':True},'aot_run':{'ok':True},'batch_size':len(items),'prepare_elapsed_s':x['prepare_elapsed_s'],'provenance':{'neutral_file':x['neutral_file'],'benchmark_protocol':protocol,'graph_v2':gv,'integrity':integ}}
            out.append({'public':pub,'private':priv,'ledger':led})
        except Exception as e:reject.append({'status':'rejected','source_index':x['source_index'],'source_id_sha256':x['source_id_sha256'],'reason':'assembly_or_graph_failure','diagnostic':str(e)[-5000:]})
    shutil.rmtree(wd,ignore_errors=True)
    return out,reject


def write_jsonl(path:Path,rows:list[dict]):
    with path.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=DEFAULT_INPUT);ap.add_argument('--output-dir',type=Path,default=DEFAULT_OUT);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--limit',type=int);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True);input_sha=core.file_sha256(args.input);dv=core.run([str(DART),'--version'],cwd=out);dart_version=(dv['stderr'] or dv['stdout']).strip()
    rows=[]
    for i,line in enumerate(args.input.open(encoding='utf-8')):
        if line.strip():rows.append((i,json.loads(line)))
        if args.limit and len(rows)>=args.limit:break
    prep_path=out/'prepared_results.jsonl';prep=[];reject=[]
    print(json.dumps({'phase':'prepare','rows':len(rows),'dart':dart_version}),flush=True)
    with prep_path.open('w',encoding='utf-8') as f,cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(prepare_row,x,input_sha,dart_version) for x in rows]
        for n,fu in enumerate(cf.as_completed(futs),1):
            r=fu.result();f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            if r.get('ok'):prep.append(r)
            else:reject.append(r['reject'])
            if n%25==0 or n==len(rows):print(json.dumps({'phase':'prepare','done':n,'prepared':len(prep),'rejected':len(reject)}),flush=True)
    prep.sort(key=lambda x:x['source_index']);work=out/'batch_work';shutil.rmtree(work,ignore_errors=True);work.mkdir()
    compiled=[];batch_results=out/'batch_results.jsonl'
    with batch_results.open('w',encoding='utf-8') as bf:
        total=(len(prep)+args.batch_size-1)//args.batch_size
        for bi,start in enumerate(range(0,len(prep),args.batch_size),1):
            good,bad=compile_group(prep[start:start+args.batch_size],work,f'b{bi:04d}',dart_version,input_sha)
            compiled.extend(good);reject.extend(bad)
            for x in good:bf.write(json.dumps({'ok':True,**x},ensure_ascii=False)+'\n')
            for x in bad:bf.write(json.dumps({'ok':False,'reject':x},ensure_ascii=False)+'\n')
            bf.flush();os.fsync(bf.fileno())
            print(json.dumps({'phase':'compile','batch':bi,'batches':total,'compiled':len(compiled),'rejected':len(reject)}),flush=True)
    shutil.rmtree(work,ignore_errors=True)
    compiled.sort(key=lambda x:x['ledger']['source_index'])
    hold_src,hold_asm=core.load_holdout_hashes([ROOT/'grpo_data_graphv2_signature_scrubbed_private.jsonl',ROOT/'master_dart_cfg_dfg/master_dart_cfg_dfg_heldout.jsonl'])
    final=[];ss={};aa={}
    for x in compiled:
        l=x['ledger'];cs=l['canonical_source_sha256'];ca=l['canonical_assembly_sha256'];reason=None
        if cs in hold_src:reason='heldout_source_overlap'
        elif ca in hold_asm:reason='heldout_assembly_overlap'
        elif cs in ss:reason='duplicate_transformed_source'
        elif ca in aa:reason='duplicate_transformed_assembly'
        if reason:reject.append({'status':'rejected_postbuild','source_index':l['source_index'],'source_id_sha256':l['source_id_sha256'],'neutral_id':l['neutral_id'],'reason':reason,'duplicate_of':ss.get(cs) or aa.get(ca)})
        else:ss[cs]=l['neutral_id'];aa[ca]=l['neutral_id'];final.append(x)
    pubs=[x['public'] for x in final];privs=[x['private'] for x in final];leds=[x['ledger'] for x in final]
    pp=out/'master_dart_graphv2_signature_scrubbed_public.jsonl';pr=out/'master_dart_graphv2_signature_scrubbed_private.jsonl';lp=out/'master_dart_graphv2_compile_ledger.jsonl';qp=out/'master_dart_graphv2_quarantine.jsonl'
    write_jsonl(pp,pubs);write_jsonl(pr,privs);write_jsonl(lp,leds);write_jsonl(qp,sorted(reject,key=lambda x:x.get('source_index',10**9)))
    core.gzip_copy(pp,pp.with_suffix('.jsonl.gz'));core.gzip_copy(pr,pr.with_suffix('.jsonl.gz'))
    bad=[]
    for i,r in enumerate(pubs):
        missing=set(core.PUBLIC_MODEL_FIELDS)-set(r);extra=set(r)-set(core.PUBLIC_MODEL_FIELDS)
        if missing or extra:bad.append([i,{'missing':sorted(missing),'extra':sorted(extra)}])
        if r.get('function')!='candidate' or r.get('dart_function_signature')!='':bad.append([i,'identity'])
        if 'static void candidate(void)' in r.get('assembly','') or 'File file:///' in r.get('assembly',''):bad.append([i,'synthetic_header_or_file_id'])
        try:core.validate_cfg_mnemonics(r.get('cfg') or [])
        except Exception as exc:bad.append([i,str(exc)])
        residues=core.symbol_residue(r.get('assembly',''),core.source_symbol_names(privs[i]['dart_source']))
        if residues:bad.append([i,{'user_symbol_residue':residues}])
    mains=sum(1 for r in privs if re.search(r'(?m)^\s*(?:Future\s*<\s*void\s*>|void|dynamic)?\s*main\s*\(',r['dart_source']))
    manifest={'schema':'scrubbed-master-build-manifest-v2','build_version':core.BUILD_VERSION,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'input':{'path':str(args.input),'sha256':input_sha,'rows':len(rows)},'dart_sdk':dart_version,'requirements':{'real_dart_jit':True,'real_dart_aot_compile':True,'real_dartaotruntime':True,'main_logic_moved_to_candidate':True,'public_signature_scrubbed':True,'tests_hidden_from_public':True},'counts':{'processed':len(rows),'prepared':len(prep),'runtime_retained_before_dedup':len(compiled),'final_retained':len(final),'quarantined':len(reject),'main_rows_retained':sum(x['ledger']['main_transformed'] for x in final),'provided_harness_rows':sum(x['ledger']['test_kind']=='dart_harness' for x in final),'generated_function_test_rows':sum(x['ledger']['test_kind']=='differential_function' for x in final),'generated_main_test_rows':sum(x['ledger']['main_transformed'] for x in final)},'audit':{'public_forbidden_field_violations':bad,'private_sources_with_top_level_main':mains,'unique_canonical_sources':len(ss),'unique_canonical_assemblies':len(aa),'heldout_source_hashes':len(hold_src),'heldout_assembly_hashes':len(hold_asm),'passed':not bad and mains==0 and len(final)==len(ss)==len(aa)},'outputs':{}}
    for p in [pp,pr,pp.with_suffix('.jsonl.gz'),pr.with_suffix('.jsonl.gz'),lp,qp,prep_path,batch_results]:manifest['outputs'][p.name]={'sha256':core.file_sha256(p),'size_bytes':p.stat().st_size}
    mp=out/'master_dart_graphv2_scrubbed_manifest.json';mp.write_text(json.dumps(manifest,indent=2))
    (out/'README.md').write_text(f"""# Scrubbed HumanEval-style Dart CFG+DFG training set\n\n- Real compiler: `{dart_version}`\n- Final retained rows: **{len(final)}**\n- Quarantined rows: **{len(reject)}**\n- `main` rows converted to callable `candidate`: **{manifest['counts']['main_rows_retained']}**\n\nThe public file is a strict model-input allowlist containing only neutral `candidate` assembly/CFG/DFG and minimal contract fields. Identifiers, filenames, hashes, build provenance, source, tests, signatures, and expected values are excluded. The private file adds the withheld neutralized source, evaluation-only signature, and executable tests. Per-row identifiers, hashes, graph integrity, and provenance remain only in the compile ledger. Every row passed real Dart JIT, AOT compilation, and `dartaotruntime` execution.\n""")
    checks=out/'SHA256SUMS.txt';fs=[p for p in out.iterdir() if p.is_file() and p.name not in ('SHA256SUMS.txt',)];checks.write_text(''.join(f'{core.file_sha256(p)}  {p.name}\n' for p in sorted(fs)))
    zip_path=ROOT/'scrubbed_master_v2_bundle.zip';
    if zip_path.exists():zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix('')),'zip',root_dir=out);(ROOT/'scrubbed_master_v2_bundle.zip.sha256').write_text(f'{core.file_sha256(zip_path)}  {zip_path.name}\n')
    print(json.dumps(manifest['counts'],indent=2));print(json.dumps(manifest['audit'],indent=2));print(zip_path)
if __name__=='__main__':main()
