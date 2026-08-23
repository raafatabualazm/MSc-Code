#!/usr/bin/env python3
"""Leakage-safe split preparation for the direct compact-Qwen experiment.

This stage does not fit a codebook and does not emit model-facing examples. It
only selects original release lines and writes private alignment sidecars.
"""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path
from typing import Iterable


HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(WORKSPACE))
import build_scrubbed_dataset as release_builder
from hybrid_training_patch_v2_3.scripts.training import hybrid_data_controls as controls


SCHEMA='direct-compact-split-preparation-v1'
JACCARD_THRESHOLD=.55
SEQUENCE_THRESHOLD=.70
DEV_FRACTION=.10
DEFAULT_SEED='direct-compact-split-v1-20260719'


def file_sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text:str)->str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_jsonl(path:Path)->list[dict]:
    rows=[]
    with path.open(encoding='utf-8') as handle:
        for line_number,line in enumerate(handle,1):
            if not line.strip():continue
            value=json.loads(line)
            if not isinstance(value,dict):
                raise ValueError(f'{path}:{line_number}: expected object')
            rows.append(value)
    return rows


def write_jsonl(path:Path,rows:Iterable[dict])->int:
    count=0
    with path.open('w',encoding='utf-8',newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
            count+=1
    return count


def write_json(path:Path,value:dict)->None:
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def remove_demo_main(source:str,function_name:str)->str:
    """Remove top-level demo main, retaining it only when it is the target."""
    parser=release_builder.parser_new()
    functions=release_builder.functions_from_source(source,parser)
    remove=[fn for fn in functions if fn.name=='main' and function_name!='main']
    if not remove:return source
    return release_builder.apply_edits(source,[(fn.full_start,fn.body_end,'') for fn in remove])


def fingerprint_source(row:dict)->tuple[str,str,str,tuple[str,...],frozenset[str]]:
    source=controls.source_text(row)
    if not source.strip():raise ValueError('missing_source')
    function_name=controls.infer_function_name(row)
    if not function_name:raise ValueError('missing_function_name')
    source=remove_demo_main(source,function_name)
    # Retention/inlining directives are compiler controls, not program
    # semantics. Ignoring them prevents build-protocol differences from hiding
    # a benchmark collision.
    source_lines=[]
    for line in source.splitlines():
        compact=''.join(line.split()).replace('"',"'")
        if compact in {
            "@pragma('vm:entry-point')", "@pragma('vm:never-inline')",
            "@pragma('vm:prefer-inline')",
        }:
            continue
        source_lines.append(line)
    prepared={'dart_source':'\n'.join(source_lines),'function':function_name}
    neutral=controls.normalized_source(prepared)
    alpha=controls.alpha_normalized_source(prepared)
    alpha_tokens=tuple(alpha.split())
    # Token 5-grams retain local order and are substantially less prone than a
    # unigram set to declaring unrelated Dart boilerplate near-identical.
    # SequenceMatcher below remains the independent second gate.
    width=5
    shingles=(
        frozenset('\x1f'.join(alpha_tokens[index:index+width]) for index in range(len(alpha_tokens)-width+1))
        if len(alpha_tokens)>=width else frozenset(alpha_tokens)
    )
    return (
        text_sha256(neutral),text_sha256(alpha),alpha,alpha_tokens,
        shingles,
    )


@dataclasses.dataclass(frozen=True)
class Entry:
    original_line:int
    neutral_sha256:str
    alpha_sha256:str
    alpha_text:str
    alpha_tokens:tuple[str,...]
    token_set:frozenset[str]
    corpus:str='release'


def make_entry(row:dict,line:int,corpus:str)->Entry:
    neutral,alpha_hash,alpha_text,tokens,token_set=fingerprint_source(row)
    return Entry(line,neutral,alpha_hash,alpha_text,tokens,token_set,corpus)


class SimilarityIndex:
    """Exact inverted-index candidate generation for the two-part near gate."""
    def __init__(self,entries:list[Entry]):
        self.entries=entries
        self.postings:dict[str,list[int]]=defaultdict(list)
        for index,entry in enumerate(entries):
            for token in entry.token_set:self.postings[token].append(index)

    def query(self,entry:Entry)->tuple[list[tuple[int,float,float]],int]:
        intersections=Counter()
        for token in entry.token_set:
            for index in self.postings.get(token,()):intersections[index]+=1
        matches=[];jaccard_compares=0
        for index,intersection in intersections.items():
            other=self.entries[index]
            union=len(entry.token_set)+len(other.token_set)-intersection
            jaccard=(intersection/union) if union else 1.0
            jaccard_compares+=1
            if jaccard<JACCARD_THRESHOLD:continue
            # Use SequenceMatcher's documented default autojunk heuristic. It
            # prevents quadratic behavior on long Dart token streams dominated
            # by punctuation while retaining the requested ratio definition.
            sequence=difflib.SequenceMatcher(
                None,entry.alpha_tokens,other.alpha_tokens,
            ).ratio()
            if sequence>=SEQUENCE_THRESHOLD:
                matches.append((index,jaccard,sequence))
        matches.sort(key=lambda item:(-item[1],-item[2],self.entries[item[0]].corpus,self.entries[item[0]].original_line))
        return matches,jaccard_compares


class DisjointSet:
    def __init__(self,size:int):
        self.parent=list(range(size));self.rank=[0]*size

    def find(self,value:int)->int:
        while self.parent[value]!=value:
            self.parent[value]=self.parent[self.parent[value]]
            value=self.parent[value]
        return value

    def union(self,left:int,right:int)->None:
        left=self.find(left);right=self.find(right)
        if left==right:return
        if self.rank[left]<self.rank[right]:left,right=right,left
        self.parent[right]=left
        if self.rank[left]==self.rank[right]:self.rank[left]+=1


def select_dev_groups(groups:list[list[int]],entries:list[Entry],seed:str,target:int)->set[int]:
    """Deterministic subset-sum selection without splitting semantic groups."""
    ordered=[]
    for group_index,members in enumerate(groups):
        identity='|'.join(sorted(entries[index].alpha_sha256 for index in members))
        order_key=text_sha256(seed+'|'+identity)
        ordered.append((order_key,group_index,len(members)))
    ordered.sort()
    reachable={0:None}
    for _,group_index,size in ordered:
        additions={}
        for subtotal in sorted(reachable,reverse=True):
            candidate=subtotal+size
            if candidate not in reachable and candidate not in additions:
                additions[candidate]=(subtotal,group_index)
        reachable.update(additions)
    chosen_total=min(reachable,key=lambda value:(abs(value-target),value>target,value))
    selected=set();cursor=chosen_total
    while cursor:
        previous,group_index=reachable[cursor]
        selected.add(group_index);cursor=previous
    return selected


def safe_path(path:Path)->str:
    try:return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()
    except ValueError:return path.name


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--release-dir',type=Path,default=HERE)
    parser.add_argument('--codec-quarantine',type=Path,default=HERE/'compact_qwen_v1/quarantine.jsonl')
    parser.add_argument('--forbidden',type=Path,action='append',default=None)
    parser.add_argument('--output-dir',type=Path,default=HERE/'direct_compact_split_v1')
    parser.add_argument('--seed',default=DEFAULT_SEED)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    forbidden_paths=args.forbidden or [
        WORKSPACE/'data/testing/grpo_data_graphv2_sigscrub_v3_opaque_nameonly_private.jsonl',
        WORKSPACE/'data/testing/grpo_data_graphv2_sigscrub_v3_opaque_neutralexact_private.jsonl',
        WORKSPACE/'data/testing/grpo_data_graphv2_signature_scrubbed_private.jsonl',
        WORKSPACE/'data/testing/fresh_graphv2_holdout_s44.jsonl',
    ]
    release=args.release_dir
    public_path=release/'master_dart_graphv2_signature_scrubbed_public.jsonl'
    private_path=release/'master_dart_graphv2_signature_scrubbed_private.jsonl'
    ledger_path=release/'master_dart_graphv2_compile_ledger.jsonl'
    required=[public_path,private_path,ledger_path,args.codec_quarantine,*forbidden_paths]
    missing=[str(path) for path in required if not path.exists()]
    if missing:raise SystemExit(f'missing inputs: {missing}')
    output=args.output_dir
    if output.exists():
        if not args.force:raise SystemExit(f'output exists: {output}; pass --force')
        resolved=output.resolve()
        if resolved.parent!=HERE.resolve():
            raise SystemExit(f'refusing to replace output outside release directory: {resolved}')
        shutil.rmtree(output)
    output.mkdir(parents=True)
    started=time.monotonic()

    public=load_jsonl(public_path);private=load_jsonl(private_path);ledger=load_jsonl(ledger_path)
    if not(len(public)==len(private)==len(ledger)):
        raise SystemExit(f'alignment count mismatch: {len(public)}/{len(private)}/{len(ledger)}')
    for index,(pub,pri,led) in enumerate(zip(public,private,ledger),1):
        if pub.get('task_id') and pri.get('task_id') and pub['task_id']!=pri['task_id']:
            raise SystemExit(f'public/private id mismatch at line {index}')
        if pub.get('task_id') and led.get('neutral_id') and pub['task_id']!=led['neutral_id']:
            raise SystemExit(f'public/ledger id mismatch at line {index}')
        if hashlib.sha256(str(pri.get('dart_source') or '').encode()).hexdigest()!=led.get('source_sha256'):
            raise SystemExit(f'private/ledger source hash mismatch at line {index}')
        if hashlib.sha256(str(pub.get('assembly') or '').encode()).hexdigest()!=led.get('assembly_sha256'):
            raise SystemExit(f'public/ledger assembly hash mismatch at line {index}')

    codec_rows=load_jsonl(args.codec_quarantine)
    codec_lines=set()
    for row in codec_rows:
        line=int(row.get('line') or 0)
        if not 1<=line<=len(public):raise SystemExit(f'invalid codec quarantine line: {line}')
        expected=row.get('task_id');observed=public[line-1].get('task_id')
        if expected and observed and expected!=observed:
            raise SystemExit(f'codec quarantine identity mismatch at line {line}')
        codec_lines.add(line)

    forbidden_entries=[];forbidden_counts={};forbidden_input=[]
    for path in forbidden_paths:
        rows=load_jsonl(path);forbidden_counts[safe_path(path)]=len(rows)
        forbidden_input.append({'path':safe_path(path),'sha256':file_sha256(path),'rows':len(rows)})
        for line,row in enumerate(rows,1):
            forbidden_entries.append(make_entry(row,line,safe_path(path)))
    forbidden_neutral:dict[str,list[Entry]]=defaultdict(list)
    forbidden_alpha:dict[str,list[Entry]]=defaultdict(list)
    for entry in forbidden_entries:
        forbidden_neutral[entry.neutral_sha256].append(entry)
        forbidden_alpha[entry.alpha_sha256].append(entry)
    forbidden_index=SimilarityIndex(forbidden_entries)

    rejections=[];near_audit=[];candidates=[];reason_counts=Counter()
    seen_neutral={};seen_alpha={}
    for line,row in enumerate(private,1):
        if line in codec_lines:
            reason='codec_quarantine';reason_counts[reason]+=1
            rejections.append({'original_line':line,'reason':reason})
            continue
        entry=make_entry(row,line,'release')
        if entry.neutral_sha256 in forbidden_neutral:
            ref=sorted(forbidden_neutral[entry.neutral_sha256],key=lambda item:(item.corpus,item.original_line))[0]
            reason='forbidden_exact_neutral';reason_counts[reason]+=1
            rejections.append({'original_line':line,'reason':reason,'forbidden_corpus':ref.corpus,'forbidden_line':ref.original_line})
            continue
        if entry.alpha_sha256 in forbidden_alpha:
            ref=sorted(forbidden_alpha[entry.alpha_sha256],key=lambda item:(item.corpus,item.original_line))[0]
            reason='forbidden_alpha_structural';reason_counts[reason]+=1
            rejections.append({'original_line':line,'reason':reason,'forbidden_corpus':ref.corpus,'forbidden_line':ref.original_line})
            continue
        if entry.neutral_sha256 in seen_neutral:
            reason='duplicate_exact_neutral';reason_counts[reason]+=1
            rejections.append({'original_line':line,'reason':reason,'duplicate_original_line':seen_neutral[entry.neutral_sha256]})
            continue
        if entry.alpha_sha256 in seen_alpha:
            reason='duplicate_alpha_structural';reason_counts[reason]+=1
            rejections.append({'original_line':line,'reason':reason,'duplicate_original_line':seen_alpha[entry.alpha_sha256]})
            continue
        seen_neutral[entry.neutral_sha256]=line;seen_alpha[entry.alpha_sha256]=line
        candidates.append(entry)

    retained=[];forbidden_near_compares=0
    for entry in candidates:
        matches,compares=forbidden_index.query(entry);forbidden_near_compares+=compares
        if matches:
            index,jaccard,sequence=matches[0];ref=forbidden_entries[index]
            reason='forbidden_near_clone';reason_counts[reason]+=1
            detail={'original_line':entry.original_line,'reason':reason,'forbidden_corpus':ref.corpus,
                    'forbidden_line':ref.original_line,'jaccard':round(jaccard,6),'sequence':round(sequence,6)}
            rejections.append(detail);near_audit.append({'scope':'forbidden',**detail})
        else:retained.append(entry)

    internal_index=SimilarityIndex([]);internal_index.entries=[]
    dsu=DisjointSet(len(retained));internal_edges=[];internal_compares=0
    # Build incrementally so each unordered pair is evaluated exactly once.
    postings:dict[str,list[int]]=defaultdict(list)
    for right,entry in enumerate(retained):
        counts=Counter()
        for token in entry.token_set:
            for left in postings.get(token,()):counts[left]+=1
        for left,intersection in counts.items():
            other=retained[left];union=len(entry.token_set)+len(other.token_set)-intersection
            jaccard=(intersection/union) if union else 1.0;internal_compares+=1
            if jaccard<JACCARD_THRESHOLD:continue
            sequence=difflib.SequenceMatcher(None,entry.alpha_tokens,other.alpha_tokens).ratio()
            if sequence<SEQUENCE_THRESHOLD:continue
            dsu.union(left,right);internal_edges.append((left,right,jaccard,sequence))
        for token in entry.token_set:postings[token].append(right)
    groups_by_root:dict[int,list[int]]=defaultdict(list)
    for index in range(len(retained)):groups_by_root[dsu.find(index)].append(index)
    groups=sorted(groups_by_root.values(),key=lambda members:min(retained[index].original_line for index in members))
    group_number={member:number for number,members in enumerate(groups,1) for member in members}
    for left,right,jaccard,sequence in sorted(internal_edges,key=lambda edge:(retained[edge[1]].original_line,retained[edge[0]].original_line)):
        near_audit.append({'scope':'retained_internal','original_line':retained[right].original_line,
                           'other_original_line':retained[left].original_line,'jaccard':round(jaccard,6),
                           'sequence':round(sequence,6),'semantic_group':group_number[right]})

    target_dev=round(len(retained)*DEV_FRACTION)
    selected_dev_groups=select_dev_groups(groups,retained,args.seed,target_dev)
    train=[];dev=[]
    for group_index,members in enumerate(groups):
        destination=dev if group_index in selected_dev_groups else train
        destination.extend(members)
    train.sort(key=lambda index:retained[index].original_line)
    dev.sort(key=lambda index:retained[index].original_line)
    train_lines={retained[index].original_line for index in train};dev_lines={retained[index].original_line for index in dev}
    if train_lines&dev_lines or len(train_lines|dev_lines)!=len(retained):
        raise SystemExit('split partition invariant failed')
    if any(any((member in train) for member in members) and any((member in dev) for member in members) for members in groups):
        raise SystemExit('semantic group crossed train/dev')

    train_alignment=[{'original_line':retained[index].original_line,'split_line':position,
                      'semantic_group':group_number[index]} for position,index in enumerate(train,1)]
    dev_alignment=[{'original_line':retained[index].original_line,'split_line':position,
                    'semantic_group':group_number[index]} for position,index in enumerate(dev,1)]
    output_counts={}
    output_counts['train_private_alignment.jsonl']=write_jsonl(output/'train_private_alignment.jsonl',train_alignment)
    output_counts['dev_private_alignment.jsonl']=write_jsonl(output/'dev_private_alignment.jsonl',dev_alignment)
    output_counts['rejections.jsonl']=write_jsonl(output/'rejections.jsonl',sorted(rejections,key=lambda row:row['original_line']))
    output_counts['near_clone_audit.jsonl']=write_jsonl(output/'near_clone_audit.jsonl',near_audit)

    largest_groups=sorted((len(group) for group in groups),reverse=True)
    rejection_categories=(
        'codec_quarantine','forbidden_exact_neutral','forbidden_alpha_structural',
        'duplicate_exact_neutral','duplicate_alpha_structural','forbidden_near_clone',
    )
    report={
        'schema':SCHEMA,
        'status':'passed',
        'counts':{
            'release_rows':len(private),'codec_quarantine_rows':len(codec_lines),
            'forbidden_rows_raw':len(forbidden_entries),
            'forbidden_neutral_fingerprints':len(forbidden_neutral),
            'forbidden_alpha_fingerprints':len(forbidden_alpha),
            'pre_near_candidates':len(candidates),'retained':len(retained),
            'rejected':len(rejections),'train':len(train),'dev':len(dev),
            'semantic_groups':len(groups),'internal_near_edges':len(internal_edges),
        },
        'rejection_reasons':{reason:reason_counts[reason] for reason in rejection_categories},
        'near_clone':{
            'jaccard_threshold':JACCARD_THRESHOLD,'sequence_threshold':SEQUENCE_THRESHOLD,
            'jaccard_representation':'set of alpha-normalized token 5-grams',
            'forbidden_token_set_comparisons':forbidden_near_compares,
            'internal_token_set_comparisons':internal_compares,
            'forbidden_matches':reason_counts['forbidden_near_clone'],
            'internal_edges':len(internal_edges),
        },
        'split':{
            'requested_train_fraction':1-DEV_FRACTION,'requested_dev_fraction':DEV_FRACTION,
            'actual_train_fraction':(len(train)/len(retained)) if retained else 0,
            'actual_dev_fraction':(len(dev)/len(retained)) if retained else 0,
            'target_dev_rows':target_dev,'seed':args.seed,
            'largest_semantic_groups':largest_groups[:20],
            'downstream_codebook_fit_source':'train_private_alignment.jsonl only',
            'dev_excluded_from_codebook_fit':True,
            'forbidden_families_excluded_from_codebook_fit':True,
        },
        'invariants':{
            'public_private_ledger_aligned':True,'codec_quarantine_applied_by_original_line':True,
            'no_exact_neutral_forbidden_overlap':True,'no_alpha_structural_forbidden_overlap':True,
            'no_threshold_near_forbidden_overlap':True,'no_retained_exact_or_alpha_duplicates':True,
            'semantic_groups_do_not_cross_split':True,'alignment_sidecars_use_original_line_only':True,
            'codebook_fitted':False,'model_facing_rows_emitted':False,
        },
        'elapsed_seconds':round(time.monotonic()-started,3),
    }
    write_json(output/'report.json',report)
    output_counts['report.json']=1
    manifest={
        'schema':SCHEMA,
        'created_unix':int(time.time()),
        'script':{'path':safe_path(Path(__file__)),'sha256':file_sha256(Path(__file__))},
        'inputs':{
            'public':{'path':safe_path(public_path),'sha256':file_sha256(public_path),'rows':len(public)},
            'private':{'path':safe_path(private_path),'sha256':file_sha256(private_path),'rows':len(private)},
            'ledger':{'path':safe_path(ledger_path),'sha256':file_sha256(ledger_path),'rows':len(ledger)},
            'codec_quarantine':{'path':safe_path(args.codec_quarantine),'sha256':file_sha256(args.codec_quarantine),'rows':len(codec_rows)},
            'forbidden':forbidden_input,
        },
        'fingerprints':{
            'exact':'hybrid_data_controls.normalized_source_sha256 after demo-main/VM-pragma removal',
            'alpha':'hybrid_data_controls.alpha_structural_sha256 after demo-main/VM-pragma removal',
        },
        'thresholds':{'jaccard':JACCARD_THRESHOLD,'jaccard_representation':'alpha-token-5gram-set',
                      'sequence_matcher':SEQUENCE_THRESHOLD},
        'split':{'method':'deterministic semantic-component subset-sum 90/10','seed':args.seed},
        'report_sha256':file_sha256(output/'report.json'),
        'outputs':{},
    }
    for name,count in sorted(output_counts.items()):
        path=output/name
        manifest['outputs'][name]={'sha256':file_sha256(path),'size_bytes':path.stat().st_size,'rows':count}
    write_json(output/'split_manifest.json',manifest)
    print(json.dumps(report,indent=2,sort_keys=True))
    print(output/'split_manifest.json')


if __name__=='__main__':main()
