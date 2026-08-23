#!/usr/bin/env python3
"""Build and audit a direct-Qwen compact, signature-scrubbed graph stream.

The instruction codebook is fitted from ``--fit`` public rows only.  ``--measure``
rows never influence it.  CFG edges are encoded explicitly; block DFG edges are
recomputed with the frozen release extractor and must match exactly.
"""
from __future__ import annotations

import argparse, collections, hashlib, importlib.util, json, re
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SCHEMA = "direct-compact-causal-v1"
FAKE_SIGNATURE = "static void candidate(void)"
KNOWN_X86 = set("add addps addsd and call cdq cmovge cmovl cmp comisd cqo cvtsd2ss cvtsi2sd cvtss2sd cvttsd2si divsd idiv imul int3 ja jae jb jbe je jg jge jl jle jmp jne jno jnp jo jp lea mov movabs movaps movmskpd movq movsd movss movsx movsxd movups movzx mul mulps mulsd neg not or pop push ret roundsd sar sete setne shl shr shufps sqrtsd sub subsd test xchg xor xorpd xorps".split())
EDGE_TOKEN = {"conditional_true":"<CT>", "conditional_false":"<CF>",
              "linear_fallthrough":"<CN>", "loop_backedge":"<CL>",
              "unconditional":"<CU>", "unconditional_jump":"<CJ>"}
TOKEN_EDGE = {v:k for k,v in EDGE_TOKEN.items()}
CONTROL = ["<G2C1>","<AX64>","<ENTRY>","<BLOCKS>","<CFG>","<END>","<R>","<E>"] + list(TOKEN_EDGE)
TARGET_RE = re.compile(r"0x([0-9a-fA-F]+)\s*<([^>]+)>")
TAG_RE = re.compile(r"<[^>]+>")
RUNTIME_POLICY={"version":"runtime-symbol-policy-v1","trusted":["stub _iso_stub_*","dart:*","print"],
                "self":["candidate","candidate+*","candidate.*"],"untrusted":"per_function_@U#"}

def sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def stable(obj: Any) -> bytes: return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",",":")).encode()
def digest(value: str, label: str) -> str:
    value=str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}",value): raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value
def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for n,line in enumerate(f,1):
            if line.strip(): yield n,json.loads(line)

def load_dfg(path: Path):
    spec=importlib.util.spec_from_file_location("compact_v1_frozen_dfg",path)
    mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
    return mod.build_cross_block_dfg

def canonicalize(row: dict[str,Any], symbol_policy: str="runtime_aware") -> dict[str,Any]:
    cfg=row.get("cfg") or []
    starts={str(b.get("start_address","")).lower().removeprefix("0x"):int(b["id"]) for b in cfg}
    external: dict[str,int]={}; blocks=[]
    for b in cfg:
        ins=[]
        for raw in b.get("instructions") or []:
            s=str(raw).strip()
            if s == FAKE_SIGNATURE: continue
            parts=s.split()
            if not parts: continue
            op=parts[0].lower()
            if re.fullmatch(r"local_\d+",op) or op not in KNOWN_X86:
                raise ValueError(f"unknown_or_corrupt_mnemonic:{op}")
            def repl(m: re.Match[str]) -> str:
                addr,sym=m.group(1).lower(),m.group(2)
                if addr in starts: return f"@B{starts[addr]}"
                if sym.startswith("candidate+"): return "@REL+"+sym.split("+",1)[1]
                if sym=="candidate": return "@SELF"
                if sym.startswith("candidate."): return "@SELF_CLOSURE"
                if symbol_policy=="runtime_aware":
                    stub=re.match(r"^stub _iso_stub_([A-Za-z0-9_]+)",sym)
                    if stub: return "@STUB:"+stub.group(1)
                    if sym.startswith("dart:"): return "@SDK:"+re.sub(r"[^A-Za-z0-9_:.-]","_",sym)
                    if sym=="print": return "@SDK:print"
                # Every untrusted source type/helper is encounter-order neutral.
                if sym not in external: external[sym]=len(external)
                return f"@U{external[sym]}"
            s=TARGET_RE.sub(repl,s)
            s=s.replace("@SELF_CLOSURE>","@SELF_CLOSURE")
            s=re.sub(r"\s+"," ",s); s=re.sub(r"\s*,\s*",",",s)
            ins.append(s)
        blocks.append({"id":int(b["id"]),"instructions":ins})
    cfg_edges=[]; dfg=[]
    for e in row.get("edges") or []:
        z={"source":int(e["source"]),"target":int(e["target"]),"edge_type":str(e["edge_type"])}
        (dfg if z["edge_type"]=="dataflow" else cfg_edges).append(z)
    unknown=sorted({e["edge_type"] for e in cfg_edges}-set(EDGE_TOKEN))
    if unknown: raise ValueError("unknown_cfg_edge_type:"+",".join(unknown))
    return {"architecture":"x86_64","entry_blocks":list(row.get("integrity",{}).get("entry_blocks") or [0]),
            "blocks":blocks,"cfg_edges":cfg_edges,"dfg_edges":sorted(dfg,key=lambda e:(e["source"],e["target"],e["edge_type"]))}

def source_token_contract(tokenizer_path: Path, model_vocab_size: int, expansions: list[str], max_blocks: int) -> tuple[int,dict[str,list[int]],dict[str,int]]:
    """Semantic initialization map for every added input-only token."""
    base=Tokenizer.from_file(str(tokenizer_path)); tokenizer_size=base.get_vocab_size(with_added_tokens=True)
    human={"<G2C1>":" compact graph version one", "<AX64>":" x86 64", "<ENTRY>":" entry blocks",
           "<BLOCKS>":" basic blocks", "<CFG>":" control flow edges", "<END>":" end compact graph",
           "<R>":" raw instruction", "<E>":" end raw instruction", "<CT>":" conditional true",
           "<CF>":" conditional false", "<CN>":" linear fallthrough", "<CL>":" loop backedge",
           "<CU>":" unconditional", "<CJ>":" unconditional jump"}
    atoms=[f"<I{i}>" for i in range(len(expansions))]+[f"<B{i}>" for i in range(max_blocks)]+CONTROL
    atom_ids={token:model_vocab_size+i for i,token in enumerate(atoms)}; mapping={}
    for i,line in enumerate(expansions): mapping[str(atom_ids[f"<I{i}>"])]=base.encode(line,add_special_tokens=False).ids
    for i in range(max_blocks): mapping[str(atom_ids[f"<B{i}>"])]=base.encode(f" block {i}",add_special_tokens=False).ids
    for token,text in human.items(): mapping[str(atom_ids[token])]=base.encode(text,add_special_tokens=False).ids
    if any(k=="None" or not v for k,v in mapping.items()): raise ValueError("invalid_source_token_expansion")
    return tokenizer_size,mapping,atom_ids

def compact_ids(text: str, base: Tokenizer, atom_ids: dict[str,int]) -> list[int]:
    """Encode logical compact text without assigning IDs inside Qwen's model vocab."""
    out=[]; cursor=0
    for m in TAG_RE.finditer(text):
        if m.start()>cursor: out.extend(base.encode(text[cursor:m.start()],add_special_tokens=False).ids)
        token=m.group()
        if token not in atom_ids: raise ValueError("unknown_compact_atom:"+token)
        out.append(atom_ids[token]); cursor=m.end()
    if cursor<len(text): out.extend(base.encode(text[cursor:],add_special_tokens=False).ids)
    return out

def encode(c: dict[str,Any], code: dict[str,int]) -> str:
    out=["<G2C1>","<AX64>","<ENTRY>"]+[f"<B{x}>" for x in c["entry_blocks"]]+["<BLOCKS>"]
    for b in c["blocks"]:
        out.append(f"<B{b['id']}>")
        for s in b["instructions"]: out.append(f"<I{code[s]}>") if s in code else out.extend(("<R>",s,"<E>"))
    out.append("<CFG>")
    for e in c["cfg_edges"]: out.extend((EDGE_TOKEN[e["edge_type"]],f"<B{e['source']}>",f"<B{e['target']}>"))
    out.append("<END>"); return "".join(out)

def decode(text: str, expansions: list[str]) -> dict[str,Any]:
    tags=list(TAG_RE.finditer(text)); pos=0
    def take(expected=None):
        nonlocal pos
        if pos>=len(tags): raise ValueError("unexpected_eof")
        z=tags[pos].group(); pos+=1
        if expected and z!=expected: raise ValueError(f"expected_{expected}_got_{z}")
        return z
    take("<G2C1>"); take("<AX64>"); take("<ENTRY>"); entries=[]
    while pos<len(tags) and tags[pos].group()!="<BLOCKS>": entries.append(int(take()[2:-1]))
    take("<BLOCKS>")
    blocks=[]
    while pos<len(tags) and tags[pos].group()!="<CFG>":
        bid=int(take()[2:-1]); ins=[]
        while pos<len(tags) and not (tags[pos].group().startswith("<B") or tags[pos].group()=="<CFG>"):
            z=take()
            if z.startswith("<I"): ins.append(expansions[int(z[2:-1])])
            elif z=="<R>":
                start=tags[pos-1].end(); end=tags[pos].start(); ins.append(text[start:end]); take("<E>")
            else: raise ValueError("bad_instruction_token:"+z)
        blocks.append({"id":bid,"instructions":ins})
    take("<CFG>"); cfg=[]
    while tags[pos].group()!="<END>":
        et=take(); src=int(take()[2:-1]); dst=int(take()[2:-1]); cfg.append({"source":src,"target":dst,"edge_type":TOKEN_EDGE[et]})
    take("<END>")
    return {"architecture":"x86_64","entry_blocks":entries,"blocks":blocks,"cfg_edges":cfg}

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--fit",required=True,type=Path); ap.add_argument("--measure",action="append",type=Path,default=[])
    ap.add_argument("--output-dir",required=True,type=Path); ap.add_argument("--tokenizer-json",required=True,type=Path)
    ap.add_argument("--model-config",type=Path,default=None)
    ap.add_argument("--dfg-extractor",type=Path,default=ROOT/"scrubbed_master_v2_release/extractors/dfg_extractor.py")
    ap.add_argument("--codebook-size",type=int,default=16384); ap.add_argument("--max-blocks",type=int,default=4096)
    ap.add_argument("--max-source-tokens","--max-prompt-tokens",dest="max_source_tokens",type=int,default=9000)
    ap.add_argument("--max-target-tokens",type=int,default=3072)
    ap.add_argument("--max-total-tokens",type=int,default=12288)
    ap.add_argument("--tokenizer-fingerprint-sha256",required=True,
                    help="Logical HF tokenizer fingerprint printed by fingerprint_direct_tokenizer.py")
    ap.add_argument("--decoder-model",default="Qwen/Qwen3-8B")
    ap.add_argument("--decoder-revision",required=True,
                    help="Immutable decoder commit used with --model-config")
    ap.add_argument("--target-function",default="candidate")
    ap.add_argument("--symbol-policy",choices=["runtime_aware","strict_all_alias"],default="runtime_aware")
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    tokenizer_fingerprint_sha256=digest(a.tokenizer_fingerprint_sha256,"tokenizer_fingerprint_sha256")
    if not 0<a.max_source_tokens<=9000: raise ValueError("max_source_tokens must be in [1,9000]")
    if a.max_target_tokens<=0 or a.max_total_tokens<=0: raise ValueError("target/total limits must be positive")
    if a.max_total_tokens<a.max_source_tokens: raise ValueError("max_total_tokens must fit max_source_tokens")
    if not a.decoder_model.strip() or not a.decoder_revision.strip(): raise ValueError("decoder model and immutable revision are required")
    if not re.fullmatch(r"(?:candidate|fn\d+)",a.target_function.strip()): raise ValueError("target_function must be neutral")
    build_dfg=load_dfg(a.dfg_extractor); fit_good=[]; quarantine=[]; freq=collections.Counter()
    for n,r in rows(a.fit):
        try: c=canonicalize(r,a.symbol_policy); fit_good.append((n,r,c)); freq.update(s for b in c["blocks"] for s in b["instructions"])
        except Exception as e: quarantine.append({"dataset":str(a.fit),"line":n,"task_id":r.get("task_id"),"reason":str(e)})
    expansions=[s for s,_ in freq.most_common(a.codebook_size)]; code={s:i for i,s in enumerate(expansions)}
    model_config=a.model_config or a.tokenizer_json.with_name("config.json")
    model_cfg=json.loads(model_config.read_text(encoding="utf-8")); model_vocab_size=int(model_cfg["vocab_size"])
    base_tok=Tokenizer.from_file(str(a.tokenizer_json))
    tokenizer_vocab_size,source_expansions,atom_ids=source_token_contract(a.tokenizer_json,model_vocab_size,expansions,a.max_blocks)
    custom_ids=sorted(map(int,source_expansions))
    if not custom_ids or custom_ids[0]!=model_vocab_size or custom_ids!=list(range(model_vocab_size,model_vocab_size+len(custom_ids))):
        raise ValueError("custom source IDs must be contiguous after model vocab")
    datasets=[("fit",a.fit,fit_good)];
    for p in a.measure:
        good=[]
        for n,r in rows(p):
            try: good.append((n,r,canonicalize(r,a.symbol_policy)))
            except Exception as e: quarantine.append({"dataset":str(p),"line":n,"task_id":r.get("task_id"),"reason":str(e)})
        datasets.append(("measure",p,good))
    records=[]; lengths=[]; failures=[]; roundtrip_verified=0; dfg_edges_verified=0
    for role,p,good in datasets:
        for n,r,c in good:
            if len(c["blocks"])>a.max_blocks: failures.append({"task_id":r.get("task_id"),"reason":"block_vocab_overflow"}); continue
            text=encode(c,code); dec=decode(text,expansions)
            dec_dfg=build_dfg(dec["blocks"],dec["cfg_edges"],max_edges=100000)
            dec["dfg_edges"]=sorted(dec_dfg,key=lambda e:(e["source"],e["target"],e["edge_type"]))
            if dec!=c: failures.append({"task_id":r.get("task_id"),"reason":"canonical_or_graph_roundtrip_mismatch"}); continue
            roundtrip_verified+=1; dfg_edges_verified+=len(dec["dfg_edges"])
            # The training/inference stack owns the natural-language decoder
            # prompt.  compact_input_ids is only the reversible compact source;
            # embedding a second prompt here would silently double-prompt Qwen.
            ids=compact_ids(text,base_tok,atom_ids); length=len(ids); lengths.append(length)
            if length>a.max_source_tokens: failures.append({"task_id":r.get("task_id"),"reason":"source_token_overflow","tokens":length})
            records.append({"compact_input_ids":ids,"role":role,"dataset":str(p),"line":n,"task_id":r.get("task_id"),"compact_text":text,"source_tokens":length,
                            "canonical_sha256":sha(stable(c)),"compact_sha256":sha(text.encode()),"fallback_instructions":sum(s not in code for b in c["blocks"] for s in b["instructions"])})
    def pct(q):
        v=sorted(lengths); return v[round((len(v)-1)*q)] if v else 0
    cb={"schema":"compact-qwen-v1-codebook","fit_public_sha256":sha(a.fit.read_bytes()),"fit_retained":len(fit_good),
        "fit_quarantined":sum(x["dataset"]==str(a.fit) for x in quarantine),"codebook_size":len(expansions),"expansions":expansions,
        "added_token_scheme":{"instruction":"<I{index}>","block":"<B{index}>","control_tokens":CONTROL,"edge_tokens":EDGE_TOKEN},
        "tokenizer_json_sha256":sha(a.tokenizer_json.read_bytes()),"tokenizer_vocab_size":tokenizer_vocab_size,
        "model_config_sha256":sha(model_config.read_bytes()),"decoder_model":a.decoder_model.strip(),
        "decoder_revision":a.decoder_revision.strip(),"model_vocab_size":model_vocab_size,"base_vocab_size":model_vocab_size,
        "source_token_expansions":source_expansions,"source_atom_ids":atom_ids,
        "dfg_extractor_sha256":sha(a.dfg_extractor.read_bytes()),"max_blocks":a.max_blocks,"symbol_policy":a.symbol_policy,
        "runtime_symbol_policy":RUNTIME_POLICY if a.symbol_policy=="runtime_aware" else None,
        "runtime_symbol_policy_sha256":sha(stable(RUNTIME_POLICY)) if a.symbol_policy=="runtime_aware" else None}
    leakage={"candidate":sum("candidate" in s.lower() for s in expansions),"file_uri":sum("file://" in s.lower() for s in expansions),
             "absolute_symbol_address":sum(bool(re.search(r"0x[0-9a-fA-F]+\s*<",s)) for s in expansions),
             "private_field_terms":sum(bool(re.search(r"dart_source|semantic_function|original_source|\btests\b",s,re.I)) for s in expansions)}
    role_counts=dict(collections.Counter(x["role"] for x in records))
    report={"schema":"compact-qwen-v1-preflight","rows_retained":len(records),"rows_by_role":role_counts,"quarantined":len(quarantine),
            "quarantine_reasons":dict(collections.Counter(x["reason"] for x in quarantine)),"failures_count":len(failures),"failure_examples":failures[:50],
            "tokens":{"kind":"compact_source_only","min":min(lengths) if lengths else 0,"p50":pct(.5),"p95":pct(.95),"p99":pct(.99),"max":max(lengths) if lengths else 0,"limit":a.max_source_tokens},
            "lossless_invariants":{"lossless_domain":"scrubbed_canonical_graph",
                "privacy_scrub_is_only_intentional_irreversibility":True,
                "exact_instruction_entry_cfg_roundtrip_rows":roundtrip_verified,
                "dfg_regenerated_and_matched_rows":roundtrip_verified,
                "dfg_edges_matched_edge_for_edge":dfg_edges_verified,
                "dfg_extractor_sha256":sha(a.dfg_extractor.read_bytes()),
                "unknown_tokens":0,"truncated_rows":0,"raw_fallback_is_reversible":True},
            "codebook_expansion_leakage_scan":leakage,"passed":not failures and not any(leakage.values()),"exploratory_full_release_fit":not a.measure}
    cb_bytes=(json.dumps(cb,ensure_ascii=False,indent=2)+"\n").encode(); (a.output_dir/"codebook.json").write_bytes(cb_bytes)
    codebook_sha=sha(cb_bytes); codec_sha=sha(Path(__file__).read_bytes()); tokenizer_sha=sha(a.tokenizer_json.read_bytes())
    report["contract"]={"compact_codec_sha256":codec_sha,"compact_codebook_sha256":codebook_sha,
                        "compact_tokenizer_sha256":tokenizer_sha,"base_vocab_size":model_vocab_size,
                        "tokenizer_vocab_size":tokenizer_vocab_size,"model_vocab_size":model_vocab_size,
                        "model_config_sha256":cb["model_config_sha256"],
                        "decoder_model":cb["decoder_model"],"decoder_revision":cb["decoder_revision"],
                        "target_function":a.target_function.strip(),
                        "target_language":"Dart",
                        "tokenizer_fingerprint_sha256":tokenizer_fingerprint_sha256,
                        "source_token_expansion_count":len(source_expansions),
                        "runtime_symbol_policy_sha256":cb["runtime_symbol_policy_sha256"]}
    contract={"schema":CONTRACT_SCHEMA,"codec_sha256":codec_sha,"codebook_sha256":codebook_sha,
              "tokenizer_json_sha256":tokenizer_sha,"tokenizer_fingerprint_sha256":tokenizer_fingerprint_sha256,
              "model_config_sha256":cb["model_config_sha256"],"decoder_model":cb["decoder_model"],
              "decoder_revision":cb["decoder_revision"],"max_source_tokens":a.max_source_tokens,
              "target_function":a.target_function.strip(),
              "target_language":"Dart",
              "dfg_extractor_sha256":cb["dfg_extractor_sha256"],
              "lossless_domain":"scrubbed_canonical_graph",
              "max_target_tokens":a.max_target_tokens,"max_total_tokens":a.max_total_tokens,
              "base_vocab_size":model_vocab_size,"source_token_ids":custom_ids,
              "source_token_expansions":source_expansions,"source_embedding_init":"codebook_mean"}
    (a.output_dir/"compact_contract.json").write_text(json.dumps(contract,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (a.output_dir/"compact_model_inputs.jsonl").open("w",encoding="utf-8") as f:
        for x in records:
            model={"compact_input_ids":x["compact_input_ids"],"compact_codec_sha256":codec_sha,
                   "compact_codebook_sha256":codebook_sha,"compact_tokenizer_sha256":tokenizer_sha}
            assert set(model)=={"compact_input_ids","compact_codec_sha256","compact_codebook_sha256","compact_tokenizer_sha256"}
            f.write(json.dumps(model,separators=(",",":"))+"\n")
    with (a.output_dir/"alignment_private.jsonl").open("w",encoding="utf-8") as f:
        for i,x in enumerate(records):
            private={k:v for k,v in x.items() if k!="compact_input_ids"}; private["model_row"]=i
            f.write(json.dumps(private,ensure_ascii=False,separators=(",",":"))+"\n")
    with (a.output_dir/"quarantine.jsonl").open("w",encoding="utf-8") as f:
        for x in quarantine:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    with (a.output_dir/"failures.jsonl").open("w",encoding="utf-8") as f:
        for x in failures:f.write(json.dumps(x,ensure_ascii=False)+"\n")
    (a.output_dir/"preflight_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    names=["codebook.json","compact_contract.json","compact_model_inputs.jsonl","alignment_private.jsonl","quarantine.jsonl","failures.jsonl","preflight_report.json"]
    (a.output_dir/"SHA256SUMS.txt").write_text("".join(f"{sha((a.output_dir/n).read_bytes())}  {n}\n" for n in names),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["passed"]: raise SystemExit(1)

if __name__=="__main__": main()
