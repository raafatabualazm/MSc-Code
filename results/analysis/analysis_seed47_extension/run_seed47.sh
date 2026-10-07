#!/usr/bin/env bash
# Seed-47 extension: three arms, preregistered single-primary design.
# Amendment sha256: 526eef2cd6eaa5681cff3caea5c93c8af9ae5be1ca0b2ddca1afe4ec3dddce8a
set -uo pipefail
WS=/workspace
PROJ=$WS/hybrid_training_patch_v2_3
PY=/venv/main/bin/python
CK=$WS/artifacts/t5gemma2_4b4b_enriched_sft_2epoch_v1/checkpoint-optstep-000348
DATA=$WS/multifunction_v1/build
OUT=$WS/artifacts/t5gemma2_f2_seed47_extension_v1
ST=$OUT/STATUS.txt
BASE_REV=487d4acf21a4d70c70bf534265b5263c9424979e
mkdir -p "$OUT"
note(){ echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$ST"; }

ln -sfn $WS/dart-sdk /root/dart-sdk

# ---- 1. wait for HF auth to the gated base (max 4h) --------------------------
note "waiting for HF auth to gated base"
for i in $(seq 1 240); do
  if $PY - <<PYEOF 2>/dev/null
from huggingface_hub import model_info
model_info("google/t5gemma-2-4b-4b", revision="$BASE_REV")
PYEOF
  then note "HF auth OK"; break; fi
  [ $i -eq 240 ] && { note "HOLD: no HF auth after 4h"; exit 78; }
  sleep 60
done

# ---- 2. tokenizer: regenerate + content-hash gate ---------------------------
if [ ! -s "$CK/tokenizer/tokenizer.json" ]; then
  note "regenerating tokenizer from pinned base"
  $PY - <<PYEOF || { note "HOLD: tokenizer regen failed"; exit 79; }
import hashlib, json, sys
from pathlib import Path
from transformers import AutoTokenizer
ck = Path("$CK")/"tokenizer"
tok = AutoTokenizer.from_pretrained("google/t5gemma-2-4b-4b", revision="$BASE_REV")
tok.save_pretrained(ck)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec = {f.name: sha(f) for f in sorted(ck.iterdir()) if f.is_file()}
stm = rec.get("special_tokens_map.json","")
vocab_candidates = {n:h for n,h in rec.items() if n in ("tokenizer.model","tokenizer.json")}
try:
    vj = hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True,
         separators=(",",":")).encode()).hexdigest()
    vocab_candidates["canonical_vocab_json"] = vj
except Exception: pass
rec["_vocab_candidates"] = vocab_candidates
Path("$OUT/tokenizer_regen_hashes.json").write_text(json.dumps(rec, indent=1))
SEALED_STM = "a237afa0a3964f4db32b59e1031adcc948cf01b552badfdf5a96092422a19884"
SEALED_VOC = "aacdb7cfcf76c202d7f7a157fda1392c83138d20e0d2b433a062e8dfd99b4c91"
ok_stm = (stm == SEALED_STM)
ok_voc = SEALED_VOC in vocab_candidates.values()
print("special_tokens_map match:", ok_stm, " vocab match:", ok_voc)
sys.exit(0 if (ok_stm and ok_voc) else 1)
PYEOF
fi
note "tokenizer in place: $(sha256sum $CK/tokenizer/tokenizer.json | cut -c1-16)"

# ---- 3. gold Rank-0 gate ----------------------------------------------------
if [ ! -s "$OUT/gold_k1_score.json" ]; then
  note "gold round-trip gate"
  $PY - <<PYEOF || { note "ABORT: gold prediction build failed"; exit 80; }
import json
from pathlib import Path
rows=[json.loads(l) for l in open("$DATA/dev_multifunction_binary.jsonl",encoding="utf-8") if l.strip()]
preds=[{"id":r["task_id"],"predictions":[r["dart_source"]]} for r in rows]
Path("$OUT/gold_k1_predictions.json").write_text(json.dumps(preds))
print(len(preds),"gold rows")
PYEOF
  PYTHONPATH=$PROJ $PY $PROJ/scripts/evaluation/score_direct_compact_passk.py \
    --predictions "$OUT/gold_k1_predictions.json" \
    --evaluation_file "$DATA/dev_multifunction_binary.jsonl" \
    --output "$OUT/gold_k1_score.json" \
    --k 1 --workers 32 --timeout 30 --stability_runs 2 \
    || { note "ABORT: gold scoring failed"; exit 80; }
fi
GOLD=$($PY -c "import json;d=json.load(open('$OUT/gold_k1_score.json'));print(d['pass_at_k']['count'])")
[ "$GOLD" = "175" ] || { note "ABORT: gold round-trip $GOLD/175"; exit 80; }
note "gold gate 175/175"

# ---- 4. arms, seed 47 -------------------------------------------------------
common=( --dataset "$DATA/dev_multifunction_binary.jsonl"
  --dataset_seal "$DATA/dev_multifunction_binary.seal.json"
  --f2_jsonl "$DATA/dev_multifunction_binary_f2.jsonl"
  --f2_manifest "$DATA/dev_multifunction_binary_f2.jsonl.manifest.json"
  --sft_checkpoint "$CK" --arm sft
  --num_samples 10 --generation_batch_size 10
  --max_source_tokens 32768 --max_new_tokens 4096
  --temperature 0.8 --top_p 0.95 --bf16 --seed 47 )

run_arm(){ # name script extra...
  local name=$1 script=$2; shift 2
  local pred=$OUT/${name}_seed47_k10_predictions.json
  local score=$OUT/${name}_seed47_k10_score.json
  [ -s "$score" ] && { note "skip $name (scored)"; return 0; }
  note "generate $name"
  PYTHONPATH=$PROJ $PY $PROJ/scripts/evaluation/$script \
    "${common[@]}" "$@" --output "$pred" \
    >> "$OUT/${name}.log" 2>&1 || { note "ABORT: generation failed $name"; exit 81; }
  note "score $name"
  PYTHONPATH=$PROJ $PY $PROJ/scripts/evaluation/score_direct_compact_passk.py \
    --predictions "$pred" --evaluation_file "$DATA/dev_multifunction_binary.jsonl" \
    --output "$score" --k 10 --workers 32 --timeout 30 --stability_runs 2 \
    >> "$OUT/${name}.log" 2>&1 || { note "ABORT: scoring failed $name"; exit 82; }
  note "done $name: $($PY -c "import json;d=json.load(open('$score'));print('pass',d['pass_at_k']['count'],'compile',d['compile_at_k']['count'])")"
}

run_arm baseline            t5gemma2_f2_passk_inference.py
run_arm typed_opaque_contract t5gemma2_measurement_audit_inference.py --input_view typed_opaque_contract
run_arm semantic_body_swap  t5gemma2_measurement_audit_inference.py --input_view semantic_body_swap

# ---- 5. preregistered analysis ---------------------------------------------
$PY - <<'PYEOF' > "$OUT/seed47_analysis.json.tmp" 2>>"$ST"
import json, itertools
from statistics import mean
OUT="/workspace/artifacts/t5gemma2_f2_seed47_extension_v1"
LEAK="sigless_8bf7f40ca356"
def counts(p):
    d=json.load(open(p)); tr=d.get("task_results") or []
    P=sum(1 for r in tr if r.get("pass_at_k") and str(r["task_id"])!=LEAK)
    C=sum(1 for r in tr if r.get("compile_at_k") and str(r["task_id"])!=LEAK)
    return P,C
s47={a:counts(f"{OUT}/{a}_seed47_k10_score.json")
     for a in ("baseline","typed_opaque_contract","semantic_body_swap")}
prior={  # clean-174, seeds 42-46, sealed artifacts
 "baseline":{"pass":[6,6,6,4,2],"compile":[123,124,134,125,124]},
 "typed_opaque_contract":{"pass":[9,11,10,9,11],"compile":[169,169,168,165,171]},
 "semantic_body_swap":{"pass":[4,3,8,5,6],"compile":[117,113,122,118,117]},
}
def flip(d):
    obs=abs(mean(d)); c=0
    for sg in itertools.product([1,-1],repeat=len(d)):
        if abs(mean([s*v for s,v in zip(sg,d)]))>=obs-1e-12: c+=1
    return c/2**len(d)
res={"seed47_clean174":{a:{"pass":s47[a][0],"compile":s47[a][1]} for a in s47}}
def contrast(arm,metric,idx):
    d=[t-b for t,b in zip(prior[arm][metric],prior["baseline"][metric])]
    d6=d+[s47[arm][idx]-s47["baseline"][idx]]
    return {"diffs_n6":d6,"mean":round(mean(d6),3),"p_paired_n6":flip(d6),
            "sign_consistent":all(x>0 for x in d6) or all(x<0 for x in d6)}
res["PRIMARY_typed_pass10"]=contrast("typed_opaque_contract","pass",0)
res["secondary_typed_compile10"]=contrast("typed_opaque_contract","compile",1)
res["secondary_bodyswap_compile10"]=contrast("semantic_body_swap","compile",1)
p=res["PRIMARY_typed_pass10"]
res["preregistered_verdict"]=(
 "PRIMARY SIGNIFICANT at alpha=0.05 (p=%.5f, n=6 paired blocks)"%p["p_paired_n6"]
 if p["sign_consistent"] and p["p_paired_n6"]<=0.05 else
 "PRIMARY NOT SIGNIFICANT (p=%.5f); report as registered"%p["p_paired_n6"])
res["amendment_sha256"]="526eef2cd6eaa5681cff3caea5c93c8af9ae5be1ca0b2ddca1afe4ec3dddce8a"
res["hard_stop"]="no seed beyond 47 will be added"
print(json.dumps(res,indent=1))
PYEOF
mv "$OUT/seed47_analysis.json.tmp" "$OUT/seed47_analysis.json"
note "ANALYSIS COMPLETE"
$PY -c "import json;print(json.load(open('$OUT/seed47_analysis.json'))['preregistered_verdict'])" | tee -a "$ST"
note "ALL DONE"
