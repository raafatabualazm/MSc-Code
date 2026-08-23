# Frontier-ceiling audit — 2026-07-23

Remote scope: `ssh -p 24424 root@98.218.15.126`, `/workspace`.

## Verdict

The reported DeepSeek result, `5/60 = 8.33%`, is an observed result from one
incomplete, non-auditable evaluation protocol. It is **not a defensible
frontier ceiling**, and the claimed `8.33% - 5.71% = 2.62 pp` teacher/student
gap is not an apples-to-apples estimate.

No API calls were made during this audit and no running job was interrupted.
The six stored student candidate files were re-scored with the current exact
Dart harness; all published full-set pass and compile totals reproduced.

## Exact evaluation mapping

All six student results use the same 175 semantic holdout task IDs and the same
acceptance-test mapping. The v3 rows are ordered differently and use a
different compact contract/target name.

| Result | Checkpoint / predictions | Evaluation input | pass@10 | 95% Wilson CI |
|---|---|---|---:|---:|
| baseline | `direct_compact_fn0_sft_v1`; `artifacts/passk_gen.json` | raw compact, common 175 | 1/175 (0.57%) | 0.10–3.17% |
| soft-KD | `direct_compact_softkd_v1`; `artifacts/passk_softkd_gen.json` | real-constant compact, common 175 | 2/175 (1.14%) | 0.31–4.07% |
| GPT hard target | `direct_compact_gpt_sft_v1`; `artifacts/passk_gpt_gen.json` | real-constant compact, common 175 | 3/175 (1.71%) | 0.58–4.92% |
| v3 clean | `direct_compact_v3_clean_sft_v1`; `artifacts/v3_passk_gen.json` | v3-native compact, common 175 | 3/175 (1.71%) | 0.58–4.92% |
| real-enriched | `direct_compact_fn0_real_sft_v1`; `artifacts/passk_real_gen.json` | real-constant compact, common 175 | 4/175 (2.29%) | 0.89–5.73% |
| proxy | `direct_compact_fn0_enr_sft_v1`; `artifacts/passk_enr_gen.json` | source-regex/oracle constants, common 175 | 10/175 (5.71%) | 3.13–10.20% |
| DeepSeek | no candidate artifact; aggregate log only | truncated raw disassembly + a different real-constant extraction, first 60 only | 5/60 (8.33%) | 3.61–18.07% |

Generation input hashes:

- raw common holdout public file: `63b1af9224760a2ca2652b328d1f9fbf3754e79dd9f73462da9c9ee0454f171a`
- real-constant public file: `d42355b6d3e6a0af368ff1e0a8ffdcd44dcba3e99577931bf5f903ef9d2055fb`
- source-regex proxy public file: `42eea51c66ea85ea2095f25f6d74ef4fe8c13936d2838b83d45b5dfb59273bed`
- v3-native public file: `bad3a869bfc3373d6a1e7cf1ae42050efaf743b9a7f4c7c895e0408b212eb699`

## Exact first-60 comparison

DeepSeek used `dev_fn0_real.jsonl[:60]`, not a registered/random 60-task
manifest. Re-scoring each student on those exact IDs gives:

| Arm | pass@10 | compile@10 |
|---|---:|---:|
| baseline | 0/60 | 38/60 |
| soft-KD | 0/60 | 40/60 |
| GPT | 0/60 | 39/60 |
| v3 | 1/60 | 34/60 |
| real-enriched | 0/60 | 43/60 |
| proxy | 3/60 (5.00%; CI 1.71–13.70%) | 43/60 |
| DeepSeek aggregate | 5/60 (8.33%; CI 3.61–18.07%) | 16/60 |

Therefore, even using the flawed DeepSeek aggregate, the task-matched proxy
difference is 3.33 pp, not 2.62 pp. Because DeepSeek did not save its five
passing task IDs, the paired table cannot be reconstructed. For every possible
overlap between the 5 DeepSeek and 3 proxy successes, the exact two-sided
McNemar p-value is 0.50–0.73. Against the real-constant student's 0/60,
McNemar's exact p-value would be 0.0625. These data do not establish a small
or statistically resolved teacher/student gap.

Student pass IDs (independent re-score):

- baseline: `sigless_adc978f7e65b`
- soft-KD: `sigless_a507056ae077`, `sigless_b33d2de9af80`
- GPT: `sigless_8bf7f40ca356`, `sigless_a507056ae077`, `sigless_b885d1ba1001`
- v3: `sigless_949e6c11e478`, `sigless_96546ea01643`, `sigless_ed4d7dcd5f20`
- real-enriched: `sigless_8bf7f40ca356`, `sigless_a507056ae077`,
  `sigless_b33d2de9af80`, `sigless_b885d1ba1001`
- proxy: `sigless_083954eb2fb3`, `sigless_34e9dc99606e`,
  `sigless_4b8ae6c364f9`, `sigless_6ad70e03dabe`,
  `sigless_6f74b303025c`, `sigless_897c175fe34d`,
  `sigless_8bf7f40ca356`, `sigless_96546ea01643`,
  `sigless_c8721da111a6`, `sigless_edd7ed243c48`

## Protocol defects that invalidate “ceiling”

1. **The assembly is incomplete.** `teacher_repair_qwen.disasm()` returns
   `"\n".join(chunks)[:12000]`. In the DeepSeek first 60, 33/60 cached prompts
   are exactly 12,000 characters and do not contain `End of assembler dump.`
   Thus 55% lose the function tail.

2. **Anonymous and other task-local functions are omitted.** The symbol parser
   `r"\b(fn0(?:\.[\w$]+)?)\b"` cannot capture names containing spaces/angle
   brackets. For `sigless_8230c078962e`, GDB reports both `fn0` and
   `fn0.<anonymous closure>`, while the cached prompt contains only `fn0`.
   Class/helper methods not prefixed by `fn0` are also outside the search.

3. **The cache is not content-addressed.** `asm_cache.jsonl` stores only
   `{task_id, asm}`. It has no source SHA, AOT/toolchain identity, disassembler
   schema, symbol manifest, or completeness bit. Merely fixing `disasm()` will
   continue to reuse the stale truncated rows.

4. **Effective K is unknown.** API exceptions, 180-second timeouts, empty
   answers, and max-token terminations are silently counted as failed samples.
   The code saves neither candidates nor errors, retries, finish reasons,
   request IDs, per-task K, or prompt/answer/reasoning token counts. `MAXTOK`
   was 8,000, not the requested 12K reasoning budget.

5. **Inputs differ across the comparison.** DeepSeek reads
   `real_constants_whole.jsonl`; the real/GPT/soft-KD compact input was built
   from `real_constants.jsonl`. Their records differ on 13/175 tasks, including
   4/60 DeepSeek tasks. The proxy's 5.71% uses regex extraction from gold source,
   not binary-recovered constants; its compact IDs differ from the real-constant
   input on all first 60 tasks.

6. **Generation protocols differ.** Students use K=10, 1,024 answer tokens,
   temperature 0.8, top-p 0.95, seed 42. DeepSeek uses up to 8,000 total
   completion/reasoning tokens, temperature 0.8, provider-default top-p, no
   seed, and silent failures. The student is domain-trained; DeepSeek receives
   one zero-shot prompt.

7. **The result is not reproducible.** `frontier_passk.py` only prints aggregate
   tuples held in memory. Unlike the student candidate artifacts, the 5/60
   result cannot be re-scored or paired.

## v3 / Opus ARM split audit

The published v3 student result is bound by provenance to:

- checkpoint: `/workspace/artifacts/direct_compact_v3_clean_sft_v1`
- train SHA: `92b243f75965977374e01a6ad706273dba1bd7f94a7f9c67742467283a17e5ce`
  = `artifacts/v3_clean/v3clean_train.jsonl` (2,776)
- eval SHA: `e30808c0f7c64011f5e578794e6669f1bda4ed3eff1616029e07eaae244d84ae`
  = `artifacts/v3_clean/v3clean_eval.jsonl` (175)

The v3 measure/eval task-ID set equals the common 175 holdout exactly.
Conversely, `v3_frontier_dev.jsonl`
(`702687e76f09ac3394dcae0a13ea699527843dbbf99d43c0d7a0e4b31e19201b`)
contains 175/175 task IDs and canonicalized sources from v3
`private_fit`/training, and has zero task-ID/source overlap with
`private_measure`/the common holdout. Therefore the DeepSeek “v3” rerun is on
v3 training tasks, not the Opus ARM's 175-task held-out evaluation.

Train/eval split checks:

- baseline, real, proxy, and GPT arms: zero train/eval task-ID overlap and zero
  canonical-source overlap.
- v3: zero train/eval task-ID overlap and zero normalized-source overlap.
  There is exactly one identical compact canonical representation across the
  split: train `sigless_de6a5acac393` and eval
  `sigless_561333b633a8`. Their sources implement the same minute-based
  departure sorting function. The eval twin was not one of v3's three passing
  tasks, so it did not contribute to 3/175.

## Required matched evaluation

Use a frozen, hash-bound manifest of the same 175 tasks for every arm. Build
full disassembly and compact representations from the same AOT binaries and one
shared binary-constant artifact. Full disassembly must include every reachable
task-local symbol and must fail closed if truncated.

Run the factorial comparison:

1. same frontier model, full representation versus compact representation;
2. frontier model versus student, both on the same compact representation.

For every task require exactly ten valid completed answers, a 12K reasoning/
completion allowance with a suitable timeout, infrastructure retries, and
saved prompt hashes, raw answers, finish reasons, reasoning/answer token use,
errors, and per-candidate verifier results. Report paired per-task differences
with bootstrap confidence intervals and McNemar tests, plus multiple generation
seeds. Until that run exists, call 5/60 a flawed observed run, not a ceiling.

## Evidence hashes

- `/workspace/frontier_passk.py`:
  `8e60cb5b007375a47d98e2dd6016147999a6d9d4d8534f5affa4cde2a2554c38`
- `/workspace/teacher_repair_qwen.py`:
  `4dd730926906d980d5bac441dadd2d101216be562d4b4fd928dc9927840ea537`
- `/workspace/build_v3_frontier_dev.py`:
  `b63351563dd7a63feafd97b39abb2ffc1b8b912983429f41a6a7c275cd1d0649`
- exact Dart evaluator:
  `/workspace/scripts/evaluation/graph_compile_at_k_antigravity.py`,
  `d70ff375b57117893953e7e76da84d5a3156f27c61842f4956fd59ffe3a3bea1`
- exact extraction/evaluation wrapper:
  `/workspace/scripts/training/teacher_repair_dataset_antigravity.py`,
  `2ae9c3b012d11baa0f65224b4ab8e18b05807e16fa750850b4a25b7e2790c72a`
