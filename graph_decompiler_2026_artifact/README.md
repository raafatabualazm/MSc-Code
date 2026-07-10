# Graph-Conditioned Dart AOT Decompilation Artifact

This bundle supports the paper **Graph-Conditioned Neural Decompilation of Dart AOT Binaries: Pass@k, Reward Adaptation, and Multi-Arm Candidate Ensembling**.

It releases the graph-conditioned Qwen3-8B pipeline, GRPO and RS-SFT code, the two evaluation corpora, raw prediction pools, frontier-model controls, statistical outputs, and the exact public model-weight revision used by the paper.

## Headline Scope

- Decoder: `Qwen/Qwen3-8B` (there is no `-Base` suffix in the released identifier).
- Graph channel: LoRA-adapted GraphCodeBERT/local-block encoder, CFG/DFG edges, and learned Qwen prefix tokens.
- Functional benchmark: 154 test-equipped HumanEval-Dart tasks.
- Primary compile metric: JIT/pass-harness compile on the same 154 candidate-plus-tests files as pass@k.
- Historical diagnostics: CodeBLEU and standalone AOT compile on a separate 126-row standalone-program corpus.
- Frontier snapshot date: July 8, 2026.

The two corpora are not subsets of one another. See [DATA_CARD.md](DATA_CARD.md).

## Released Results

- Clean G3: pass@10 `0.1558` (24/154 covered tasks).
- Binary GRPO: pass@10 `0.1818` (28/154).
- Seventeen-arm union: 43/154, using a much larger and unmatched candidate budget.
- Azure `gpt-chat-latest` snapshot: pass@10 `0.7013`.
- GPT-5.5 v1 full-assembly ablation: pass@10 `0.7143`; this is a separate stochastic run with a different output cap and is not a replacement for the canonical frontier row.

These are fixed-pool, task-level results, not repeated-seed estimates.

## Layout

```text
code/                 Training, GRPO, data, evaluation, and baseline clients
data/benchmark/       154-task functional and 126-row legacy corpora
data/harvest/         Released RS-SFT harvest rows and report
data/future/          Compressed synthetic and ARM64 future-work assets
frontier/             Raw hosted-model and GPT-5.5 ablation pools
results/              Raw specialized predictions and metric/statistical outputs
environment/          Captured local verification package manifest
manifests/            Hugging Face and local SHA-256 manifests
paper/                Paper source, bibliography, and built PDF
docs/                 Commands and project handoff
```

See [VERIFICATION.md](VERIFICATION.md) for the release audit and
`manifests/files_sha256.csv` for per-file integrity hashes.

## Environment

The archived local verification environment used Python 3.12.7 with:

```text
torch==2.11.0+cu128
transformers==5.9.0
peft==0.18.0
torch-geometric==2.7.0
accelerate==1.6.0
```

Dart must be on `PATH` for compile/pass evaluation. The pass harness invokes `dart run`; the aligned compile classifier uses that same path and distinguishes front-end errors from runtime/test failures.

## Metric Reproduction

From this artifact root:

```bash
python code/scripts/evaluation/graph_pass_at_k_antigravity.py \
  --predictions results/specialized_predictions/qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_graphonly_pass_predictions.json \
  --k_values 1,5,10 \
  --workers 16 \
  --timeout 30
```

```bash
python code/scripts/evaluation/graph_compile_at_k_antigravity.py \
  --predictions results/specialized_predictions/qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_x86_g3_graphonly_pass_predictions.json \
  --compile_mode jit_tests \
  --k_values 1,5,10 \
  --workers 16 \
  --timeout 30
```

Precomputed outputs are under `results/metrics/compile154_jitpassharness/` and `results/metrics/pass154_same_pools/`.

## Model Weights

Weights are hosted at:

```text
https://huggingface.co/raafatabualazm/antigravity-qwen3-8b-artifacts
revision: bc992cecbb6968be2e20b6e99e8a4c420e32242c
```

Download only the paper's 8B checkpoints with:

```bash
hf download raafatabualazm/antigravity-qwen3-8b-artifacts \
  --revision bc992cecbb6968be2e20b6e99e8a4c420e32242c \
  --include "artifacts/qwen3-8b-*/*" \
  --local-dir .
```

See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) and `manifests/huggingface_files_qwen3_8b.csv` before loading a checkpoint.

## Known Reproducibility Gaps

1. The ephemeral training-pod `pip freeze` was not captured before pod destruction. Local verification pins, runner settings, `training_args.bin` files, and hosted file hashes are available, but they are not a pod-side environment snapshot.
2. Literal training commands were not preserved for four auxiliary repair pools used only in reranking/union analysis. Their predictions and metrics are released, but they are not independently reproducible core training arms.
3. Training was not repeated over multiple random seeds.
4. The Azure `gpt-chat-latest` identifier is a moving hosted alias. Raw outputs and query metadata preserve the July 8, 2026 snapshot, but the provider cannot be forced to recreate it later.

The hosted comparison is evaluated at a fixed budget of ten requests per task. Empty content and request failures remain failed slots: 115/1,540 slots are blank for GPT-5.5, 765 for Sonnet 5, 1,233 for DeepSeek V4 Pro, and 1,188 for GLM-5.2. This is an end-to-end API baseline, not a comparison conditioned on obtaining ten non-empty programs.

See [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md) for the complete audit trail.

## License and Dual Use

No open-source license has yet been selected by all authors. See [LICENSE_STATUS.md](LICENSE_STATUS.md) before reuse. Third-party model and benchmark terms continue to apply. The release contains benchmark artifacts only, not proprietary applications, malware, credentials, or exploit chains; see [SECURITY_AND_DUAL_USE.md](SECURITY_AND_DUAL_USE.md).
