# Exact Architecture and Run Configuration

## Identity

- Experiment: `qwen3-8b-base_lora_enc_dec_r64_5e6_gcb_a128_arm64v21_s42_prefix_no_gine_regions16`
- Seed: 42
- Runner SHA-256: `31e378614ce07c01dfef24db3f4f3f077ce0d4a1c0165fb7777d17ce3a9a3ff6`
- Decoder: `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`
- Encoder: `microsoft/graphcodebert-base` at revision `2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d`
- Checkpoint SHA-256: `bc7ccbc6dbcae4755e93be7e685dd123b20a422085f9ec7793c1f456b6391085`
- Artifact repository: `raafatabualazm/antigravity-qwen3-8b-artifacts`

## Graph path

- Trainable GraphCodeBERT encoder with LoRA.
- GNN ablation is `identity`: no GINE propagation despite four configured GNN layers.
- CFG and DFG edge records are present; reverse edges are enabled.
- Block pooling: CLS.
- Block position mode: sinusoidal; encoder position scheme: RoBERTa.
- Four block vectors configured per block.
- Region compression: linear residual.
- Region maximum: 16 blocks.
- Maximum instructions per graph block: 20.
- Strict graph validation enabled.

## Decoder conditioning

- Dynamic graph prefix enabled.
- Prefix density: 4 tokens per log2 scale, minimum 4, configured maximum 64.
- Token-level prefix gate initialized to 0.2 with RMS matching.
- Graph-only conditioning channel: prompt assembly mode `none`; no cleaned/fitted assembly in the textual prompt.
- Decoder prompt maximum length: 2,048.
- Reasoning mode disabled.

## Optimization

- Decoder and encoder both trainable through LoRA.
- LoRA rank 64, alpha 128.
- Four SFT epochs.
- Learning rate `5e-6`, cosine scheduler, warmup ratio 0.03.
- Per-device train/eval batch 4, gradient accumulation 16.
- BF16, fused AdamW, weight decay 0.01.
- No 4-bit model loading.
- SDPA attention.
- Graph environment requests gradient checkpointing. The serialized Hugging Face `TrainingArguments` reports `gradient_checkpointing=false`, so inspect the model-level activation path before assuming it was absent.

## Evaluation

- Immutable eval rows: 343.
- Ten sampled candidates per task.
- Maximum 768 new tokens.
- Compile mode: aligned `jit_tests`.
- Pass stability runs: 3.
- Hidden scoring tests: `scoring_tests_visible_to_policy=false`.
- Prompt schema: `antigravity-v2-no-test-hints`.

## Immutable datasets

- Train: 1,371 rows, SHA-256 `f21782dd60edc11988867659dd2d16a5f6b6d2f550594cae09ad7cf92b68dcb7`.
- Eval: 343 rows, SHA-256 `864dc0bb7e9ee305ba0fc4be6e5d5ecbbeb7c17fd15bb3e41cfcc6d9aaf65fac`.
- Full pool: 1,714 rows, SHA-256 `bd64a1e8d24dc93a89f05d7f58cbaa9b4a09c7232e0a85555561f9dbeaa1519b`.
- Graph schema: `antigravity-graph-v2.1`.
- Train/eval and x86 benchmark overlap audits found no exact or near overlap.

## Runtime

- NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition.
- Python 3.12.13.
- PyTorch 2.8.0 + CUDA 12.8.
- Linux 6.8.0-85, glibc 2.39.

The machine-readable authority is `results/run_provenance.json`; this document is a reading aid.
