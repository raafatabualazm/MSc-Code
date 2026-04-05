# MSc-Code

Toolkit and dataset suite for assembly-to-Dart and assembly-to-Swift decompilation research.

This repository contains:

- curated JSONL datasets for Dart and Swift decompilation
- dataset construction and validation scripts
- supervised fine-tuning and GRPO-style training scripts
- compile-oriented, CodeBLEU, and executable evaluation scripts
- archived benchmark outputs and intermediate artifacts

The codebase is organized for research use rather than as a packaged library. Most scripts are meant to be run from the repository root.

## Repository layout

```text
MSc-Code/
+-- artifacts/                 # generated models and other local artifacts
+-- configs/                   # configuration files
+-- data/
|   +-- datasets/              # base training/eval datasets
|   +-- matched/               # matched Dart/Swift corpora
|   +-- testing/               # testing and benchmark datasets
|   `-- intermediate/          # generated intermediate data and failed cases
+-- docs/                      # paper drafts and documentation
+-- results/
|   +-- cache/                 # checkpoints and resumable eval caches
|   +-- pass_at_k/             # executable benchmark summaries
|   `-- statistics/            # compile/CodeBLEU summaries
+-- scripts/
|   +-- data/                  # data validation, matching, and generation
|   +-- training/              # SFT, distillation, and GRPO scripts
|   `-- evaluation/            # compile, CodeBLEU, pass@k, and analysis scripts
+-- tree-sitter-dart/          # Dart grammar source
+-- tree-sitter-swift/         # Swift grammar source
`-- requirements.txt
```

## Main datasets

The current repository snapshot includes the following key assets:

| Path | Size | Role |
| --- | ---: | --- |
| `data/datasets/dart_all.jsonl` | 1,195 | Base Dart source/assembly training pairs |
| `data/matched/all_dart_matched.jsonl` | 658 | Dart reasoning-augmented matched data |
| `data/matched/all_swift_matched.jsonl` | 658 | Swift reasoning-augmented matched data |
| `data/matched/matched_datasets/` | 412 pairs | Token-matched Dart/Swift bilingual subset |
| `data/matched/matched_datasets-deepseek/` | 415 pairs | Alternate token-matched bilingual subset |
| `data/datasets/test-set.jsonl` | 165 | Held-out evaluation set |
| `data/testing/test-set2.jsonl` | 73 | Testing dataset |
| `data/testing/compile-test.jsonl` | 34 | Small compile-oriented benchmark |
| `data/testing/compile-test2.jsonl` | 126 | Main compile-oriented benchmark |
| `data/testing/grpo_data.jsonl` | 154 | Executable benchmark with unit tests |

Typical record fields include `assembly`, `source`, `reasoning`, `language`, and for executable tasks additional fields such as `task_id`, `tests`, and canonical function signatures.

## Script groups

### Data scripts

Located in `scripts/data/`.

- `validate_data.py`: validates JSONL structure before matching
- `token_matcher.py`: creates token-balanced Dart/Swift matched subsets
- `builder3.py`, `builder4.py`, `builder5.py`: reasoning/data generation pipelines
- `dart_decomp_two_stage_pipeline.py`: analysis -> generation -> repair workflow

### Training scripts

Located in `scripts/training/`.

- `chunked_sft_fixed.py`: long-context SFT with chunked processing and KV-cache handling
- `trainer-v1.py`, `trainer-v2.py`, `trainer-v3.py`: supervised training baselines
- `train_qwen3-v1.py`, `train_qwen3-v2.py`: Qwen-based fine-tuning variants
- `GRPO_Trainv2.py`, `GRPO_Trainv3.py`, `grpo_training_h200.py`, `Dart_grpo_training_fixed.py`: GRPO-style training variants
- `dual_distil*.py`, `unsloth_train.py`: distillation and alternate training setups

### Evaluation scripts

Located in `scripts/evaluation/`.

- `compile-test.py`: compile-oriented evaluation through an OpenAI-compatible API
- `tester.py`, `tester2.py`, `tester3.py`: model evaluation variants
- `Pass_at k_openai.py`: executable `pass@k` evaluation on `grpo_data.jsonl`
- `Two_stage_pass_at k_openai.py`: two-stage executable evaluation
- `Tokens_per_line.py`: token-length analysis utility

## Setup

Python and the exact runtime stack vary by script, but a reasonable starting point is:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Additional runtime assumptions:

- many training scripts expect a CUDA-capable GPU and recent PyTorch/Transformers versions
- API-backed scripts require an OpenAI-compatible endpoint or OpenRouter-style access
- some scripts use `OPENAI_API_KEY` or `HUGGINGFACE_TOKEN`
- compile and executable evaluation require the Dart toolchain on `PATH`
- CodeBLEU-related scripts require Tree-sitter language support; the repo includes `tree-sitter-dart/` and `tree-sitter-swift/`, but you may still need to install/build the corresponding Python bindings in your environment

## Quick start

Run all commands from the repository root.

### 1. Validate matched-language source files

```bash
python scripts/data/validate_data.py \
  --dart-file data/matched/all_dart_matched.jsonl \
  --swift-file data/matched/all_swift_matched.jsonl
```

### 2. Build a token-matched bilingual subset

```bash
python scripts/data/token_matcher.py \
  --dart-file data/matched/all_dart_matched.jsonl \
  --swift-file data/matched/all_swift_matched.jsonl \
  --output-dir data/matched/matched_datasets \
  --tolerance 0.1 \
  --max-diff 50
```

### 3. Run compile-oriented evaluation against a local OpenAI-compatible server

```bash
python scripts/evaluation/compile-test.py \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --test-set data/testing/compile-test2.jsonl \
  --k 5
```

Outputs are written under `results/cache/` and `results/statistics/`.

## Archived results

The repository already includes benchmark outputs under `results/`. Examples from the current snapshot:

- compiled CodeBLEU summary files under `results/statistics/`
- executable `pass@k` summary files under `results/pass_at_k/`
- cached generation checkpoints under `results/cache/`

These files are useful as baselines and for reproducing previously run experiments, but they were produced by multiple model families and should not be treated as one perfectly controlled benchmark suite.

## Notes and caveats

- this repo is a research artifact, so several scripts are experiment snapshots rather than polished CLI tools
- some training and evaluation scripts hard-code dataset names, model names, or output directories
- large intermediate files may be present locally but excluded from git
- local `_local.py` evaluation variants are intended for self-hosted or workstation-specific runs

## Citation

If you use this repository, cite the associated paper or repository snapshot once the archival version is finalized.
