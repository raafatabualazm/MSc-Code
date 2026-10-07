# Model Weights

## Public Repository

```text
repo: raafatabualazm/antigravity-qwen3-8b-artifacts
revision: bc992cecbb6968be2e20b6e99e8a4c420e32242c
base decoder: Qwen/Qwen3-8B
graph encoder: microsoft/graphcodebert-base
```

The repository also contains historical 9B-lineage files. The paper's x86 architecture table uses only paths beginning with `artifacts/qwen3-8b-`.

## Checkpoint Format

`pytorch_model.bin` is a composite Antigravity state dictionary. Depending on the arm it can contain:

- GraphCodeBERT/local-block encoder LoRA state;
- Qwen3-8B decoder LoRA state;
- learned graph-prefix projection/gate state;
- graph pooling/glue parameters.

It is not a standalone Hugging Face causal-language-model directory. Recreate the matching architecture with `code/configs/run_sweeps_antigravity.py` and `code/scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py`, then load the state dictionary with the same LoRA rank/alpha, prefix count, graph mode, and RMS setting.

## Integrity

- `manifests/huggingface_model_manifest.json` records every file at the pinned repository revision.
- `manifests/huggingface_files_qwen3_8b.csv` is the paper-specific 8B subset.
- LFS `sha256` values come from the Hugging Face repository API at the pinned revision.

Do not compare or merge checkpoints solely by filename. The prefix width, RMS matching, prompt mode, CFG/DFG configuration, and training objective must match.

