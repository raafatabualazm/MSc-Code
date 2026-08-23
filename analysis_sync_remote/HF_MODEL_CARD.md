---
base_model: google/t5gemma-2-4b-4b
library_name: peft
license: gemma
license_link: https://ai.google.dev/gemma/terms
tags:
  - peft
  - lora
  - t5gemma
  - research-artifact
  - software-engineering
---

# T5Gemma 2 VeRPO and RS-SFT research adapters

This repository archives 45 PEFT LoRA adapter checkpoints from 15 research-run
families studying decompilation and typed-contract interventions for Dart source
generation from an F2 binary representation.

These are research checkpoints, not standalone models and not deployment-ready
releases. Loading any checkpoint requires authorized access to the upstream
`google/t5gemma-2-4b-4b` base model and compliance with its applicable terms.

## Terms of use

The adapters are model derivatives subject to the Gemma Terms of Use, including
their use restrictions and the incorporated Gemma Prohibited Use Policy. By
downloading, using, modifying, or redistributing these adapters, recipients must
comply with those terms and applicable law. Copies of the terms and prohibited-use
policy retrieved from Google's official pages at publication time are included in
this repository, together with the required `NOTICE` file.

## Repository layout

Each loadable checkpoint is stored as:

```text
<experiment-family>/<checkpoint-optstep-N>/
  adapter_config.json
  adapter_model.safetensors
  MODIFIED_NOTICE.md
```

`manifest.jsonl` and `SHA256SUMS` provide the original artifact path, byte size,
and SHA-256 digest for every published file.

## Scope and safety

The public bundle intentionally contains only adapter weights and adapter
configuration files. It excludes:

- optimizer/RNG/training-resume state (`training_state.pt`)
- run contracts and metadata that reference private holdback material
- raw training or evaluation data
- predictions, generations, logs, and API harvests
- private holdback files and secret material
- redundant tokenizer copies (load the tokenizer from the base model)

All 45 adapter-weight files have distinct SHA-256 digests. The bundle was scanned
for credential-shaped tokens before publication.

## Important interpretation note

Checkpoint names describe their originating experiment and optimizer step; they
do not imply model selection, quality ranking, or promotion. In particular, the
typed-C2 VeRPO pilot checkpoints were sealed as non-promoted research pilots and
must not be represented as selected or production models.

## Loading an adapter

Download one checkpoint directory, then load it with PEFT on top of the upstream
base model. A typical local workflow is:

```python
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

base_id = "google/t5gemma-2-4b-4b"
checkpoint_dir = "/path/to/<experiment-family>/<checkpoint-optstep-N>"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(base_model, checkpoint_dir)
```

Exact experiment settings and evaluation evidence are maintained separately from
this public model-only repository so private protocol material cannot be detached
from its access controls.
