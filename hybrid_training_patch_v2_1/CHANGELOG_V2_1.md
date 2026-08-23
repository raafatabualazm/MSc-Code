# Hybrid Training Patch v2.1 Changelog

v2.1 addresses the two remaining load-bearing findings in the second external audit of v2.

## 1. Statistical functional gates

The v2 correct/permuted/null generation design is retained, but raw point-estimate gates are no longer sufficient.

The causal graph gate and matched-control RS-SFT gate now report and enforce:

- a minimum of 96 held-out tasks by default;
- at least eight discordant/effective task pairs;
- an exact one-sided paired sign test with `p <= 0.05`;
- exact McNemar terminology when the per-task metric is binary, as it is for the default pass@10 with ten samples;
- a 10,000-replicate task-paired bootstrap;
- a one-sided 95% lower confidence bound strictly above zero;
- a separately configurable pre-registered practical-effect floor.

The former default `1.0 pp` graph-permutation floor is now `0.0 pp`. This avoids inventing a noise threshold. A positive floor can still be supplied, but should be fixed from an external repeatability/noise study before inspecting the new checkpoint.

The RS-SFT gate keeps its pre-registered `+6 pp` matched-control effect requirement and now also requires the paired exact test, bootstrap confidence bound, task-count floor, and discordant-pair floor. A scalar historical baseline remains supported but is explicitly reported as weaker because it cannot carry task pairing or baseline uncertainty.

The bootstrap resamples held-out tasks. It does not claim to replace multi-seed generation replication for final paper estimates.

## 2. Qwen attention+MLP LoRA

New Qwen runs now target:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

The target set is recorded as the architecture field:

```text
GRAPH_QWEN_LORA_TARGETS=attention_mlp
```

`attention` remains available only for reproducing historical checkpoints.

### Historical Regions16/checkpoint migration

Old Qwen provenance did not record the target set because the implementation was unconditionally attention-only. v2.1 deterministically binds such provenance to `attention`.

A historical attention-only checkpoint can be supplied as `--initial_checkpoint` for a newly trained Stage 1. The loader may then permit only the newly introduced `gate_proj`, `up_proj`, and `down_proj` LoRA A/B tensors to be absent. Every new LoRA-B tensor must be exactly zero before loading is accepted, preserving the original policy at initialization. Any unrelated missing trainable tensor still aborts.

Supplying an old attention-only checkpoint directly as `--stage1_checkpoint` is rejected because no Stage-1 training would occur to learn the new MLP adapters.

DoRA expansion is intentionally not supported by this migration path.

## 3. Tests and contracts

New regression tests verify that:

- a one-task pass@10 flip on 96 tasks cannot pass the statistical gate;
- a consistent paired effect can pass the exact-test and bootstrap controls;
- Qwen defaults to attention+MLP targets;
- historical `attention` mode remains reproducible;
- the attention-to-attention+MLP migration accepts only zero-output MLP LoRA additions;
- undeclared trainable-checkpoint expansion still fails;
- non-Qwen architectures are not forced to provide a Qwen-specific target contract.

The packaged CPU-only suite contains 31 tests.

## Scope unchanged

v2.1 does not claim that SFT/RL redesign solves the `>=200`-instruction or approximately `>=1000`-character one-shot collapse. Long functions remain routed to a separate bridge/long-function research track.
