# Validation Report — Hybrid Training Patch v2.1

## Static validation

- All packaged Python sources parse and compile.
- The installer parses every overlay Python file before copying it.
- The patch introduces no SciPy dependency; exact binomial tails and paired bootstrap calculations use the Python standard library.

## Regression tests

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Result:

```text
Ran 31 tests
OK
```

The new Critique-2 tests cover:

- exact one-sided paired-test calculations;
- rejection of a one-task/noise-level gate result;
- acceptance of a consistent task-paired effect;
- Qwen attention+MLP target resolution;
- historical attention-only reproduction;
- zero-output attention-to-MLP LoRA migration;
- explicit allowed-missing checkpoint keys;
- rejection of undeclared missing trainables;
- non-Qwen architecture compatibility.

The original v2 tests for Phase-0 overlap blocking, neutral contracts, hidden test replay, diagnostic redaction, 50/50 RS-SFT mixing, functional graph ablations, checkpoint provenance, verified-only RL anchoring, and orchestration ordering continue to pass.

## Statistical implementation boundary

The functional gate now uses:

- task-paired exact direction testing;
- exact McNemar interpretation for binary task metrics;
- task-paired percentile bootstrap confidence intervals;
- minimum held-out-row and discordant-pair floors;
- separately declared practical-effect thresholds.

The bootstrap quantifies finite held-out-task uncertainty for one generated candidate pool. It does not estimate between-seed generation variance. Multi-seed replication remains required for final confirmatory claims.

## Qwen migration boundary

The compatibility path was validated with synthetic PyTorch modules and checkpoint contracts. It has not been executed here against the real Qwen3-8B PEFT module tree or the real Regions16 checkpoint. The implementation checks the actual model parameter names at runtime and aborts if it cannot identify the expected missing MLP LoRA tensors or if any new LoRA-B tensor is nonzero.

## Not executed in this environment

- Qwen3-8B or GraphCodeBERT model loading;
- GPU forward/backward training;
- real Regions16 checkpoint migration;
- real Dart corpus replay;
- correct/permuted/null generation over the frozen benchmark;
- frontier Responses or Batch API calls;
- empirical RS-SFT improvement over the matched gold-only control;
- multi-seed confirmatory statistics.

The archive therefore validates implementation structure and CPU-only controls, not the empirical hypothesis.
