# Graph Encoder-Decoder Strategy

## Motivation

This repository currently focuses on decoder-only SFT.

This document introduces a graph-aware encoder-decoder decompiler line.

## Core Idea

```text
Assembly -> Graph-aware encoder -> Decoder -> Canonical source
```

## Goals

- improve heterogeneous benchmark robustness
- preserve semantic structure
- reduce target ambiguity
- support CFG-aware training

## Planned Extensions

- CFG edge embeddings
- SSA reconstruction
- optimization-level prediction
- execution-guided repair
- IR intermediate supervision

## Expected Benefits

- higher compile rates
- stronger pass@k
- improved semantic consistency
- reduced stylistic overfitting
