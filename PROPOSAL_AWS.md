# AWS Research Credits — Project Proposal (Pilot)

## Project Title
**Compositional Neural Decompilation of Hardened Flutter/Dart AOT Binaries for Mobile Application Security**

*Alternative titles:*
- Breaking the Length Barrier in Neural Decompilation: Recovering Functional Dart from Obfuscated ARM64 Flutter Binaries
- Flutter-Eval: A Re-Executability Benchmark and Compositional Neural Decompiler for Production Dart AOT Binaries

---

## Applicant
**Raafat Abualazm** — MSc, Computer Engineering, Cairo University (expected 2026).
Background: offensive security / reverse engineering; binary analysis.
Prior work on this line: SANER 2026 ERA (accepted); TOSEM (in review).
Code: https://github.com/raafatabualazm/MSc-Code

---

## 1. Summary
Flutter apps ship as **ahead-of-time (AOT) compiled, ARM64, obfuscated, stripped**
native libraries (`libapp.so`), and no published neural decompiler targets
Dart/Flutter — existing tools (Blutter, unflutter, darter) recover only structure
(names, call edges, strings), never runnable source. This **pilot** funds the
decisive core of a larger program: a controlled study of how to recover
**functional** Dart from these hardened binaries, and a demonstration that
**compositional decompilation** — using the recovered control-flow graph (CFG) to
decompile a function's blocks and loop bodies independently and recompose them —
breaks the empirically measured **~200-instruction capability cliff** above which
all current models fail. All experiments run on a single AWS **g7e.12xlarge**
instance against a corpus we have **already built** (real obfuscated ARM64 Flutter
functions with executable unit tests), so the entire award goes to compute, not
data collection.

## 2. Problem and Prior Finding
Neural decompilation is mature for C/x86-64 (LLM4Decompile, Nova, CodeInverter)
but absent for Dart and the Flutter AOT toolchain that ships to real devices. Our
prior empirical study (TOSEM, in review) established two motivating results:
1. **A capability cliff.** Assembly length is the dominant difficulty factor:
   functions under 50 instructions are ~48% solvable; at 100–200, 18%; above 200,
   ~6%; above 500, **0%**. Monolithic decoding collapses on long functions.
2. **pass@k must be the primary metric** — surface metrics (CodeBLEU, compile
   rate) diverge from functional correctness, so we evaluate by compiling and
   executing candidates against unit tests (unbiased pass@k).

The cliff is the barrier between "toy" and "real," because production functions
are long. The CFG is a natural decomposition engine for crossing it.

## 3. Pilot Objective (single, sharp)
On an already-built, **leak-free, length-stratified** held-out set of real
obfuscated ARM64 Flutter functions (each test-equipped, so pass@k is measurable),
deliver two results:
- **(a) An architecture finding:** a controlled ablation determining which channel
  carries the assembly into the decoder — raw token text vs. the graph
  representation — fixing the model design and truncation strategy.
- **(b) The core technique:** evidence that **compositional** decoding lifts
  pass@k in the **>200-instruction** region where monolithic decoding scores ~0%,
  via a length-stratified monolithic-vs-compositional comparison.

*Guardrails:* strictly clean held-out (enforced in tooling), pass@k as primary,
honest scoping of what the Flutter hardening pipeline does and does not obfuscate
(it strips/renames symbols; it does not apply instruction-level obfuscation).

*Future work this pilot enables (not in this budget):* scaling to real
application semantics, a downstream security task (vulnerability/privacy
analysis), and the public **Flutter-Eval** benchmark — a separate, larger effort.

## 4. Approach
A GraphCodeBERT + graph-neural-network encoder over the recovered CFG conditions a
9B-parameter Dart decoder (LoRA fine-tuning); supervised fine-tuning is followed
by **execution-grounded reinforcement learning** with a pass@k objective
(candidates are compiled and run against unit tests; reward = functional
correctness). A custom ARM64 basic-block recovery pass (validated at 100%
intra-procedural branch resolution on the corpus) provides the CFG that drives
both graph conditioning and compositional decomposition.

## 5. Why AWS / Why g7e.12xlarge
The pilot fits a **single g7e.12xlarge** instance end-to-end, which is exactly why
it is cost-efficient:
- **Its two GPUs (NVIDIA RTX Pro 6000, 96 GB GDDR7 each; 192 GB aggregate) run
  the architecture ablation arms in parallel, two at a time**, collapsing a serial
  study into two short waves — the central reason a multi-GPU instance is the
  right tool. The 96 GB per GPU also lets the monolithic/text baseline use a
  generous input context, so the cliff we measure reflects model capability rather
  than prompt truncation.
- **The same instance's CPU cores host the sandboxed compile-and-execute pass@k
  harness** (thousands of candidate Dart programs compiled and run in isolated
  processes), so training and evaluation share one node — no separate fleet.
- **Amazon S3** holds the CFG-enriched datasets and model checkpoints (small,
  <100 GB) for reproducibility.

This single-instance, GPU-parallel design is what makes a rigorous study feasible
within a modest credit award; the workload genuinely needs a multi-GPU cloud
instance (9B model + sampling-heavy RL on long inputs) and cannot run on local or
free-tier hardware.

## 6. Budget — $550
All compute on one g7e.12xlarge instance; usage in instance-hours.

| Phase | Workload | Instance-hours |
|---|---|---|
| Architecture ablation (4 arms, 2 waves across the 2 GPUs) | SFT + eval | ~12 |
| Compositional + pass@k-RL on the winning design | RL train + eval | ~14 |
| Monolithic-vs-compositional length-stratified evaluation | inference + sandboxed tests | ~6 |
| Iteration / re-runs buffer | — | ~6 |
| **Total** | | **~38 instance-hours** |

At current g7e.12xlarge on-demand pricing this totals **≈ $550** (Spot pricing
extends it further); a small Amazon S3 allocation (<100 GB-months, a few dollars)
covers datasets and checkpoints. The corpus is already collected, so no
data-acquisition cost is incurred.

## 7. Outcomes and Impact
- A peer-reviewed software-engineering submission on the compositional
  cliff-breaking technique (the core result of this pilot).
- Open release of the ARM64 CFG-recovery and pass@k evaluation tooling and the
  trained checkpoints for reproducibility.
- A validated foundation for the broader program: the first steps toward a
  neural, functional, re-executability-benchmarked decompiler for the hardened
  Flutter binaries that actually ship — supporting downstream mobile-security
  analysis at scale.

## 8. Data Management and Responsible Use
This is **defensive** security research. The pilot corpus uses functions with
**synthetic (LLM-generated) semantics** compiled through the genuine Flutter
release pipeline, so the *binary distribution is real* while the *content carries
no third party's intellectual property*. Any future extension to real apps will
use only authorized targets (our own builds, open-source, or consented apps),
follow responsible disclosure, and comply with the AWS Acceptable Use Policy. All
datasets, splits, and configs are versioned with strict train/eval separation
enforced in tooling.
