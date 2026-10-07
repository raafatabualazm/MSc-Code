# Factual and Methodological Audit Resolution

This ledger records how the public draft and artifact address the July 2026 reviews.

| Review issue | Resolution |
|---|---|
| MultiPL-E incorrectly described as including Dart | Corrected: MultiPL-E translated into 18 additional languages but did not include Dart; HumanEval-Dart is a separate adaptation. |
| Fabricated/conflated Qwen3 author names | Bibliography now uses `Yang, An and others` for the Qwen3 technical report. |
| Nonexistent `Qwen/Qwen3-8B-Base` path | Corrected in the paper, thesis, runner registry, and commands to `Qwen/Qwen3-8B`. |
| 126 versus 154 looked like unexplained filtering | Clarified that these are distinct corpora with different schemas and no filename overlap. Primary functional conclusions use the 154-task corpus. |
| `pass@1 > compile@1` anomaly | Main compile results now use `jit_tests`, the same candidate-plus-tests `dart run` path as pass@k. Pass is nested under compile. Strict AOT remains a diagnostic. |
| Dual compile definitions obscured the main claim | The paper uses aligned 154-task compile in the main body and moves the 126-row standalone AOT view to an appendix. |
| GLM-5.2 said to match or clearly beat the union | Corrected to 44 versus 43 tasks and explicitly not interpreted as meaningful. |
| `gpt-chat-latest` is a moving alias | Reported as a July 8, 2026 snapshot; raw outputs and metadata are archived. |
| Sonnet temperature/top-p fairness | Metadata records requested controls accepted through OpenRouter. The paper no longer claims provider-side controls were independently verified. |
| Hosted API failures/empty responses omitted | All ten requests are retained. Blank slots are 115/1,540 for GPT-5.5, 765 for Sonnet, 1,233 for DeepSeek, and 1,188 for GLM. GPT-5.5/Sonnet have zero caught-error rows; DeepSeek has three and GLM one. |
| Canonical GPT row (0.7013) versus ablation full assembly (0.7143) | Explicitly identified as separate stochastic pools with different output caps/settings; the two-task difference is not treated as a prompt improvement. |
| Approximate 2,048-token cap overclaimed | Reframed as a character-estimated assembly cap affecting API token accounting on 46 rows, not an exact total-token control or instruction threshold. |
| Textual topology result generalized to small models | Corrected: topology-only serialization is lossy for that frontier prompt; G3 still consumes block instructions and edges through learned compression. |
| DFG benefit claimed without isolation | Paper now states DFG contribution is unresolved and requires a graph-only controlled ablation. |
| Causal claims from one seed | Reworded as fixed-run observations; task-level intervals and McNemar tests are not training-seed uncertainty. |
| Seventeen arms but fourteen table rows | Paper names the three auxiliary pools and scopes them to union/reranking analysis. |
| Artifact promised but unavailable | Code/results are released here; adapters are public at a pinned Hugging Face revision with LFS hashes. |
| Pod environment provenance | Added the captured working CUDA 12.8/PyTorch 2.8 `requirements.txt`; machine-local wheel/source paths are identified as a portability caveat. |
| Missing repair-arm commands | Retained as an explicit command-provenance gap instead of reconstructing or inventing invocations. |
| Thin ethics discussion | Expanded to define release contents, excluded target data, authorization boundaries, false-confidence risks, and dual use. |

## Claims Deliberately Not Made

- No state-of-the-art claim over hosted frontier models.
- No production Flutter application decompilation claim.
- No proof that DFG edges independently improve G3.
- No repeated-seed stability claim.
- No claim that the 17-arm union is a budget-matched deployable system.
- No claim that provider-side stochastic sampling can be recreated exactly.
