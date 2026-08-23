# Source Provenance

The following packaged files match SHA-256 values recorded directly in `results/run_provenance.json`:

| Packaged path | Recorded SHA-256 |
|---|---|
| `scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py` | `6234f82bd4c64c29a561160374888d1bc6916af8d7d7368e30ba97cb5f237e13` |
| `models/hierarchical_graph_encoder_antigravity.py` | `0d9ed811fd3e2793d8d21a003ced9cff67cecfe65cf03a33bb611ec44af6f7db` |
| `models/graphcodebert_tensor_builder.py` | `a324bba2c3a642176404fba513c386e6952b4c6f04d26e9be01732ed36900ed5` |
| `models/pyg_cfg_dataset.py` | `5cf35b3c2d446e3e4444a833d6d0a39c6ddd366f2375a8b866706b4cd322edb3` |
| `scripts/data/cfg_extractor.py` | `daebbbfa7ac53fed9104e66396bc861bc837a8cea5a948548204d34439ee553c` |
| `scripts/data/dfg_extractor.py` | `603c052e8a79e7f6f689e97acdfc9c87245505b4fbf497bc2c49c2343fb0ed12` |

`scripts/run_arm64_graphv21_study.py` matches the separately verified runner hash `31e378614ce07c01dfef24db3f4f3f077ce0d4a1c0165fb7777d17ce3a9a3ff6`.

The launch script was captured with the run handoff. `run_graphv2_followups.py`, `build_graph_v2_jsonl.py`, and the analysis utility are included as useful reference helpers, but their hashes were not listed in the model's saved run provenance. Treat the captured environment in `results/run_provenance.json` as authoritative if helper defaults differ.
