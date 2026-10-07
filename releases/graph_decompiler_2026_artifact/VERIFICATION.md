# Release Verification

The release tree was audited on 2026-07-10 before tagging.

## Code Checks

- Every packaged Python source file parses successfully with Python's `ast` parser.
- `scripts/training/grpo_selfcheck.py`: all 9 GRPO algorithm checks passed.
- `scripts/data/test_graph_preprocessing_fixes.py`: all 56 CFG/DFG and tensor-builder checks passed.
- Training and evaluation CLIs load and expose their argument parsers.

## Environment Records

- `environment/requirements.txt` captures the working CUDA 12.8/PyTorch 2.8
  training-pod environment.
- `environment/requirements-verification.txt` and
  `environment/env_manifest_local_verification.txt` capture the separate local
  verification environment.
- Machine-local `file://` dependencies in the training snapshot are documented
  rather than presented as portable package-index requirements.

## Artifact Checks

- 123 JSON files parsed successfully.
- 6 uncompressed JSONL files and 2 compressed JSONL files parsed row by row.
- All 17 specialized compile-prediction pools contain 126 rows.
- All 17 specialized pass-prediction pools contain 154 rows.
- The compressed synthetic pool contains 1,726 rows.
- The compressed ARM64 pool contains 1,714 rows.
- Provider-formatted API-token and concrete Azure OpenAI endpoint scans returned no matches.
- No packaged file exceeds GitHub's 100 MB per-file limit; the largest file is below 9 MB.

Nine historical `*_at_k.json` files originally combined progress output with a
final JSON object. Their complete console streams are retained as adjacent
`.log` files, while the `.json` files now contain only the parsed metric object.

## Document Checks

- The paper was rebuilt with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
- The resulting 19-page PDF was rendered to images and inspected for clipping,
  overlap, broken tables, and unresolved references.

The generated release inventory is in `manifests/artifact_inventory.json`.
SHA-256 hashes for every file other than the hash list itself are in
`manifests/files_sha256.csv`.
