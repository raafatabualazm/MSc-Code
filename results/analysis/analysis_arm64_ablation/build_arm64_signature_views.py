# -*- coding: utf-8 -*-
"""Build the ARM64 signature-ablation input views.

Cross-ISA replication of the signature decomposition, on 343 held-out functions
extracted from real, obfuscated Flutter release APKs (arm64-v8a).

The x86 measurement audit could not run this ablation: its 175 gold functions are
already named ``fn0``, so there is no semantic name left to remove. The ARM64
corpus still carries the original names, which makes it the only corpus in the
project where the full comparator -> typed-opaque -> name-only -> none ladder is
measurable. It is also the only one built from production binaries.

Views emitted (``prompt_signature_mode`` is consumed by
scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py):

  exact          full semantic signature.  THE EXISTING 5.54% pass@10 BASELINE --
                 i.e. the published ARM64 number was measured with the name supplied.
  typed_opaque   fn0 + return/parameter TYPES only; parameter names -> p0,p1,...
                 Requires a consistent rename across signature, gold source and
                 tests, because tests bind the function by name.
  name_only      semantic name kept, typed signature withheld.
  none           neither name nor signature.

Only ``typed_opaque`` rewrites record content; ``name_only`` and ``none`` are
pure flags on otherwise untouched rows, so any difference they produce cannot be
an artefact of rewriting.

Usage:
    python build_arm64_signature_views.py --eval <flutter_eval_graphv2.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

NEUTRAL_NAME = "fn0"
VIEWS = ("exact", "typed_opaque", "name_only", "none")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_params(param_text: str) -> list[str]:
    """Split a Dart parameter list on top-level commas only.

    Naive comma splitting counts separators inside generic type arguments --
    ``Map<String, int> perServing, int servings`` is two parameters, not three.
    Depth tracking over <>, (), [] and {} is required for correct arity.
    """
    out: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in param_text:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        out.append(tail)
    return [p for p in out if p]


def parse_signature(signature: str) -> tuple[str, str, list[str]]:
    """-> (return_type, function_name, [parameter declarations])."""
    match = re.fullmatch(r"\s*(.+?)\s+([A-Za-z_]\w*)\s*\((.*)\)\s*", signature, re.S)
    if match is None:
        raise ValueError(f"signature outside expected grammar: {signature!r}")
    return (
        re.sub(r"\s+", " ", match.group(1).strip()),
        match.group(2),
        split_params(match.group(3)),
    )


def parameter_type(declaration: str) -> str:
    """Recover the declared type, dropping the parameter's own name.

    Handles trailing defaults and named/optional markers. The last whitespace-
    separated token of a declaration is the identifier; everything before it is
    the type (which may itself contain spaces, e.g. ``Map<String, int>``).
    """
    value = declaration.strip()
    value = re.sub(r"^(required)\s+", "", value)
    value = re.split(r"=", value, maxsplit=1)[0].strip()
    parts = value.rsplit(" ", 1)
    if len(parts) == 1:
        return parts[0].strip()
    return parts[0].strip()


def typed_opaque_signature(signature: str) -> tuple[str, dict]:
    return_type, _name, params = parse_signature(signature)
    typed = [f"{parameter_type(p)} p{i}" for i, p in enumerate(params)]
    rendered = f"{return_type} {NEUTRAL_NAME}({', '.join(typed)})"
    return rendered, {
        "return_type": return_type,
        "arity": len(params),
        "parameter_types": [parameter_type(p) for p in params],
    }


def rename_function(text: str, old: str, new: str) -> str:
    """Whole-identifier rename; will not touch substrings of longer names."""
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def build_view(record: dict, view: str) -> dict:
    row = dict(record)
    row["prompt_signature_mode"] = "exact" if view == "typed_opaque" else view

    if view in ("exact", "name_only", "none"):
        # Untouched content; the mode flag alone drives the difference.
        return row

    original = record.get("function") or record.get("camel_case_function_name")
    signature = record.get("dart_function_signature")
    if not original or not signature:
        raise ValueError(f"row {record.get('flutter_sample_id')!r} lacks name/signature")

    rendered, diagnostics = typed_opaque_signature(signature)
    # The rename must be consistent across all three surfaces: the prompt
    # signature, the gold target, and the acceptance tests (which bind the
    # function by name via `final candidate = <name>;`). An inconsistent rename
    # would fail every task for a reason unrelated to the ablation.
    row["dart_function_signature"] = rendered
    row["function"] = NEUTRAL_NAME
    row["camel_case_function_name"] = NEUTRAL_NAME
    row["dart_source"] = rename_function(record["dart_source"], original, NEUTRAL_NAME)
    row["tests"] = rename_function(record["tests"], original, NEUTRAL_NAME)
    row["_ablation_diagnostics"] = diagnostics
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, help="flutter_eval_graphv2.jsonl")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.eval).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema": "arm64-signature-ablation-input-view-v1",
        "source": {"path": str(args.eval), "rows": len(rows),
                   "sha256": hashlib.sha256(Path(args.eval).read_bytes()).hexdigest()},
        "views": {},
    }

    for view in VIEWS:
        built, failures = [], []
        for record in rows:
            try:
                built.append(build_view(record, view))
            except Exception as exc:                      # noqa: BLE001
                failures.append({"id": record.get("flutter_sample_id"), "error": str(exc)})
        path = out_dir / f"flutter_eval_graphv2.{view}.jsonl"
        payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in built) + "\n"
        path.write_text(payload, encoding="utf-8")

        entry = {
            "path": str(path), "rows": len(built), "failures": failures,
            "sha256": sha256_text(payload),
        }
        if view == "typed_opaque":
            arities: dict[int, int] = {}
            returns: dict[str, int] = {}
            for r in built:
                d = r.get("_ablation_diagnostics", {})
                arities[d.get("arity", -1)] = arities.get(d.get("arity", -1), 0) + 1
                returns[d.get("return_type", "?")] = returns.get(d.get("return_type", "?"), 0) + 1
            entry["arity_histogram"] = dict(sorted(arities.items()))
            entry["return_type_histogram"] = dict(sorted(returns.items(), key=lambda kv: -kv[1]))
        manifest["views"][view] = entry
        print(f"{view:<14} rows={len(built):<5} failures={len(failures):<4} sha={entry['sha256'][:16]}")

    (out_dir / "input_view_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote {out_dir/'input_view_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
