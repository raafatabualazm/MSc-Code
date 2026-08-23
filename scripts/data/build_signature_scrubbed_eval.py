"""Build a neutral-name, signature-scrubbed Dart AOT evaluation dataset.

This script supports two scientifically distinct uses:

* ``existing_ablation`` rebuilds an existing benchmark, such as the 154-task
  HumanEval-Dart set. It measures sensitivity to signature/name cues but is not
  a fresh holdout.
* ``fresh_holdout`` consumes newly generated or newly curated Dart tasks after
  the architecture is frozen. A freeze manifest is mandatory.

Every target is renamed to an opaque target name (``candidate`` by default for
backward compatibility) before AOT compilation. The public prompt receives
only that neutral name and the assembly/graph. Return type, parameter types,
source, tests, and the original semantic name are withheld.
The private output retains hidden tests and the evaluator-only signature.

Run on Linux with Dart and GDB available. Graph-v2 is rebuilt from the newly
compiled assembly so the original semantic symbol cannot survive in the graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.build_graph_v2_jsonl import build_record, extractor_sha256
from scripts.data.build_humaneval_synthetic_pool import (
    ensure_entrypoint,
    infer_function,
    infer_signature,
)
from scripts.data.rebuild_sft_assembly import (
    compile_and_disassemble as compile_aot_snapshot_and_disassemble,
)
from scripts.evaluation.rerank_predictions_antigravity import (
    _resolve_dart_binary,
    strip_main_and_imports,
    validate_dart_binary,
)


PROTOCOL_SCHEMA = "dart-signature-scrubbed-v2"
PROTOCOL_SCHEMAS = (PROTOCOL_SCHEMA, "dart-signature-scrubbed-v3")
TARGET_NAME = "candidate"
PUBLIC_SIGNATURE_MODES = ("name_only", "neutral_exact")
# Per-task fingerprints of withheld secrets. They stay in the private sidecar for
# auditing but must not ship in the public policy input: an unsalted sha256 of a
# benchmark function name is dictionary-recoverable.
PUBLIC_PROTOCOL_SECRET_KEYS = {
    "semantic_function_name_sha256",
    "original_source_sha256",
}
TASK_ID_PREFIXES = {"name_only": "sigless", "neutral_exact": "sigtyped"}


def validate_target_name(target_name: str) -> str:
    """Return a safe Dart identifier for the neutral callable contract."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target_name):
        raise ValueError(
            "target_name must be a non-empty ASCII Dart identifier "
            "([A-Za-z_][A-Za-z0-9_]*)"
        )
    if target_name == "main":
        raise ValueError("target_name must not be main (reserved for the test driver)")
    return target_name


def validate_protocol_schema(protocol_schema: str) -> str:
    if protocol_schema not in PROTOCOL_SCHEMAS:
        raise ValueError(f"unsupported protocol_schema: {protocol_schema!r}")
    return protocol_schema


def effective_id_salt(
    id_salt: str,
    public_signature_mode: str,
    target_name: str = TARGET_NAME,
) -> str:
    """Bind opaque IDs to the arm while preserving the v2 default digest."""
    validate_target_name(target_name)
    arm_salt = f"{id_salt}|{public_signature_mode}"
    if target_name != TARGET_NAME:
        arm_salt += f"|target_name={target_name}"
    return arm_salt


def id_salt_summary_fields(arm_salt: str, *, redact: bool = False) -> dict[str, str]:
    """Serialize salt provenance without disclosing an optional secret salt."""
    if redact:
        return {
            "effective_id_salt_sha256": hashlib.sha256(
                arm_salt.encode("utf-8")
            ).hexdigest()
        }
    return {"effective_id_salt": arm_salt}


def protocol_summary_fields(
    protocol_schema: str, arm_salt: str, *, redact_id_salt: bool = False
) -> dict[str, str]:
    """Return schema/salt summary fields through one redaction boundary."""
    return {
        "schema": validate_protocol_schema(protocol_schema),
        **id_salt_summary_fields(arm_salt, redact=redact_id_salt),
    }


def rename_frozen_toolchain_stamp(
    *,
    frozen_dart_version: str | None = None,
    frozen_gdb_version: str | None = None,
    frozen_toolchain_version: str | None = None,
) -> dict[str, str]:
    """Build provenance for a frozen dump without claiming live observation."""
    stamp = {
        "assembly_derivation": "text_rename_of_frozen_benchmark_assembly",
    }
    asserted = {
        "asserted_frozen_dart_version": frozen_dart_version,
        "asserted_frozen_gdb_version": frozen_gdb_version,
        "asserted_frozen_toolchain_version": frozen_toolchain_version,
    }
    stamp.update({key: value for key, value in asserted.items() if value})
    return stamp


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_identifier(text: str, old: str, new: str) -> str:
    return re.sub(rf"\b{re.escape(old)}\b", new, text)


def rewrite_tests(
    tests: str, original_name: str, target_name: str = TARGET_NAME
) -> str:
    """Retarget a HumanEval-style harness without shadowing the target."""
    validate_target_name(target_name)
    alias_pattern = re.compile(
        rf"\bfinal\s+([A-Za-z_]\w*)\s*=\s*{re.escape(original_name)}\s*;"
    )
    match = alias_pattern.search(tests)
    if match and match.group(1) == target_name:
        tests = re.sub(rf"\b{re.escape(target_name)}\s*\(", "implementation(", tests)
        tests = alias_pattern.sub(
            f"final implementation = {target_name};", tests, count=1
        )
    elif match:
        alias = match.group(1)
        tests = alias_pattern.sub(f"final {alias} = {target_name};", tests, count=1)

    return replace_identifier(tests, original_name, target_name)


def neutralize_param_names(
    signature: str, target_name: str = TARGET_NAME
) -> tuple[str, bool]:
    """Rename parameter identifiers to a, b, c... while keeping types.

    Parameter names in HumanEval-style signatures are semantic hints
    ('numbers', 'threshold'); the neutral_exact arm must expose types and
    arity only. Returns (signature, fully_neutralized). Signatures with
    optional/named groups or defaults are left unchanged (flag False) —
    the stub-compile gate downstream makes any corruption loud.
    """
    validate_target_name(target_name)
    m = re.search(rf"^(.*?\b{re.escape(target_name)})\s*\((.*)\)\s*$", signature, re.S)
    if not m:
        return signature, False
    head, params = m.group(1), m.group(2)
    if not params.strip():
        return signature, True
    if any(ch in params for ch in "[]{}="):
        return signature, False
    chunks: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(params):
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        elif ch == "," and depth == 0:
            chunks.append(params[start:i])
            start = i + 1
    chunks.append(params[start:])
    if len(chunks) > 16:
        return signature, False
    neutral: list[str] = []
    for idx, chunk in enumerate(chunks):
        cm = re.match(r"^\s*(.*?[\w>?\]])\s+([A-Za-z_]\w*)\s*$", chunk, re.S)
        if not cm:
            return signature, False
        neutral.append(f"{cm.group(1)} {'abcdefghijklmnop'[idx]}")
    return f"{head.strip()}({', '.join(neutral)})", True


def _rename_symbol_contexts(text: str, name: str, replacement: str) -> str:
    """Rename one identifier in GDB-dump symbol contexts only.

    A blanket word-boundary rename would corrupt instruction text whenever the
    name collides with an x86 token (a task named ``add`` would turn
    ``add rax,rax`` into ``candidate rax,rax``). Symbol contexts are:
    ``<name...>`` annotations, ``function name`` dump headers, ``"name"``
    quoted regex headers, and ``name(`` / ``name.<closure>`` synopsis forms.
    """
    esc = re.escape(name)
    text = re.sub(rf"<{esc}(?=[+>.\-])", f"<{replacement}", text)
    text = re.sub(rf"(function\s+){esc}(?![A-Za-z0-9_])", rf"\g<1>{replacement}", text)
    text = re.sub(rf'"{esc}"', f'"{replacement}"', text)
    text = re.sub(rf"(?<![A-Za-z0-9_<.]){esc}(?=\s*[.(])", replacement, text)
    return text


def extract_source_function_names(function_source: str) -> list[str]:
    """Every function identifier declared in the (stripped) reference source.

    Helpers keep their semantic names in the frozen dump's synopsis and symbol
    annotations (``sortArray.<anonymous closure>``), which would leak task
    identity; they all get neutralized alongside the target.
    """
    names = []
    for match in re.finditer(
        r"(?m)^\s*(?:[A-Za-z_][\w<>,\[\]?]*\s+)+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:async\s*\*?\s*)?(?:\{|=>)",
        function_source,
    ):
        name = match.group(1)
        if name not in names and name not in {
            "main",
            "if",
            "for",
            "while",
            "switch",
            "catch",
        }:
            names.append(name)
    return names


def rename_assembly_symbols(
    frozen_assembly: str,
    original_name: str,
    helper_names: tuple[str, ...] = (),
    target_name: str = TARGET_NAME,
) -> str:
    validate_target_name(target_name)
    renamed = _rename_symbol_contexts(frozen_assembly, original_name, target_name)
    neutral_helpers = tuple(h for h in helper_names if h != original_name)
    for i, helper in enumerate(neutral_helpers):
        replacement = f"helper{i + 1}"
        # Nested Dart helpers are emitted in GDB symbol contexts as
        # ``target.helper``. Rename that exact qualified symbol before the
        # unqualified helper pass; the latter intentionally does not cross a
        # dot because doing so could rewrite operands or unrelated symbols.
        renamed = _rename_symbol_contexts(
            renamed,
            f"{target_name}.{helper}",
            f"{target_name}.{replacement}",
        )
        renamed = _rename_symbol_contexts(renamed, helper, replacement)
    renamed = re.sub(r"file:///\S+", "file:///scrubbed/program.dart", renamed)
    renamed = re.sub(r"/tmp/\S+", "/tmp/scrubbed", renamed)
    residue_names = (
        original_name,
        *neutral_helpers,
        *(f"{target_name}.{helper}" for helper in neutral_helpers),
    )
    for name in residue_names:
        esc = re.escape(name)
        for residue_pattern in (
            rf"<{esc}(?![A-Za-z0-9_])",
            rf"function\s+{esc}(?![A-Za-z0-9_])",
            rf'"{esc}"',
            rf"(?<![A-Za-z0-9_<.]){esc}(?=\s*[.(])",
        ):
            residue = re.search(residue_pattern, renamed)
            if residue:
                raise ValueError(
                    f"symbol residue {residue.group(0)!r} survived assembly rename"
                )
    return renamed


def _neutral_task_id(
    source: str, salt: str, index: int, prefix: str = "sigless"
) -> str:
    digest = hashlib.sha256(f"{salt}\0{index}\0{source}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:12]}"


def prepare_scrubbed_row(
    row: dict[str, Any],
    *,
    index: int,
    benchmark_kind: str,
    id_salt: str,
    input_sha256: str,
    freeze_manifest_sha256: str | None,
    public_signature_mode: str = "name_only",
    target_name: str = TARGET_NAME,
    protocol_schema: str = PROTOCOL_SCHEMA,
) -> dict[str, Any]:
    if public_signature_mode not in PUBLIC_SIGNATURE_MODES:
        raise ValueError(
            f"unsupported public_signature_mode: {public_signature_mode!r}"
        )
    validate_target_name(target_name)
    validate_protocol_schema(protocol_schema)
    source = str(row.get("dart_source") or row.get("source") or "")
    tests = str(row.get("tests") or "")
    if not source.strip():
        raise ValueError("missing dart_source/source")
    if not tests.strip():
        raise ValueError("missing hidden tests")

    original_name = infer_function({**row, "dart_source": source, "tests": tests})
    original_signature = str(
        row.get("dart_function_signature")
        or row.get("signature")
        or infer_signature(source, original_name)
    ).strip()
    imports = sorted(set(re.findall(r"^import\s+.*;\s*$", source, re.MULTILINE)))
    function_source = strip_main_and_imports(source)
    neutral_body = replace_identifier(function_source, original_name, target_name)
    neutral_source = ("\n".join(imports) + "\n\n" if imports else "") + neutral_body
    neutral_source = ensure_entrypoint(neutral_source, target_name)
    neutral_signature = replace_identifier(
        original_signature, original_name, target_name
    )
    neutral_tests = rewrite_tests(tests, original_name, target_name)
    task_id = _neutral_task_id(
        neutral_source, id_salt, index, prefix=TASK_ID_PREFIXES[public_signature_mode]
    )
    # Consumed (and popped) by build_one for assembly derivation; never serialized.
    output_original_name = original_name
    if public_signature_mode == "neutral_exact":
        public_prompt_signature, params_neutralized = neutralize_param_names(
            neutral_signature, target_name
        )
        public_dart_function_signature = public_prompt_signature
        prompt_signature_mode = "exact"
        prompt_exposes = [
            "assembly_or_graph",
            "neutral_target_name",
            "typed_signature_neutral_name_neutral_params",
        ]
        prompt_withholds = [
            "parameter_names",
            "reference_source",
            "tests",
            "semantic_function_name",
        ]
    else:
        public_prompt_signature = ""
        params_neutralized = True
        public_dart_function_signature = ""
        prompt_signature_mode = "name_only"
        prompt_exposes = ["assembly_or_graph", "neutral_target_name"]
        prompt_withholds = [
            "return_type",
            "arity",
            "parameter_count",
            "parameter_types",
            "parameter_names",
            "reference_source",
            "tests",
            "semantic_function_name",
        ]

    semantic_name_sha256 = hashlib.sha256(
        original_name.lower().encode("utf-8")
    ).hexdigest()
    source_sha256 = hashlib.sha256(function_source.encode("utf-8")).hexdigest()

    output = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "assembly",
            "camel_case_function_name",
            "cfg",
            "dart_function_signature",
            "edges",
            "filename",
            "function",
            "function_signature",
            "graph_v2",
            "integrity",
            "name",
            "python_function_name",
            "python_source",
            "signature",
            "source",
            "task_id",
            "tests",
        }
    }
    output.update(
        {
            "filename": f"{task_id}.dart",
            "function": target_name,
            "camel_case_function_name": target_name,
            "python_function_name": "",
            "dart_function_signature": public_dart_function_signature,
            "prompt_signature_mode": prompt_signature_mode,
            "public_prompt_signature": public_prompt_signature,
            "evaluation_only_dart_function_signature": neutral_signature,
            "dart_source": neutral_source,
            "assembly": "",
            "lang": str(row.get("lang") or row.get("language") or "Dart"),
            "task_id": task_id,
            "tests": neutral_tests,
            "benchmark_protocol": {
                "schema": protocol_schema,
                "benchmark_kind": benchmark_kind,
                "fresh_holdout": benchmark_kind == "fresh_holdout",
                "public_signature_mode": public_signature_mode,
                "public_signature_params_neutralized": params_neutralized,
                "input_sha256": input_sha256,
                "freeze_manifest_sha256": freeze_manifest_sha256,
                "original_source_sha256": source_sha256,
                "semantic_function_name_sha256": semantic_name_sha256,
                "neutral_target_name": target_name,
                "prompt_exposes": prompt_exposes,
                "prompt_withholds": prompt_withholds,
            },
            "__original_name": output_original_name,
        }
    )
    return output


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    private_fields = {
        "dart_source",
        "evaluation_only_dart_function_signature",
        "tests",
    }
    visible = {key: value for key, value in row.items() if key not in private_fields}
    protocol = visible.get("benchmark_protocol")
    if isinstance(protocol, dict):
        public_protocol = {
            key: value
            for key, value in protocol.items()
            if key not in PUBLIC_PROTOCOL_SECRET_KEYS
        }
        assembly_build = public_protocol.get("assembly_build")
        if isinstance(assembly_build, dict):
            public_protocol["assembly_build"] = {
                key: value
                for key, value in assembly_build.items()
                if key != "frozen_assembly_sha256"
            }
        visible["benchmark_protocol"] = public_protocol
    return visible


def build_one(
    item: tuple[int, dict[str, Any]],
    *,
    benchmark_kind: str,
    id_salt: str,
    input_sha256: str,
    freeze_manifest_sha256: str | None,
    dart_bin: str,
    gdb_bin: str,
    timeout: int,
    max_block_instrs: int,
    max_dataflow_edges: int,
    graph_extractor_sha256: str,
    prepare_only: bool,
    public_signature_mode: str,
    toolchain_stamp: dict[str, str] | None,
    assembly_mode: str,
    target_name: str = TARGET_NAME,
    protocol_schema: str = PROTOCOL_SCHEMA,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    index, raw = item
    try:
        row = prepare_scrubbed_row(
            raw,
            index=index,
            benchmark_kind=benchmark_kind,
            id_salt=id_salt,
            input_sha256=input_sha256,
            freeze_manifest_sha256=freeze_manifest_sha256,
            public_signature_mode=public_signature_mode,
            target_name=target_name,
            protocol_schema=protocol_schema,
        )
        original_name = str(row.pop("__original_name", ""))
        if prepare_only:
            return index, row, None

        if assembly_mode == "rename_frozen":
            # Text-domain rename of the frozen benchmark disassembly. Dart AOT
            # register allocation is not bit-stable under a renamed/restructured
            # source program (observed: 87/154 tasks drift by regalloc alone), so
            # recompiling can never give byte-parity with the frozen comparator.
            # Renaming symbol strings in the frozen dump gives identity by
            # construction; the leak gate verifies no residue of the old name.
            frozen_assembly = str(raw.get("assembly") or "")
            if not frozen_assembly.strip():
                raise ValueError(
                    "rename_frozen requires frozen assembly on the input row"
                )
            source_text = str(raw.get("dart_source") or raw.get("source") or "")
            helper_names = tuple(
                n
                for n in extract_source_function_names(
                    strip_main_and_imports(source_text)
                )
                if n != original_name
            )
            row["assembly"] = rename_assembly_symbols(
                frozen_assembly,
                original_name,
                helper_names,
                target_name=target_name,
            )
            assembly_build = {
                "mode": "rename_frozen",
                "frozen_assembly_sha256": hashlib.sha256(
                    frozen_assembly.encode("utf-8")
                ).hexdigest(),
                "helper_symbols_renamed": len(helper_names),
            }
        else:
            # The legacy 154-task benchmark includes reference/oracle representation
            # mismatches. Requiring the reference to pass would change its denominator.
            # Use the project's pinned AOT-snapshot path. Dart 3.11 strips the ELF
            # produced by `dart compile exe`, while an AOT snapshot preserves the
            # vm:entry-point symbol required for deterministic GDB disassembly.
            row["assembly"], assembly_build = compile_aot_snapshot_and_disassemble(
                row,
                index=index,
                dart_bin=dart_bin,
                gdb_bin=gdb_bin,
                timeout=timeout,
                retries=2,
            )
        row["benchmark_protocol"]["assembly_build"] = dict(assembly_build)
        # Toolchain provenance: rebuilt evidence must be attributable to exact
        # compiler/disassembler/extractor versions (review finding F1/F2).
        row["benchmark_protocol"]["assembly_build"].update(toolchain_stamp or {})
        row["benchmark_protocol"]["assembly_build"][
            "extractor_sha256"
        ] = graph_extractor_sha256
        graph_row, _ = build_record(
            row,
            max_block_instrs=max_block_instrs,
            max_dataflow_edges=max_dataflow_edges,
            extractor_hash=graph_extractor_sha256,
        )
        return index, graph_row, None
    except Exception as exc:
        return (
            index,
            None,
            {
                "index": index,
                "task_id": raw.get("task_id"),
                "function": raw.get("function") or raw.get("camel_case_function_name"),
                "error": str(exc),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="Private evaluator JSONL"
    )
    parser.add_argument("--public_output", type=Path, default=None)
    parser.add_argument("--rejects", required=True, type=Path)
    parser.add_argument(
        "--benchmark_kind",
        required=True,
        choices=["existing_ablation", "fresh_holdout"],
    )
    parser.add_argument(
        "--protocol_schema",
        default=PROTOCOL_SCHEMA,
        choices=list(PROTOCOL_SCHEMAS),
        help="Protocol schema recorded in every row and the build summary",
    )
    parser.add_argument("--freeze_manifest", type=Path, default=None)
    parser.add_argument("--expected_rows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--dart_bin", default=None)
    parser.add_argument("--gdb_bin", default="gdb")
    parser.add_argument("--id_salt", default="dart-signature-scrubbed-v1")
    parser.add_argument(
        "--target_name",
        default=TARGET_NAME,
        help="Opaque Dart function name used in source, tests, signatures, assembly, and protocol metadata",
    )
    parser.add_argument(
        "--redact_effective_id_salt",
        "--redact_id_salt",
        action="store_true",
        help="Write only SHA-256(effective ID salt) to the build summary",
    )
    parser.add_argument(
        "--public_signature_mode",
        default="name_only",
        choices=list(PUBLIC_SIGNATURE_MODES),
        help=(
            "name_only: prompt gets only the neutral name (harsh arm). "
            "neutral_exact: prompt gets the exact typed signature with neutral "
            "function and parameter names (decomposes name-reliance from type/arity-reliance)"
        ),
    )
    parser.add_argument(
        "--shuffle_public_seed",
        type=int,
        default=None,
        help="If set, deterministically shuffle public row order so it cannot be aligned to the source benchmark by position",
    )
    parser.add_argument(
        "--assembly_mode",
        default="rebuild",
        choices=["rebuild", "rename_frozen"],
        help=(
            "rebuild: recompile the neutralized source (subject to AOT regalloc "
            "nondeterminism vs the frozen comparator). rename_frozen: text-rename "
            "symbols in the input row's frozen assembly for byte-parity"
        ),
    )
    parser.add_argument(
        "--frozen_dart_version",
        default=None,
        help="Dart version asserted for input frozen assembly (rename_frozen only)",
    )
    parser.add_argument(
        "--frozen_gdb_version",
        default=None,
        help="GDB version asserted for input frozen assembly (rename_frozen only)",
    )
    parser.add_argument(
        "--frozen_toolchain_version",
        default=None,
        help="Optional aggregate toolchain version asserted for input frozen assembly",
    )
    parser.add_argument("--max_block_instrs", type=int, default=20)
    parser.add_argument("--max_dataflow_edges", type=int, default=0)
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only rewrite rows; skip Dart/GDB and graph extraction",
    )
    args = parser.parse_args()

    try:
        target_name = validate_target_name(args.target_name)
    except ValueError as exc:
        parser.error(str(exc))
    frozen_version_args = (
        args.frozen_dart_version,
        args.frozen_gdb_version,
        args.frozen_toolchain_version,
    )
    if args.assembly_mode != "rename_frozen" and any(frozen_version_args):
        parser.error(
            "--frozen_*_version assertions require --assembly_mode rename_frozen"
        )

    if args.benchmark_kind == "fresh_holdout":
        if args.freeze_manifest is None or not args.freeze_manifest.is_file():
            raise SystemExit(
                "fresh_holdout requires an existing --freeze_manifest created before data generation"
            )
    freeze_hash = file_sha256(args.freeze_manifest) if args.freeze_manifest else None

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")
    rows = read_jsonl(args.input)
    expected_rows = args.expected_rows or len(rows)
    if len(rows) != expected_rows:
        raise SystemExit(f"expected {expected_rows} rows, found {len(rows)}")

    dart_bin = _resolve_dart_binary(args.dart_bin)
    toolchain_stamp: dict[str, str] = {}
    if not args.prepare_only and args.assembly_mode == "rebuild":
        validate_dart_binary(dart_bin)
        if not shutil.which(args.gdb_bin):
            raise SystemExit(f"gdb not found: {args.gdb_bin}")

        def _version_line(cmd: list[str]) -> str:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            text = (proc.stdout or proc.stderr or "").strip()
            return text.splitlines()[0] if text else "unknown"

        toolchain_stamp = {
            "dart_version": _version_line([dart_bin, "--version"]),
            "gdb_version": _version_line([args.gdb_bin, "--version"]),
        }
    elif not args.prepare_only:
        toolchain_stamp = rename_frozen_toolchain_stamp(
            frozen_dart_version=args.frozen_dart_version,
            frozen_gdb_version=args.frozen_gdb_version,
            frozen_toolchain_version=args.frozen_toolchain_version,
        )

    input_hash = file_sha256(args.input)
    graph_hash = extractor_sha256()
    arm_id_salt = effective_id_salt(
        args.id_salt, args.public_signature_mode, target_name
    )
    accepted: list[tuple[int, dict[str, Any]]] = []
    rejects: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                build_one,
                (index, row),
                benchmark_kind=args.benchmark_kind,
                id_salt=arm_id_salt,
                input_sha256=input_hash,
                freeze_manifest_sha256=freeze_hash,
                dart_bin=dart_bin,
                gdb_bin=args.gdb_bin,
                timeout=args.timeout,
                max_block_instrs=args.max_block_instrs,
                max_dataflow_edges=args.max_dataflow_edges,
                graph_extractor_sha256=graph_hash,
                prepare_only=args.prepare_only,
                public_signature_mode=args.public_signature_mode,
                toolchain_stamp=toolchain_stamp,
                assembly_mode=args.assembly_mode,
                target_name=target_name,
                protocol_schema=args.protocol_schema,
            )
            for index, row in enumerate(rows)
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            index, row, reject = future.result()
            if row is not None:
                accepted.append((index, row))
            if reject is not None:
                rejects.append(reject)
            if done % 20 == 0 or done == len(futures):
                print(
                    f"[{done}/{len(futures)}] accepted={len(accepted)} rejected={len(rejects)}"
                )

    accepted.sort(key=lambda pair: pair[0])
    output_rows = [row for _, row in accepted]
    write_jsonl(args.output, output_rows)
    if args.public_output is not None:
        public_rows = [public_row(row) for row in output_rows]
        if args.shuffle_public_seed is not None:
            random.Random(args.shuffle_public_seed).shuffle(public_rows)
        write_jsonl(args.public_output, public_rows)

    args.rejects.parent.mkdir(parents=True, exist_ok=True)
    args.rejects.write_text(json.dumps(rejects, indent=2), encoding="utf-8")
    summary = {
        **protocol_summary_fields(
            args.protocol_schema,
            arm_id_salt,
            redact_id_salt=args.redact_effective_id_salt,
        ),
        "benchmark_kind": args.benchmark_kind,
        "public_signature_mode": args.public_signature_mode,
        "target_name": target_name,
        "shuffle_public_seed": args.shuffle_public_seed,
        "input": str(args.input),
        "input_sha256": input_hash,
        "freeze_manifest_sha256": freeze_hash,
        "input_rows": len(rows),
        "accepted_rows": len(output_rows),
        "rejected_rows": len(rejects),
        "prepare_only": args.prepare_only,
        "private_output": str(args.output),
        "public_output": str(args.public_output) if args.public_output else None,
        "assembly_mode": args.assembly_mode,
        "toolchain": toolchain_stamp,
        "extractor_sha256": graph_hash,
        "builder_sha256": file_sha256(Path(__file__)),
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if len(output_rows) != expected_rows:
        raise SystemExit(
            f"signature-scrubbed build changed the denominator: "
            f"expected {expected_rows}, accepted {len(output_rows)}"
        )


if __name__ == "__main__":
    main()
