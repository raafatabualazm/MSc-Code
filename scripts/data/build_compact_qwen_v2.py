#!/usr/bin/env python3
"""Build the versioned, encoder-free compact-Qwen v2 representation.

V2 keeps every property that made compact-Qwen v1 reversible while adding two
pieces of information that are present in the Phase-0 s44 corpus:

* an explicit CFG ``call`` edge atom; and
* an explicit extractor-route atom selecting the frozen or current DFG rules.

The instruction codebook is fitted from ``--fit`` only.  Measure rows never
influence instruction frequencies, token IDs, or embedding initialisation.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_SCHEMA = "direct-compact-causal-v2"
CODEBOOK_SCHEMA = "compact-qwen-v2-codebook"
PREFLIGHT_SCHEMA = "compact-qwen-v2-preflight"
FAKE_SIGNATURE = "static void candidate(void)"
TARGET_FUNCTION = "candidate"

# V1's fail-closed x86 vocabulary plus the one legitimate Phase-0 top-up
# mnemonic absent from v1 (``dec rdx``).  local_N remains unconditionally bad.
KNOWN_X86 = set(
    "add addps addsd and call cdq cmovge cmovl cmp comisd cqo cvtsd2ss "
    "cvtsi2sd cvtss2sd cvttsd2si dec divsd idiv imul int3 ja jae jb jbe "
    "je jg jge jl jle jmp jne jno jnp jo jp lea mov movabs movaps "
    "movmskpd movq movsd movss movsx movsxd movups movzx mul mulps mulsd "
    "neg not or pop push ret roundsd sar sete setne shl shr shufps sqrtsd "
    "sub subsd test xchg xor xorpd xorps".split()
)

EDGE_TOKEN = {
    "conditional_true": "<CT>",
    "conditional_false": "<CF>",
    "linear_fallthrough": "<CN>",
    "loop_backedge": "<CL>",
    "unconditional": "<CU>",
    "unconditional_jump": "<CJ>",
    "call": "<CC>",
}
TOKEN_EDGE = {value: key for key, value in EDGE_TOKEN.items()}

ROUTE_LEGACY = "legacy_release_v1"
ROUTE_CURRENT = "current_combined_v2"


@dataclass(frozen=True)
class RouteSpec:
    name: str
    atom: str
    combined_sha256: str
    cfg_sha256: str
    dfg_sha256: str
    combined_hash_algorithm: str
    allow_call_edges: bool
    dfg_metadata: str


ROUTE_SPECS = {
    ROUTE_LEGACY: RouteSpec(
        name=ROUTE_LEGACY,
        atom="<DX0>",
        combined_sha256="7c0c2270091c98cb65726ebb0404f196abf3eec9825569653beb1e7883aac2d8",
        cfg_sha256="fc0ed42b63fc743ac6f6a1726213fd1659c398a77dc6f9c71b99437963ad53ce",
        dfg_sha256="beb237cf2ad8e3d65a536e8d30b698e14486ade36a019c247d580c372b858000",
        combined_hash_algorithm="sha256(cfg_bytes || dfg_bytes)",
        allow_call_edges=False,
        dfg_metadata="endpoints_only",
    ),
    ROUTE_CURRENT: RouteSpec(
        name=ROUTE_CURRENT,
        atom="<DX1>",
        combined_sha256="7a89b10f74754a8ff43580dba0cfb3348cd8e7b370e325ba8d31667c60ac04c1",
        cfg_sha256="daebbbfa7ac53fed9104e66396bc861bc837a8cea5a948548204d34439ee553c",
        dfg_sha256="603c052e8a79e7f6f689e97acdfc9c87245505b4fbf497bc2c49c2343fb0ed12",
        combined_hash_algorithm="sha256(filename || bytes for cfg, dfg)",
        allow_call_edges=True,
        dfg_metadata="locations_and_dependency_count",
    ),
}
ROUTE_BY_COMBINED = {spec.combined_sha256: name for name, spec in ROUTE_SPECS.items()}
ROUTE_BY_ATOM = {spec.atom: name for name, spec in ROUTE_SPECS.items()}

CONTROL = [
    "<G2C2>", "<AX64>", "<ENTRY>", "<BLOCKS>", "<CFG>", "<END>",
    "<R>", "<E>",
] + [spec.atom for spec in ROUTE_SPECS.values()] + list(TOKEN_EDGE)

TARGET_RE = re.compile(r"0x([0-9a-fA-F]+)\s*<([^>]+)>")
TAG_RE = re.compile(r"<[^>]+>")
RUNTIME_POLICY = {
    "version": "runtime-symbol-policy-v1",
    "trusted": ["stub _iso_stub_*", "dart:*", "print"],
    "self": ["candidate", "candidate+*", "candidate.*"],
    "untrusted": "per_function_@U#",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest(value: str, label: str) -> str:
    value = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            yield number, value


def load_dfg(path: Path, module_name: str) -> Callable[..., list[dict[str, Any]]]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DFG extractor {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_cross_block_dfg


def combined_extractor_sha(cfg_path: Path, dfg_path: Path, route: str) -> str:
    if route == ROUTE_LEGACY:
        return sha(cfg_path.read_bytes() + dfg_path.read_bytes())
    if route == ROUTE_CURRENT:
        result = hashlib.sha256()
        for path in (cfg_path, dfg_path):
            result.update(path.name.encode("utf-8"))
            result.update(path.read_bytes())
        return result.hexdigest()
    raise ValueError(f"unknown route {route!r}")


def load_route_registry(
    legacy_cfg: Path,
    legacy_dfg: Path,
    current_cfg: Path,
    current_dfg: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        ROUTE_LEGACY: (legacy_cfg, legacy_dfg),
        ROUTE_CURRENT: (current_cfg, current_dfg),
    }
    registry: dict[str, dict[str, Any]] = {}
    for route, (cfg_path, dfg_path) in paths.items():
        spec = ROUTE_SPECS[route]
        observed_cfg = sha(cfg_path.read_bytes())
        observed_dfg = sha(dfg_path.read_bytes())
        observed_combined = combined_extractor_sha(cfg_path, dfg_path, route)
        if observed_cfg != spec.cfg_sha256:
            raise ValueError(
                f"{route}: CFG extractor SHA mismatch: {observed_cfg} != {spec.cfg_sha256}"
            )
        if observed_dfg != spec.dfg_sha256:
            raise ValueError(
                f"{route}: DFG extractor SHA mismatch: {observed_dfg} != {spec.dfg_sha256}"
            )
        if observed_combined != spec.combined_sha256:
            raise ValueError(
                f"{route}: combined extractor SHA mismatch: "
                f"{observed_combined} != {spec.combined_sha256}"
            )
        registry[route] = {
            "spec": spec,
            "cfg_path": cfg_path,
            "dfg_path": dfg_path,
            "build_dfg": load_dfg(dfg_path, f"compact_v2_{route}_dfg"),
        }
    return registry


def extractor_route(row: dict[str, Any], route_override: str | None = None) -> str:
    if route_override:
        if route_override in ROUTE_SPECS:
            return route_override
        if route_override in ROUTE_BY_COMBINED:
            return ROUTE_BY_COMBINED[route_override]
        if route_override in ROUTE_BY_ATOM:
            return ROUTE_BY_ATOM[route_override]
        raise ValueError(f"unknown_extractor_route_override:{route_override}")
    graph = row.get("graph_v2")
    if not isinstance(graph, dict):
        raise ValueError("missing_graph_v2_provenance")
    observed = str(graph.get("extractor_sha256") or "").lower()
    if observed not in ROUTE_BY_COMBINED:
        raise ValueError(f"unknown_graph_extractor_sha256:{observed or 'missing'}")
    return ROUTE_BY_COMBINED[observed]


def _canonical_dfg_edge(edge: dict[str, Any], route: str) -> dict[str, Any]:
    base = {
        "source": int(edge["source"]),
        "target": int(edge["target"]),
        "edge_type": "dataflow",
    }
    if route == ROUTE_LEGACY:
        allowed = {"source", "target", "edge_type"}
        if set(edge) != allowed:
            raise ValueError(
                "legacy_dfg_metadata_drift:" + ",".join(sorted(set(edge) - allowed))
            )
        return base
    allowed = {"source", "target", "edge_type", "locations", "dependency_count"}
    if set(edge) != allowed:
        raise ValueError(
            "current_dfg_metadata_drift:" + ",".join(sorted(set(edge) ^ allowed))
        )
    locations = sorted(str(value) for value in edge.get("locations") or [])
    dependency_count = int(edge.get("dependency_count", -1))
    if dependency_count != len(locations):
        raise ValueError("current_dfg_dependency_count_mismatch")
    return {**base, "locations": locations, "dependency_count": dependency_count}


def _sort_dfg(edges: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        edges,
        key=lambda edge: (
            int(edge["source"]),
            int(edge["target"]),
            str(edge.get("edge_type")),
            tuple(edge.get("locations") or ()),
            int(edge.get("dependency_count", 0)),
        ),
    )


def canonicalize(
    row: dict[str, Any],
    symbol_policy: str = "runtime_aware",
    route_override: str | None = None,
) -> dict[str, Any]:
    """Return the exact reversible object represented by the compact stream."""
    if str(row.get("function") or "") != TARGET_FUNCTION:
        raise ValueError("target_function_must_be_candidate")
    route = extractor_route(row, route_override)
    route_spec = ROUTE_SPECS[route]
    cfg = row.get("cfg")
    if not isinstance(cfg, list) or not cfg:
        raise ValueError("missing_or_empty_cfg")
    block_ids = [int(block.get("id", -1)) for block in cfg]
    if block_ids != list(range(len(cfg))):
        raise ValueError("block_ids_must_be_contiguous_and_position_aligned")
    starts = {
        str(block.get("start_address", "")).lower().removeprefix("0x"): int(block["id"])
        for block in cfg
    }
    external: dict[str, int] = {}
    blocks: list[dict[str, Any]] = []
    for block in cfg:
        instructions: list[str] = []
        for raw in block.get("instructions") or []:
            instruction = str(raw).strip()
            if instruction == FAKE_SIGNATURE:
                continue
            parts = instruction.split()
            if not parts:
                continue
            opcode = parts[0].lower()
            if re.fullmatch(r"local_\d+", opcode) or opcode not in KNOWN_X86:
                raise ValueError(f"unknown_or_corrupt_mnemonic:{opcode}")

            def replace_target(match: re.Match[str]) -> str:
                address, symbol = match.group(1).lower(), match.group(2)
                if address in starts:
                    return f"@B{starts[address]}"
                if symbol.startswith("candidate+"):
                    return "@REL+" + symbol.split("+", 1)[1]
                if symbol == "candidate":
                    return "@SELF"
                if symbol.startswith("candidate."):
                    return "@SELF_CLOSURE"
                if symbol_policy == "runtime_aware":
                    stub = re.match(r"^stub _iso_stub_([A-Za-z0-9_]+)", symbol)
                    if stub:
                        return "@STUB:" + stub.group(1)
                    if symbol.startswith("dart:"):
                        return "@SDK:" + re.sub(r"[^A-Za-z0-9_:.-]", "_", symbol)
                    if symbol == "print":
                        return "@SDK:print"
                if symbol not in external:
                    external[symbol] = len(external)
                return f"@U{external[symbol]}"

            instruction = TARGET_RE.sub(replace_target, instruction)
            instruction = instruction.replace("@SELF_CLOSURE>", "@SELF_CLOSURE")
            instruction = re.sub(r"\s+", " ", instruction)
            instruction = re.sub(r"\s*,\s*", ",", instruction)
            instructions.append(instruction)
        blocks.append({"id": int(block["id"]), "instructions": instructions})

    cfg_edges: list[dict[str, Any]] = []
    dfg_edges: list[dict[str, Any]] = []
    for edge in row.get("edges") or []:
        if not isinstance(edge, dict):
            raise ValueError("edge_must_be_object")
        edge_type = str(edge.get("edge_type") or "")
        source, target = int(edge["source"]), int(edge["target"])
        if not 0 <= source < len(blocks) or not 0 <= target < len(blocks):
            raise ValueError("edge_endpoint_out_of_range")
        if edge_type == "dataflow":
            dfg_edges.append(_canonical_dfg_edge(edge, route))
            continue
        if edge_type not in EDGE_TOKEN:
            raise ValueError(f"unknown_cfg_edge_type:{edge_type}")
        if edge_type == "call" and not route_spec.allow_call_edges:
            raise ValueError("call_edge_not_allowed_for_legacy_route")
        cfg_edges.append({"source": source, "target": target, "edge_type": edge_type})

    entries = [int(value) for value in (row.get("integrity", {}).get("entry_blocks") or [0])]
    if not entries or len(entries) != len(set(entries)):
        raise ValueError("invalid_entry_blocks")
    if any(value < 0 or value >= len(blocks) for value in entries):
        raise ValueError("entry_block_out_of_range")
    return {
        "architecture": "x86_64",
        "dfg_route": route,
        "entry_blocks": entries,
        "blocks": blocks,
        "cfg_edges": cfg_edges,
        "dfg_edges": _sort_dfg(dfg_edges),
    }


def source_token_contract(
    tokenizer_path: Path,
    model_vocab_size: int,
    expansions: list[str],
    max_blocks: int,
) -> tuple[int, dict[str, list[int]], dict[str, int]]:
    base = Tokenizer.from_file(str(tokenizer_path))
    tokenizer_size = base.get_vocab_size(with_added_tokens=True)
    human = {
        "<G2C2>": " compact graph version two",
        "<AX64>": " x86 64",
        "<DX0>": " frozen legacy data flow extractor",
        "<DX1>": " current combined data flow extractor",
        "<ENTRY>": " entry blocks",
        "<BLOCKS>": " basic blocks",
        "<CFG>": " control flow edges",
        "<END>": " end compact graph",
        "<R>": " raw instruction",
        "<E>": " end raw instruction",
        "<CT>": " conditional true",
        "<CF>": " conditional false",
        "<CN>": " linear fallthrough",
        "<CL>": " loop backedge",
        "<CU>": " unconditional",
        "<CJ>": " unconditional jump",
        "<CC>": " internal call edge",
    }
    atoms = (
        [f"<I{index}>" for index in range(len(expansions))]
        + [f"<B{index}>" for index in range(max_blocks)]
        + CONTROL
    )
    if len(atoms) != len(set(atoms)):
        raise ValueError("duplicate_compact_atoms")
    atom_ids = {token: model_vocab_size + index for index, token in enumerate(atoms)}
    mapping: dict[str, list[int]] = {}
    for index, line in enumerate(expansions):
        mapping[str(atom_ids[f"<I{index}>"])] = base.encode(
            line, add_special_tokens=False
        ).ids
    for index in range(max_blocks):
        mapping[str(atom_ids[f"<B{index}>"])] = base.encode(
            f" block {index}", add_special_tokens=False
        ).ids
    for token, text in human.items():
        mapping[str(atom_ids[token])] = base.encode(text, add_special_tokens=False).ids
    if any(not value for value in mapping.values()):
        raise ValueError("invalid_source_token_expansion")
    return tokenizer_size, mapping, atom_ids


def compact_ids(text: str, base: Tokenizer, atom_ids: dict[str, int]) -> list[int]:
    result: list[int] = []
    cursor = 0
    for match in TAG_RE.finditer(text):
        if match.start() > cursor:
            result.extend(base.encode(text[cursor : match.start()], add_special_tokens=False).ids)
        token = match.group()
        if token not in atom_ids:
            raise ValueError("unknown_compact_atom:" + token)
        result.append(atom_ids[token])
        cursor = match.end()
    if cursor < len(text):
        result.extend(base.encode(text[cursor:], add_special_tokens=False).ids)
    return result


def encode(canonical: dict[str, Any], code: dict[str, int]) -> str:
    route = str(canonical["dfg_route"])
    if route not in ROUTE_SPECS:
        raise ValueError("unknown_canonical_route")
    output = ["<G2C2>", "<AX64>", ROUTE_SPECS[route].atom, "<ENTRY>"]
    output.extend(f"<B{value}>" for value in canonical["entry_blocks"])
    output.append("<BLOCKS>")
    for block in canonical["blocks"]:
        output.append(f"<B{block['id']}>")
        for instruction in block["instructions"]:
            if instruction in code:
                output.append(f"<I{code[instruction]}>")
            else:
                output.extend(("<R>", instruction, "<E>"))
    output.append("<CFG>")
    for edge in canonical["cfg_edges"]:
        output.extend(
            (
                EDGE_TOKEN[edge["edge_type"]],
                f"<B{edge['source']}>",
                f"<B{edge['target']}>",
            )
        )
    output.append("<END>")
    return "".join(output)


def decode(text: str, expansions: list[str]) -> dict[str, Any]:
    tags = list(TAG_RE.finditer(text))
    position = 0

    def take(expected: str | None = None) -> str:
        nonlocal position
        if position >= len(tags):
            raise ValueError("unexpected_eof")
        value = tags[position].group()
        position += 1
        if expected and value != expected:
            raise ValueError(f"expected_{expected}_got_{value}")
        return value

    take("<G2C2>")
    take("<AX64>")
    route_atom = take()
    if route_atom not in ROUTE_BY_ATOM:
        raise ValueError("missing_or_unknown_extractor_route_atom")
    route = ROUTE_BY_ATOM[route_atom]
    take("<ENTRY>")
    entries: list[int] = []
    while position < len(tags) and tags[position].group() != "<BLOCKS>":
        token = take()
        if not re.fullmatch(r"<B\d+>", token):
            raise ValueError("invalid_entry_block_atom")
        entries.append(int(token[2:-1]))
    take("<BLOCKS>")
    blocks: list[dict[str, Any]] = []
    while position < len(tags) and tags[position].group() != "<CFG>":
        block_atom = take()
        if not re.fullmatch(r"<B\d+>", block_atom):
            raise ValueError("invalid_block_atom")
        block_id = int(block_atom[2:-1])
        instructions: list[str] = []
        while position < len(tags) and not (
            tags[position].group().startswith("<B")
            or tags[position].group() == "<CFG>"
        ):
            token = take()
            if re.fullmatch(r"<I\d+>", token):
                index = int(token[2:-1])
                if index >= len(expansions):
                    raise ValueError("instruction_atom_out_of_range")
                instructions.append(expansions[index])
            elif token == "<R>":
                start = tags[position - 1].end()
                if position >= len(tags):
                    raise ValueError("unterminated_raw_instruction")
                end = tags[position].start()
                instructions.append(text[start:end])
                take("<E>")
            else:
                raise ValueError("bad_instruction_token:" + token)
        blocks.append({"id": block_id, "instructions": instructions})
    take("<CFG>")
    cfg_edges: list[dict[str, Any]] = []
    while position < len(tags) and tags[position].group() != "<END>":
        edge_atom = take()
        if edge_atom not in TOKEN_EDGE:
            raise ValueError("unknown_edge_atom:" + edge_atom)
        source_atom, target_atom = take(), take()
        if not re.fullmatch(r"<B\d+>", source_atom) or not re.fullmatch(
            r"<B\d+>", target_atom
        ):
            raise ValueError("invalid_edge_endpoint_atom")
        edge_type = TOKEN_EDGE[edge_atom]
        if edge_type == "call" and not ROUTE_SPECS[route].allow_call_edges:
            raise ValueError("call_edge_not_allowed_for_legacy_route")
        cfg_edges.append(
            {
                "source": int(source_atom[2:-1]),
                "target": int(target_atom[2:-1]),
                "edge_type": edge_type,
            }
        )
    take("<END>")
    if position != len(tags):
        raise ValueError("trailing_compact_atoms")
    if [block["id"] for block in blocks] != list(range(len(blocks))):
        raise ValueError("decoded_block_ids_not_contiguous")
    return {
        "architecture": "x86_64",
        "dfg_route": route,
        "entry_blocks": entries,
        "blocks": blocks,
        "cfg_edges": cfg_edges,
    }


def route_contract(registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for route, record in registry.items():
        spec: RouteSpec = record["spec"]
        result[route] = {
            "route_atom": spec.atom,
            "graph_extractor_sha256": spec.combined_sha256,
            "cfg_extractor_sha256": spec.cfg_sha256,
            "dfg_extractor_sha256": spec.dfg_sha256,
            "combined_hash_algorithm": spec.combined_hash_algorithm,
            "allow_call_edges": spec.allow_call_edges,
            "dfg_metadata": spec.dfg_metadata,
        }
    return result


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--fit", required=True, type=Path)
    parser.add_argument("--measure", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-json", required=True, type=Path)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument(
        "--legacy-cfg-extractor",
        type=Path,
        default=ROOT / "scrubbed_master_v2_release/extractors/cfg_extractor.py",
    )
    parser.add_argument(
        "--legacy-dfg-extractor",
        type=Path,
        default=ROOT / "scrubbed_master_v2_release/extractors/dfg_extractor.py",
    )
    parser.add_argument(
        "--current-cfg-extractor",
        type=Path,
        default=ROOT / "scripts/data/cfg_extractor.py",
    )
    parser.add_argument(
        "--current-dfg-extractor",
        type=Path,
        default=ROOT / "scripts/data/dfg_extractor.py",
    )
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--max-blocks", type=int, default=4096)
    parser.add_argument(
        "--max-source-tokens", "--max-prompt-tokens",
        dest="max_source_tokens", type=int, default=9000,
    )
    parser.add_argument("--max-target-tokens", type=int, default=3072)
    parser.add_argument("--max-total-tokens", type=int, default=12288)
    parser.add_argument("--tokenizer-fingerprint-sha256", required=True)
    parser.add_argument("--decoder-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--decoder-revision", required=True)
    parser.add_argument("--target-function", default=TARGET_FUNCTION)
    parser.add_argument(
        "--symbol-policy", choices=["runtime_aware", "strict_all_alias"],
        default="runtime_aware",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_fingerprint_sha256 = digest(
        args.tokenizer_fingerprint_sha256, "tokenizer_fingerprint_sha256"
    )
    if args.target_function != TARGET_FUNCTION:
        raise ValueError("compact-Qwen v2 release requires target_function=candidate")
    if not 0 < args.max_source_tokens <= 9000:
        raise ValueError("max_source_tokens must be in [1,9000]")
    if args.max_target_tokens <= 0 or args.max_total_tokens < args.max_source_tokens:
        raise ValueError("invalid target/total token limits")
    registry = load_route_registry(
        args.legacy_cfg_extractor,
        args.legacy_dfg_extractor,
        args.current_cfg_extractor,
        args.current_dfg_extractor,
    )

    fit_good: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    quarantine: list[dict[str, Any]] = []
    frequencies: collections.Counter[str] = collections.Counter()
    for line_number, row in rows(args.fit):
        try:
            canonical = canonicalize(row, args.symbol_policy)
            fit_good.append((line_number, row, canonical))
            frequencies.update(
                instruction
                for block in canonical["blocks"]
                for instruction in block["instructions"]
            )
        except Exception as error:
            quarantine.append(
                {
                    "dataset": str(args.fit),
                    "line": line_number,
                    "task_id": row.get("task_id"),
                    "reason": str(error),
                }
            )
    expansions = [value for value, _ in frequencies.most_common(args.codebook_size)]
    code = {value: index for index, value in enumerate(expansions)}

    model_config = args.model_config or args.tokenizer_json.with_name("config.json")
    model_cfg = json.loads(model_config.read_text(encoding="utf-8"))
    model_vocab_size = int(model_cfg["vocab_size"])
    base_tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    tokenizer_vocab_size, source_expansions, atom_ids = source_token_contract(
        args.tokenizer_json, model_vocab_size, expansions, args.max_blocks
    )
    custom_ids = sorted(map(int, source_expansions))
    if (
        not custom_ids
        or custom_ids[0] != model_vocab_size
        or custom_ids != list(range(model_vocab_size, model_vocab_size + len(custom_ids)))
    ):
        raise ValueError("custom source IDs must be contiguous after model vocab")

    datasets: list[tuple[str, Path, list[tuple[int, dict[str, Any], dict[str, Any]]]]] = [
        ("fit", args.fit, fit_good)
    ]
    for path in args.measure:
        good: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        for line_number, row in rows(path):
            try:
                good.append((line_number, row, canonicalize(row, args.symbol_policy)))
            except Exception as error:
                quarantine.append(
                    {
                        "dataset": str(path),
                        "line": line_number,
                        "task_id": row.get("task_id"),
                        "reason": str(error),
                    }
                )
        datasets.append(("measure", path, good))

    records: list[dict[str, Any]] = []
    lengths: list[int] = []
    failures: list[dict[str, Any]] = []
    dfg_by_route: collections.Counter[str] = collections.Counter()
    rows_by_route: collections.Counter[str] = collections.Counter()
    cfg_types: collections.Counter[str] = collections.Counter()
    fallback_by_role: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    fallback_by_family: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for role, path, good in datasets:
        for line_number, row, canonical in good:
            task_id = row.get("task_id")
            if len(canonical["blocks"]) > args.max_blocks:
                failures.append({"task_id": task_id, "reason": "block_vocab_overflow"})
                continue
            text = encode(canonical, code)
            decoded = decode(text, expansions)
            route = decoded["dfg_route"]
            regenerated = registry[route]["build_dfg"](
                decoded["blocks"], decoded["cfg_edges"], max_edges=100000
            )
            decoded["dfg_edges"] = _sort_dfg(
                _canonical_dfg_edge(edge, route) for edge in regenerated
            )
            if decoded != canonical:
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": "canonical_or_graph_roundtrip_mismatch",
                        "route": route,
                        "expected_sha256": sha(stable(canonical)),
                        "observed_sha256": sha(stable(decoded)),
                    }
                )
                continue
            ids = compact_ids(text, base_tokenizer, atom_ids)
            source_tokens = len(ids)
            lengths.append(source_tokens)
            if source_tokens > args.max_source_tokens:
                failures.append(
                    {
                        "task_id": task_id,
                        "reason": "source_token_overflow",
                        "tokens": source_tokens,
                    }
                )
            instruction_count = sum(len(block["instructions"]) for block in canonical["blocks"])
            fallback = sum(
                instruction not in code
                for block in canonical["blocks"]
                for instruction in block["instructions"]
            )
            metadata = row.get("compact_private_metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"{task_id}: compact_private_metadata must be an object")
            family = str(metadata.get("family") or "")
            fallback_by_role[role]["instructions"] += instruction_count
            fallback_by_role[role]["fallback"] += fallback
            if family:
                fallback_by_family[family]["rows"] += 1
                fallback_by_family[family]["instructions"] += instruction_count
                fallback_by_family[family]["fallback"] += fallback
            rows_by_route[route] += 1
            dfg_by_route[route] += len(decoded["dfg_edges"])
            cfg_types.update(edge["edge_type"] for edge in decoded["cfg_edges"])
            records.append(
                {
                    "compact_input_ids": ids,
                    "role": role,
                    "dataset": str(path),
                    "line": line_number,
                    "task_id": task_id,
                    "compact_text": text,
                    "source_tokens": source_tokens,
                    "canonical_sha256": sha(stable(canonical)),
                    "compact_sha256": sha(text.encode("utf-8")),
                    "fallback_instructions": fallback,
                    "instruction_count": instruction_count,
                    "dfg_route": route,
                    "graph_extractor_sha256": ROUTE_SPECS[route].combined_sha256,
                    **metadata,
                }
            )

    extractor_routes = route_contract(registry)
    codebook = {
        "schema": CODEBOOK_SCHEMA,
        "fit_public_sha256": sha(args.fit.read_bytes()),
        "fit_retained": len(fit_good),
        "fit_quarantined": sum(item["dataset"] == str(args.fit) for item in quarantine),
        "codebook_size": len(expansions),
        "expansions": expansions,
        "added_token_scheme": {
            "instruction": "<I{index}>",
            "block": "<B{index}>",
            "control_tokens": CONTROL,
            "edge_tokens": EDGE_TOKEN,
            "extractor_route_tokens": {
                route: spec.atom for route, spec in ROUTE_SPECS.items()
            },
        },
        "tokenizer_json_sha256": sha(args.tokenizer_json.read_bytes()),
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_config_sha256": sha(model_config.read_bytes()),
        "decoder_model": args.decoder_model.strip(),
        "decoder_revision": args.decoder_revision.strip(),
        "model_vocab_size": model_vocab_size,
        "base_vocab_size": model_vocab_size,
        "source_token_expansions": source_expansions,
        "source_atom_ids": atom_ids,
        "extractor_routes": extractor_routes,
        "max_blocks": args.max_blocks,
        "symbol_policy": args.symbol_policy,
        "runtime_symbol_policy": RUNTIME_POLICY if args.symbol_policy == "runtime_aware" else None,
        "runtime_symbol_policy_sha256": (
            sha(stable(RUNTIME_POLICY)) if args.symbol_policy == "runtime_aware" else None
        ),
    }
    leakage = {
        "candidate": sum("candidate" in value.lower() for value in expansions),
        "file_uri": sum("file://" in value.lower() for value in expansions),
        "absolute_symbol_address": sum(
            bool(re.search(r"0x[0-9a-fA-F]+\s*<", value)) for value in expansions
        ),
        "private_field_terms": sum(
            bool(re.search(r"dart_source|semantic_function|original_source|\btests\b", value, re.I))
            for value in expansions
        ),
    }
    role_counts = dict(collections.Counter(record["role"] for record in records))
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "rows_retained": len(records),
        "rows_by_role": role_counts,
        "rows_by_route": dict(rows_by_route),
        "quarantined": len(quarantine),
        "quarantine_reasons": dict(collections.Counter(item["reason"] for item in quarantine)),
        "failures_count": len(failures),
        "failure_examples": failures[:50],
        "tokens": {
            "kind": "compact_source_only",
            "min": min(lengths) if lengths else 0,
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": max(lengths) if lengths else 0,
            "limit": args.max_source_tokens,
        },
        "fallback_by_role": {key: dict(value) for key, value in fallback_by_role.items()},
        "fallback_by_family": {key: dict(value) for key, value in fallback_by_family.items()},
        "cfg_edge_types": dict(cfg_types),
        "lossless_invariants": {
            "lossless_domain": "scrubbed_canonical_graph_v2",
            "privacy_scrub_is_only_intentional_irreversibility": True,
            "exact_instruction_entry_cfg_route_roundtrip_rows": len(records),
            "dfg_regenerated_and_matched_rows": len(records),
            "dfg_edges_matched_edge_for_edge": sum(dfg_by_route.values()),
            "dfg_edges_by_route": dict(dfg_by_route),
            "extractor_routes": extractor_routes,
            "unknown_tokens": 0,
            "truncated_rows": 0,
            "raw_fallback_is_reversible": True,
            "call_edges_encoded_explicitly": True,
        },
        "codebook_expansion_leakage_scan": leakage,
        "passed": bool(records) and not failures and not any(leakage.values()),
        "exploratory_full_release_fit": not args.measure,
    }

    codebook_bytes = (json.dumps(codebook, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (args.output_dir / "codebook.json").write_bytes(codebook_bytes)
    codebook_sha = sha(codebook_bytes)
    codec_sha = sha(Path(__file__).read_bytes())
    tokenizer_sha = sha(args.tokenizer_json.read_bytes())
    report["contract"] = {
        "compact_codec_sha256": codec_sha,
        "compact_codebook_sha256": codebook_sha,
        "compact_tokenizer_sha256": tokenizer_sha,
        "base_vocab_size": model_vocab_size,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "model_vocab_size": model_vocab_size,
        "model_config_sha256": codebook["model_config_sha256"],
        "decoder_model": codebook["decoder_model"],
        "decoder_revision": codebook["decoder_revision"],
        "target_function": TARGET_FUNCTION,
        "target_language": "Dart",
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        "source_token_expansion_count": len(source_expansions),
        "runtime_symbol_policy_sha256": codebook["runtime_symbol_policy_sha256"],
        "extractor_routes": extractor_routes,
    }
    contract = {
        "schema": CONTRACT_SCHEMA,
        "codec_sha256": codec_sha,
        "codebook_sha256": codebook_sha,
        "tokenizer_json_sha256": tokenizer_sha,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        "model_config_sha256": codebook["model_config_sha256"],
        "decoder_model": codebook["decoder_model"],
        "decoder_revision": codebook["decoder_revision"],
        "max_source_tokens": args.max_source_tokens,
        "target_function": TARGET_FUNCTION,
        "target_language": "Dart",
        "extractor_routes": extractor_routes,
        "runtime_symbol_policy_sha256": codebook["runtime_symbol_policy_sha256"],
        "lossless_domain": "scrubbed_canonical_graph_v2",
        "max_target_tokens": args.max_target_tokens,
        "max_total_tokens": args.max_total_tokens,
        "base_vocab_size": model_vocab_size,
        "source_token_ids": custom_ids,
        "source_token_expansions": source_expansions,
        "source_embedding_init": "codebook_mean",
    }
    (args.output_dir / "compact_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "compact_model_inputs.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for record in records:
            model = {
                "compact_input_ids": record["compact_input_ids"],
                "compact_codec_sha256": codec_sha,
                "compact_codebook_sha256": codebook_sha,
                "compact_tokenizer_sha256": tokenizer_sha,
            }
            if set(model) != {
                "compact_input_ids", "compact_codec_sha256",
                "compact_codebook_sha256", "compact_tokenizer_sha256",
            }:
                raise AssertionError("strict model schema drift")
            handle.write(json.dumps(model, separators=(",", ":")) + "\n")
    with (args.output_dir / "alignment_private.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for model_row, record in enumerate(records):
            private = {key: value for key, value in record.items() if key != "compact_input_ids"}
            private["model_row"] = model_row
            handle.write(
                json.dumps(private, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    with (args.output_dir / "quarantine.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for item in quarantine:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "failures.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for item in failures:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (args.output_dir / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    names = [
        "codebook.json", "compact_contract.json", "compact_model_inputs.jsonl",
        "alignment_private.jsonl", "quarantine.jsonl", "failures.jsonl",
        "preflight_report.json",
    ]
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha((args.output_dir / name).read_bytes())}  {name}\n" for name in names),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
