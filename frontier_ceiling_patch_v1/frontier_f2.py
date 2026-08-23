"""Lossless, tokenizer-aware semantic text codec for frontier prompts.

F2 uses visible task-local one-token Unicode symbols as ordinary aliases.  It
never exposes tokenizer IDs and it proves an exact canonical round trip before
returning text.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Mapping, Sequence


class F2CodecError(ValueError):
    """The semantic graph cannot be represented or decoded without ambiguity."""


F2_SCHEMA = "lossless-semantic-f2"
TEXT_MACRO_TRIGGER_TOKENS = 10_500
MAX_TEXT_MACROS = 4

EDGE_TYPE_TO_CODE = {
    "conditional_true": "T",
    "conditional_false": "F",
    "linear_fallthrough": "N",
    "loop_backedge": "L",
    "unconditional": "U",
    "unconditional_jump": "J",
}
EDGE_CODE_TO_TYPE = {value: key for key, value in EDGE_TYPE_TO_CODE.items()}

OPCODE_TO_ALIAS = {
    "movsd": "s",
    "jne": "ne",
    "jae": "ae",
    "jnp": "np",
    "comisd": "o",
    "movups": "u",
    "add": "+",
    "jbe": "be",
    "mulsd": "*",
    "xorps": "x",
    "movaps": "v",
    "subsd": "_",
    "divsd": "/",
    "cvtsi2sd": "i",
    "shr": ">",
    "movmskpd": "k",
    "roundsd": "r",
    "cvttsd2si": "f",
    "shl": "<",
    "jge": "ge",
    "jle": "le",
    "jno": "no",
    "jg": "g",
}
ALIAS_TO_OPCODE = {value: key for key, value in OPCODE_TO_ALIAS.items()}

SIZE_TO_ALIAS = {
    "QWORD PTR ": "Q",
    "DWORD PTR ": "D",
    "WORD PTR ": "W",
    "BYTE PTR ": "B",
    "XMMWORD PTR ": "X",
    "YMMWORD PTR ": "Y",
    "ZMMWORD PTR ": "Z",
}
ALIAS_TO_SIZE = {value: key for key, value in SIZE_TO_ALIAS.items()}


def _make_register_aliases() -> dict[str, str]:
    aliases = {
        "rax": "A",
        "rbx": "B",
        "rcx": "C",
        "rdx": "D",
        "rdi": "I",
        "rsi": "S",
        "rbp": "P",
        "rsp": "R",
        "r8": "E",
        "r9": "F",
        "r10": "G",
        "r11": "H",
        "r12": "K",
        "r13": "M",
        "r14": "T",
        "r15": "V",
        "rip": "IP",
    }
    legacy = {
        "rax": ("eax", "ax", "al"),
        "rbx": ("ebx", "bx", "bl"),
        "rcx": ("ecx", "cx", "cl"),
        "rdx": ("edx", "dx", "dl"),
        "rdi": ("edi", "di", "dil"),
        "rsi": ("esi", "si", "sil"),
        "rbp": ("ebp", "bp", "bpl"),
        "rsp": ("esp", "sp", "spl"),
    }
    for full, (dword, word, byte) in legacy.items():
        base = aliases[full]
        aliases[dword] = base + "d"
        aliases[word] = base + "w"
        aliases[byte] = base + "b"
    for number, base in zip(
        range(8, 16), ("E", "F", "G", "H", "K", "M", "T", "V")
    ):
        aliases[f"r{number}d"] = base + "d"
        aliases[f"r{number}w"] = base + "w"
        aliases[f"r{number}b"] = base + "b"
    aliases["eip"] = "IPd"
    for number in range(32):
        aliases[f"xmm{number}"] = f"X{number}"
        aliases[f"ymm{number}"] = f"Y{number}"
        aliases[f"zmm{number}"] = f"Z{number}"
    return aliases


REGISTER_TO_ALIAS = _make_register_aliases()
ALIAS_TO_REGISTER = {value: key for key, value in REGISTER_TO_ALIAS.items()}
REGISTER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(
        re.escape(value)
        for value in sorted(REGISTER_TO_ALIAS, key=len, reverse=True)
    )
    + r")(?![A-Za-z0-9_])"
)
REGISTER_ALIAS_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_~])("
    + "|".join(
        re.escape(value)
        for value in sorted(ALIAS_TO_REGISTER, key=len, reverse=True)
    )
    + r")(?![A-Za-z0-9_])"
)
DIRECT_BLOCK_BRANCH = re.compile(r"^(j[a-z0-9]+) @B([0-9]+)$")
SYMBOLIC_TARGET_OPCODES = {"call", "callq", "jmp", "jmpq"}
WHOLE_USER_TARGET = re.compile(r"@U[0-9]+(?:\+0x[0-9a-fA-F]+)?(?:>)?\Z")
WHOLE_EXTERNAL_TARGET = re.compile(r"@X[0-9]+(?:\+0x[0-9a-fA-F]+)?(?:>)?\Z")
WHOLE_SELF_TARGET = re.compile(r"@SELF(?:\+0x[0-9a-fA-F]+)?(?:>)?\Z")
COMPACT_HEX_LITERAL = re.compile(r"#[0-9a-fA-F]+")


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids if hasattr(encoded, "ids") else encoded
    return [int(value) for value in ids]


def _tokenizer_decode(tokenizer: Any, ids: Sequence[int]) -> str:
    if not ids:
        return ""
    return str(tokenizer.decode(list(ids), skip_special_tokens=False))


def visible_one_token_symbols(tokenizer: Any) -> tuple[str, ...]:
    """Find visible CJK scalars that are exact one-token round trips."""
    candidates: list[tuple[str, int]] = []
    size = int(tokenizer.get_vocab_size(with_added_tokens=True))
    for token_id in range(size):
        symbol = _tokenizer_decode(tokenizer, [token_id])
        if len(symbol) == 1 and 0x4E00 <= ord(symbol) < 0x9FFF:
            candidates.append((symbol, token_id))
    candidates.sort(key=lambda value: ord(value[0]))
    if hasattr(tokenizer, "encode_batch"):
        encoded = tokenizer.encode_batch(
            [symbol for symbol, _ in candidates],
            add_special_tokens=False,
        )
        symbols = [
            symbol
            for (symbol, token_id), value in zip(candidates, encoded)
            if [int(item) for item in value.ids] == [token_id]
        ]
    else:
        symbols = [
            symbol
            for symbol, token_id in candidates
            if _tokenizer_encode(tokenizer, symbol) == [token_id]
        ]
    if not symbols:
        raise F2CodecError(
            "pinned tokenizer has no visible one-token F2 task symbols"
        )
    return tuple(symbols)


def _encoded_opcode(opcode: str) -> str:
    alias = OPCODE_TO_ALIAS.get(opcode)
    if alias is not None:
        return alias
    if (
        opcode in ALIAS_TO_OPCODE
        or opcode.startswith(".")
        or any(character in opcode for character in ":{}|\r\n")
    ):
        return "." + opcode
    return opcode


def _decoded_opcode(value: str) -> str:
    if value.startswith("."):
        if len(value) == 1:
            raise F2CodecError("empty escaped F2 opcode")
        return value[1:]
    return ALIAS_TO_OPCODE.get(value, value)


def _compact_generic_operands(operands: str) -> str:
    # A symbolic alias can occur inside an otherwise generic operand (for
    # example a binary annotation on lea).  Whole-operand targets have shorter
    # dedicated encodings below.  Percent escapes preserve literal characters
    # that F2 otherwise uses for hexadecimal values and symbolic aliases.
    value = (
        operands.replace("%", "%25")
        .replace("#", "%23")
        .replace("~", "%7E")
        .replace("@", "~")
    )
    for original, alias in SIZE_TO_ALIAS.items():
        value = value.replace(original, alias)
    value = REGISTER_PATTERN.sub(
        lambda match: REGISTER_TO_ALIAS[match.group(1)], value
    )
    return re.sub(r"0x([0-9a-fA-F]+)", r"#\1", value)


def _expand_generic_operands(operands: str) -> str:
    invalid_escape = re.search(r"%(?!25|23|7E)", operands)
    if invalid_escape is not None:
        raise F2CodecError("generic F2 operand has an invalid percent escape")
    value = re.sub(r"#([0-9a-fA-F]+)", r"0x\1", operands)
    value = re.sub(
        # String/memory instructions can retain an explicit x86 segment
        # override between the size alias and '[' (for example
        # ``Bes:[rdi]``).  The compacting side intentionally preserves that
        # segment text, so the inverse must recognize both ordinary and
        # segment-qualified memory operands.
        r"([QDWBXYZ])(?=(?:(?:cs|ds|es|fs|gs|ss):)?\[)",
        lambda match: ALIAS_TO_SIZE[match.group(1)],
        value,
    )
    value = REGISTER_ALIAS_PATTERN.sub(
        lambda match: ALIAS_TO_REGISTER[match.group(1)], value
    )
    value = value.replace("~", "@")
    return re.sub(
        r"%(25|23|7E)",
        lambda match: {
            "25": "%",
            "23": "#",
            "7E": "~",
        }[match.group(1)],
        value,
    )


def _branch_target_can_be_elided(
    instructions: Sequence[str],
    instruction_index: int,
    edges: Sequence[Mapping[str, Any]],
) -> bool:
    instruction = instructions[instruction_index]
    match = DIRECT_BLOCK_BRANCH.fullmatch(instruction)
    if match is None or instruction_index != len(instructions) - 1:
        return False
    if sum(
        DIRECT_BLOCK_BRANCH.fullmatch(candidate) is not None
        for candidate in instructions
    ) != 1:
        return False
    opcode, target_text = match.groups()
    target = int(target_text)
    if opcode == "jmp":
        eligible = [
            int(edge["target"])
            for edge in edges
            if str(edge["edge_type"])
            in {"unconditional", "unconditional_jump", "loop_backedge"}
        ]
    else:
        eligible = [
            int(edge["target"])
            for edge in edges
            if str(edge["edge_type"]) in {"conditional_true", "loop_backedge"}
        ]
    return eligible == [target]


def _compact_instruction(
    instruction: str,
    block_symbol_by_id: Mapping[int, str],
    *,
    elide_target: bool,
) -> str:
    if not instruction or any(
        character in instruction for character in "{}|\r\n\t"
    ):
        raise F2CodecError(
            "normalized instruction contains an F2 stream delimiter"
        )
    opcode, separator, operands = instruction.partition(" ")
    encoded_opcode = _encoded_opcode(opcode)
    if elide_target:
        if not separator or DIRECT_BLOCK_BRANCH.fullmatch(instruction) is None:
            raise F2CodecError("invalid F2 branch-target elision request")
        return encoded_opcode
    if not separator:
        return encoded_opcode
    if operands.startswith("@STUB:"):
        compact_operands = "S:" + operands[len("@STUB:") :]
    elif operands.startswith("@SDK:"):
        compact_operands = "K:" + operands[len("@SDK:") :]
    elif operands.startswith("@REL+"):
        compact_operands = "R+" + _compact_generic_operands(
            operands[len("@REL+") :]
        )
    elif WHOLE_USER_TARGET.fullmatch(operands):
        compact_operands = "U" + operands[len("@U") :]
    elif WHOLE_EXTERNAL_TARGET.fullmatch(operands):
        compact_operands = "X" + operands[len("@X") :]
    elif WHOLE_SELF_TARGET.fullmatch(operands):
        compact_operands = operands
    else:
        block_target = re.fullmatch(r"@B([0-9]+)", operands)
        if block_target is not None:
            target = int(block_target.group(1))
            if target not in block_symbol_by_id:
                raise F2CodecError(
                    f"instruction targets unknown block B{target}"
                )
            compact_operands = "@" + block_symbol_by_id[target]
        else:
            compact_operands = _compact_generic_operands(operands)
    return encoded_opcode + ":" + compact_operands


def _resolved_elided_target(
    opcode: str,
    edges: Sequence[Mapping[str, Any]],
) -> int:
    if opcode == "jmp":
        targets = [
            int(edge["target"])
            for edge in edges
            if str(edge["edge_type"])
            in {"unconditional", "unconditional_jump", "loop_backedge"}
        ]
    elif opcode.startswith("j"):
        targets = [
            int(edge["target"])
            for edge in edges
            if str(edge["edge_type"]) in {"conditional_true", "loop_backedge"}
        ]
    else:
        raise F2CodecError(
            f"operandless F2 instruction {opcode!r} is not a branch"
        )
    if len(targets) != 1:
        raise F2CodecError(
            f"operandless F2 branch {opcode!r} has {len(targets)} CFG targets"
        )
    return targets[0]


def _expand_compact_instruction(
    compact: str,
    edges: Sequence[Mapping[str, Any]],
    block_id_by_symbol: Mapping[str, int],
    hex_macros: Mapping[str, str] | None = None,
    text_macros: Mapping[str, str] | None = None,
) -> str:
    for symbol, expansion in (text_macros or {}).items():
        compact = compact.replace(symbol, expansion)
    for symbol, digits in (hex_macros or {}).items():
        compact = compact.replace(symbol, "#" + digits)
    encoded_opcode, separator, operands = compact.partition(":")
    opcode = _decoded_opcode(encoded_opcode)
    if not separator:
        if opcode.startswith("j"):
            return f"{opcode} @B{_resolved_elided_target(opcode, edges)}"
        return opcode
    if operands.startswith("S:") and opcode in {"call", "jmp"}:
        expanded = "@STUB:" + operands[2:]
    elif operands.startswith("K:") and opcode == "call":
        expanded = "@SDK:" + operands[2:]
    elif operands.startswith("R+") and opcode.startswith("j"):
        expanded = "@REL+" + _expand_generic_operands(operands[2:])
    elif operands.startswith("U") and opcode in SYMBOLIC_TARGET_OPCODES | {"fn"}:
        expanded = "@U" + operands[1:]
    elif operands.startswith("X") and opcode in SYMBOLIC_TARGET_OPCODES:
        expanded = "@X" + operands[1:]
    elif operands.startswith("@SELF") and opcode in SYMBOLIC_TARGET_OPCODES | {"fn"}:
        expanded = operands
    elif operands.startswith("@"):
        symbol = operands[1:]
        if len(symbol) != 1 or symbol not in block_id_by_symbol:
            raise F2CodecError(
                "F2 instruction has an unknown block target symbol"
            )
        expanded = f"@B{block_id_by_symbol[symbol]}"
    else:
        expanded = _expand_generic_operands(operands)
    return opcode + " " + expanded


def _encode_cfg(
    source: int,
    edges: Sequence[Mapping[str, Any]],
    block_symbol_by_id: Mapping[int, str],
) -> str:
    pieces: list[str] = []
    index = 0
    while index < len(edges):
        edge = edges[index]
        edge_type = str(edge["edge_type"])
        if edge_type not in EDGE_TYPE_TO_CODE:
            raise F2CodecError(f"unknown CFG edge type {edge_type!r}")
        target = int(edge["target"])
        if target not in block_symbol_by_id:
            raise F2CodecError(f"CFG edge targets unknown block B{target}")
        if (
            index + 1 < len(edges)
            and edge_type in {"conditional_true", "loop_backedge"}
            and str(edges[index + 1]["edge_type"]) == "conditional_false"
            and int(edges[index + 1]["target"]) == source + 1
        ):
            pieces.append(
                ("t" if edge_type == "conditional_true" else "l")
                + block_symbol_by_id[target]
            )
            index += 2
            continue
        code = EDGE_TYPE_TO_CODE[edge_type]
        if target == source + 1 and code == "N":
            pieces.append("n")
        elif target == source + 1 and code == "F":
            pieces.append("f")
        else:
            pieces.append(code + block_symbol_by_id[target])
        index += 1
    return "".join(pieces)


def _decode_cfg(
    source: int,
    value: str,
    block_id_by_symbol: Mapping[str, int],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    index = 0
    while index < len(value):
        code = value[index]
        index += 1
        if code in {"n", "f"}:
            target = source + 1
            edge_type = (
                "linear_fallthrough" if code == "n" else "conditional_false"
            )
            if target not in block_id_by_symbol.values():
                raise F2CodecError(
                    "implicit F2 CFG edge targets no next block"
                )
        elif code in {"t", "l"}:
            if index >= len(value):
                raise F2CodecError("truncated compound F2 CFG edge")
            symbol = value[index]
            index += 1
            if symbol not in block_id_by_symbol:
                raise F2CodecError("F2 CFG edge has unknown target symbol")
            next_block = source + 1
            if next_block not in block_id_by_symbol.values():
                raise F2CodecError(
                    "compound F2 CFG edge targets no false-next block"
                )
            edges.extend(
                [
                    {
                        "source": source,
                        "target": block_id_by_symbol[symbol],
                        "edge_type": (
                            "conditional_true"
                            if code == "t"
                            else "loop_backedge"
                        ),
                    },
                    {
                        "source": source,
                        "target": next_block,
                        "edge_type": "conditional_false",
                    },
                ]
            )
            continue
        else:
            if code not in EDGE_CODE_TO_TYPE or index >= len(value):
                raise F2CodecError("malformed F2 CFG edge stream")
            symbol = value[index]
            index += 1
            if symbol not in block_id_by_symbol:
                raise F2CodecError("F2 CFG edge has unknown target symbol")
            target = block_id_by_symbol[symbol]
            edge_type = EDGE_CODE_TO_TYPE[code]
        edges.append(
            {"source": source, "target": target, "edge_type": edge_type}
        )
    return edges


def decode_f2(text: str) -> tuple[str, dict[str, Any]]:
    """Decode F2 text back to the exact constants prefix and canonical graph."""
    payload = text.encode("utf-8")
    if not payload.startswith(b"F2\nC"):
        raise F2CodecError("F2 text has an invalid header")
    length_end = payload.find(b"\n", 4)
    if length_end < 0:
        raise F2CodecError("F2 constants length is unterminated")
    try:
        constant_bytes = int(payload[4:length_end].decode("ascii"))
    except Exception as exc:
        raise F2CodecError("F2 constants length is invalid") from exc
    if constant_bytes < 0:
        raise F2CodecError("F2 constants length is negative")
    constant_start = length_end + 1
    constant_end = constant_start + constant_bytes
    if constant_end > len(payload):
        raise F2CodecError("F2 constants payload is truncated")
    try:
        prefix_text = payload[constant_start:constant_end].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise F2CodecError("F2 constants payload is not UTF-8") from exc
    if payload[constant_end : constant_end + 1] != b"\n":
        raise F2CodecError(
            "F2 constants payload has no structural terminator"
        )
    try:
        structural = payload[constant_end + 1 :].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise F2CodecError("F2 structure is not UTF-8") from exc
    if not structural.endswith("\n"):
        raise F2CodecError("F2 structure has no final newline")
    lines = structural[:-1].split("\n")
    if len(lines) < 5 or not lines[0].startswith("A"):
        raise F2CodecError("F2 structure has no architecture")
    architecture = lines[0][1:]
    if not lines[1].startswith("E"):
        raise F2CodecError(
            "F2 structure has malformed entry/definition headers"
        )
    entry_symbols = list(lines[1][1:])
    cursor = 2
    hex_macros: dict[str, str] = {}
    if cursor < len(lines) and lines[cursor] == "H":
        cursor += 1
        while cursor < len(lines) and lines[cursor] not in {"M", "D"}:
            line = lines[cursor]
            if len(line) < 2 or re.fullmatch(
                r"[0-9a-fA-F]+", line[1:]
            ) is None:
                raise F2CodecError("malformed F2 hexadecimal macro")
            symbol, digits = line[0], line[1:]
            if symbol in hex_macros:
                raise F2CodecError("duplicate F2 hexadecimal macro")
            hex_macros[symbol] = digits
            cursor += 1
    text_macros: dict[str, str] = {}
    if cursor < len(lines) and lines[cursor] == "M":
        cursor += 1
        while cursor < len(lines) and lines[cursor] != "D":
            line = lines[cursor]
            if (
                len(line) < 2
                or line[0] in hex_macros
                or any(
                    ord(character) < 0x20
                    or ord(character) > 0x7E
                    or character in "{}"
                    for character in line[1:]
                )
            ):
                raise F2CodecError("malformed F2 text macro")
            symbol, expansion = line[0], line[1:]
            if symbol in text_macros:
                raise F2CodecError("duplicate F2 text macro")
            text_macros[symbol] = expansion
            cursor += 1
    if cursor >= len(lines) or lines[cursor] != "D":
        raise F2CodecError(
            "F2 structure has malformed macro/definition headers"
        )
    definition_header = cursor
    try:
        sequence_header = lines.index("S", definition_header + 1)
    except ValueError as exc:
        raise F2CodecError(
            "F2 structure has no sequence header"
        ) from exc
    block_candidates = [
        (index, line)
        for index, line in enumerate(lines[sequence_header + 1 :], sequence_header + 1)
        if line in {"B", "B2"}
    ]
    if len(block_candidates) != 1:
        raise F2CodecError("F2 structure has no unique block header")
    block_header, block_mode = block_candidates[0]
    if lines[-1] != "X":
        raise F2CodecError("F2 structure has no exact terminator")

    definitions: dict[str, str] = {}
    for line in lines[definition_header + 1 : sequence_header]:
        if len(line) < 2:
            raise F2CodecError("F2 definition is empty")
        symbol, compact = line[0], line[1:]
        if (
            symbol in definitions
            or symbol in hex_macros
            or symbol in text_macros
        ):
            raise F2CodecError("duplicate F2 instruction symbol")
        definitions[symbol] = compact

    encoded_sequences: dict[str, str] = {}
    for line in lines[sequence_header + 1 : block_header]:
        if len(line) < 2:
            raise F2CodecError("F2 sequence definition is empty")
        symbol, stream = line[0], line[1:]
        if (
            symbol in hex_macros
            or symbol in text_macros
            or symbol in definitions
            or symbol in encoded_sequences
        ):
            raise F2CodecError("duplicate or overlapping F2 definition symbol")
        encoded_sequences[symbol] = stream

    encoded_blocks: list[tuple[str, str, str]] = []
    if block_mode == "B":
        for line in lines[block_header + 1 : -1]:
            if len(line) < 2 or "|" not in line[1:]:
                raise F2CodecError("malformed F2 block record")
            block_symbol = line[0]
            stream, cfg = line[1:].split("|", 1)
            encoded_blocks.append((block_symbol, stream, cfg))
    else:
        if len(lines[block_header + 1 : -1]) != 1:
            raise F2CodecError("B2 must contain one concatenated block stream")
        payload = lines[block_header + 1]
        instruction_symbols = set(definitions) | set(encoded_sequences)
        cfg_codes = set(EDGE_CODE_TO_TYPE) | {"n", "f", "t", "l"}
        index = 0
        while index < len(payload):
            block_symbol = payload[index]
            if (
                block_symbol in instruction_symbols
                or block_symbol in cfg_codes
                or block_symbol in "{}"
            ):
                raise F2CodecError("B2 block begins with a reserved symbol")
            index += 1
            stream_start = index
            while index < len(payload):
                value = payload[index]
                if value == "{":
                    close = payload.find("}", index + 1)
                    if close < 0 or "{" in payload[index + 1 : close]:
                        raise F2CodecError(
                            "malformed B2 inline instruction"
                        )
                    index = close + 1
                elif value in instruction_symbols:
                    index += 1
                else:
                    break
            stream = payload[stream_start:index]
            cfg_start = index
            while index < len(payload) and payload[index] in cfg_codes:
                code = payload[index]
                index += 1
                if code not in {"n", "f"}:
                    if index >= len(payload):
                        raise F2CodecError("truncated B2 CFG target")
                    target = payload[index]
                    if (
                        target in instruction_symbols
                        or target in cfg_codes
                        or target in "{}"
                    ):
                        raise F2CodecError(
                            "B2 CFG target is not a block symbol"
                        )
                    index += 1
            encoded_blocks.append(
                (block_symbol, stream, payload[cfg_start:index])
            )
    block_symbols = [value[0] for value in encoded_blocks]
    if len(set(block_symbols)) != len(block_symbols):
        raise F2CodecError("duplicate F2 block symbol")
    if set(block_symbols) & (
        set(hex_macros)
        | set(text_macros)
        | set(definitions)
        | set(encoded_sequences)
    ):
        raise F2CodecError("F2 block and definition symbols overlap")
    block_id_by_symbol = {
        symbol: block_id for block_id, symbol in enumerate(block_symbols)
    }
    try:
        entry_blocks = [
            block_id_by_symbol[symbol] for symbol in entry_symbols
        ]
    except KeyError as exc:
        raise F2CodecError("F2 entry references an unknown block") from exc

    def parse_stream(stream: str, *, allow_sequences: bool) -> list[str]:
        compact_instructions: list[str] = []
        index = 0
        while index < len(stream):
            if stream[index] == "{":
                close = stream.find("}", index + 1)
                if close < 0:
                    raise F2CodecError(
                        "unterminated F2 inline instruction"
                    )
                compact = stream[index + 1 : close]
                if not compact or "{" in compact:
                    raise F2CodecError("malformed F2 inline instruction")
                compact_instructions.append(compact)
                index = close + 1
            else:
                symbol = stream[index]
                index += 1
                if symbol not in definitions:
                    if allow_sequences and symbol in decoded_sequences:
                        compact_instructions.extend(decoded_sequences[symbol])
                    else:
                        raise F2CodecError(
                            "F2 stream references an unknown definition"
                        )
                else:
                    compact_instructions.append(definitions[symbol])
        return compact_instructions

    decoded_sequences: dict[str, list[str]] = {}
    for symbol, stream in encoded_sequences.items():
        decoded_sequences[symbol] = parse_stream(
            stream, allow_sequences=False
        )

    blocks: list[dict[str, Any]] = []
    cfg_edges: list[dict[str, Any]] = []
    for block_id, (_, stream, cfg_text) in enumerate(encoded_blocks):
        edges = _decode_cfg(block_id, cfg_text, block_id_by_symbol)
        cfg_edges.extend(edges)
        compact_instructions = parse_stream(stream, allow_sequences=True)
        instructions = [
            _expand_compact_instruction(
                compact,
                edges,
                block_id_by_symbol,
                hex_macros,
                text_macros,
            )
            for compact in compact_instructions
        ]
        blocks.append({"id": block_id, "instructions": instructions})
    return prefix_text, {
        "architecture": architecture,
        "entry_blocks": entry_blocks,
        "blocks": blocks,
        "cfg_edges": cfg_edges,
    }


def serialize_f2(
    prefix_text: str,
    canonical: Mapping[str, Any],
    *,
    tokenizer: Any,
    visible_symbols: Sequence[str] | None = None,
) -> str:
    """Encode and internally prove a lossless F2 semantic graph."""
    symbols = tuple(visible_symbols or visible_one_token_symbols(tokenizer))
    if len(set(symbols)) != len(symbols):
        raise F2CodecError("F2 visible symbol pool contains duplicates")
    for symbol in symbols:
        if len(symbol) != 1:
            raise F2CodecError(
                "F2 symbol pool contains a non-scalar value"
            )

    architecture = str(canonical.get("architecture") or "")
    if architecture != "x86_64":
        raise F2CodecError(f"unsupported F2 architecture {architecture!r}")
    canonical_blocks = list(canonical.get("blocks") or [])
    expected_ids = list(range(len(canonical_blocks)))
    actual_ids = [int(block["id"]) for block in canonical_blocks]
    if actual_ids != expected_ids:
        raise F2CodecError(
            "F2 requires ordered contiguous block IDs; refusing implicit remap"
        )
    if len(symbols) < len(canonical_blocks):
        raise F2CodecError("not enough visible one-token F2 block symbols")
    block_symbol_by_id = {
        block_id: symbols[block_id] for block_id in expected_ids
    }

    normalized_edges: list[dict[str, Any]] = []
    edges_by_source: dict[int, list[dict[str, Any]]] = {
        block_id: [] for block_id in expected_ids
    }
    for edge_value in canonical.get("cfg_edges") or []:
        edge = {
            "source": int(edge_value["source"]),
            "target": int(edge_value["target"]),
            "edge_type": str(edge_value["edge_type"]),
        }
        if edge["source"] not in edges_by_source:
            raise F2CodecError("CFG edge has unknown source block")
        if edge["target"] not in block_symbol_by_id:
            raise F2CodecError("CFG edge has unknown target block")
        if edge["edge_type"] not in EDGE_TYPE_TO_CODE:
            raise F2CodecError(
                f"unknown CFG edge type {edge['edge_type']!r}"
            )
        normalized_edges.append(edge)
        edges_by_source[edge["source"]].append(edge)
    grouped_edges = [
        edge for block_id in expected_ids for edge in edges_by_source[block_id]
    ]
    if grouped_edges != normalized_edges:
        raise F2CodecError(
            "F2 cannot preserve a CFG whose edge rows are not source-grouped"
        )

    raw_compact_blocks: list[list[str]] = []
    for block_id, block in enumerate(canonical_blocks):
        instructions = [
            str(value) for value in block.get("instructions") or []
        ]
        compact_instructions: list[str] = []
        for index, instruction in enumerate(instructions):
            compact = _compact_instruction(
                instruction,
                block_symbol_by_id,
                elide_target=_branch_target_can_be_elided(
                    instructions, index, edges_by_source[block_id]
                ),
            )
            compact_instructions.append(compact)
        raw_compact_blocks.append(compact_instructions)

    # Repeated hexadecimal literals are a large source of tokenizer cost in
    # unique memory-addressing instructions.  Give profitable literals
    # task-local, visible, one-token aliases before instruction/sequence
    # factoring.  H definitions retain the exact hexadecimal spelling; no
    # numeric value, sign, width, or occurrence is dropped.
    hex_counts: dict[str, int] = {}
    hex_first_occurrence: dict[str, int] = {}
    hex_occurrence = 0
    for compact_instructions in raw_compact_blocks:
        for compact in compact_instructions:
            for match in COMPACT_HEX_LITERAL.finditer(compact):
                literal = match.group()
                hex_counts[literal] = hex_counts.get(literal, 0) + 1
                hex_first_occurrence.setdefault(literal, hex_occurrence)
                hex_occurrence += 1
    hex_candidates = [
        literal for literal, count in hex_counts.items() if count >= 2
    ]
    hex_candidates.sort(
        key=lambda literal: (
            -(
                hex_counts[literal]
                * (
                    len(_tokenizer_encode(tokenizer, literal))
                    - 1
                )
            ),
            hex_first_occurrence[literal],
            literal,
        )
    )
    hex_macro_by_literal: dict[str, str] = {}
    hex_definitions: list[tuple[str, str]] = []
    for literal in hex_candidates:
        symbol_index = len(canonical_blocks) + len(hex_definitions)
        if symbol_index >= len(symbols):
            raise F2CodecError(
                "not enough visible one-token F2 hexadecimal symbols"
            )
        symbol = symbols[symbol_index]
        count = hex_counts[literal]
        inline_cost = count * len(
            _tokenizer_encode(tokenizer, literal)
        )
        macro_cost = len(
            _tokenizer_encode(tokenizer, symbol + literal[1:] + "\n")
        ) + count
        if macro_cost < inline_cost:
            hex_macro_by_literal[literal] = symbol
            hex_definitions.append((symbol, literal[1:]))

    compact_blocks = [
        [
            COMPACT_HEX_LITERAL.sub(
                lambda match: hex_macro_by_literal.get(
                    match.group(), match.group()
                ),
                compact,
            )
            for compact in compact_instructions
        ]
        for compact_instructions in raw_compact_blocks
    ]
    compact_counts: dict[str, int] = {}
    first_occurrence: dict[str, int] = {}
    occurrence_index = 0
    for compact_instructions in compact_blocks:
        for compact in compact_instructions:
            compact_counts[compact] = compact_counts.get(compact, 0) + 1
            first_occurrence.setdefault(compact, occurrence_index)
            occurrence_index += 1

    candidate_values = [
        value for value, count in compact_counts.items() if count >= 2
    ]
    candidate_values.sort(
        key=lambda value: (
            -(
                compact_counts[value]
                * len(_tokenizer_encode(tokenizer, "{" + value + "}"))
                - len(_tokenizer_encode(tokenizer, value + "\n"))
                - compact_counts[value]
            ),
            first_occurrence[value],
            value,
        )
    )
    available_refs = symbols[
        len(canonical_blocks) + len(hex_definitions) :
    ]
    instruction_refs: dict[str, str] = {}
    definitions: list[tuple[str, str]] = []
    for compact in candidate_values:
        if len(definitions) >= len(available_refs):
            raise F2CodecError("not enough visible one-token F2 IREF symbols")
        symbol = available_refs[len(definitions)]
        count = compact_counts[compact]
        inline_cost = count * len(
            _tokenizer_encode(tokenizer, "{" + compact + "}")
        )
        reference_cost = len(
            _tokenizer_encode(tokenizer, symbol + compact + "\n")
        ) + count * len(_tokenizer_encode(tokenizer, symbol))
        if reference_cost < inline_cost:
            instruction_refs[compact] = symbol
            definitions.append((symbol, compact))

    base_entries = [
        [
            instruction_refs.get(compact, "{" + compact + "}")
            for compact in compact_instructions
        ]
        for compact_instructions in compact_blocks
    ]

    sequence_positions: dict[
        tuple[str, ...], list[tuple[int, int]]
    ] = {}
    for block_id, compact_instructions in enumerate(compact_blocks):
        for width in range(2, min(16, len(compact_instructions)) + 1):
            for start in range(len(compact_instructions) - width + 1):
                key = tuple(compact_instructions[start : start + width])
                sequence_positions.setdefault(key, []).append((block_id, start))
    sequence_candidates = [
        (key, positions)
        for key, positions in sequence_positions.items()
        if len(positions) >= 2
    ]

    def base_sequence_text(key: Sequence[str]) -> str:
        return "".join(
            instruction_refs.get(compact, "{" + compact + "}")
            for compact in key
        )

    sequence_candidates.sort(
        key=lambda item: (
            -(
                len(item[1])
                * len(_tokenizer_encode(tokenizer, base_sequence_text(item[0])))
                - len(_tokenizer_encode(tokenizer, base_sequence_text(item[0]) + "\n"))
                - len(item[1])
            ),
            -len(item[0]),
            item[1][0],
            item[0],
        )
    )
    occupied = [
        [False] * len(compact_instructions)
        for compact_instructions in compact_blocks
    ]
    sequence_at: dict[tuple[int, int], tuple[str, int]] = {}
    sequence_definitions: list[tuple[str, str]] = []
    next_symbol_index = (
        len(canonical_blocks) + len(hex_definitions) + len(definitions)
    )
    for key, positions in sequence_candidates:
        selected: list[tuple[int, int]] = []
        temporarily_selected: dict[int, set[int]] = {}
        width = len(key)
        for block_id, start in positions:
            indices = range(start, start + width)
            local = temporarily_selected.setdefault(block_id, set())
            if any(occupied[block_id][index] or index in local for index in indices):
                continue
            selected.append((block_id, start))
            local.update(indices)
        if len(selected) < 2:
            continue
        if next_symbol_index >= len(symbols):
            raise F2CodecError("not enough visible one-token F2 SREF symbols")
        symbol = symbols[next_symbol_index]
        body = base_sequence_text(key)
        inline_cost = len(selected) * len(
            _tokenizer_encode(tokenizer, body)
        )
        macro_cost = len(
            _tokenizer_encode(tokenizer, symbol + body + "\n")
        ) + len(selected) * len(_tokenizer_encode(tokenizer, symbol))
        if macro_cost >= inline_cost:
            continue
        sequence_definitions.append((symbol, body))
        next_symbol_index += 1
        for block_id, start in selected:
            for index in range(start, start + width):
                occupied[block_id][index] = True
            sequence_at[(block_id, start)] = (symbol, width)

    entry_blocks = [
        int(value) for value in canonical.get("entry_blocks") or []
    ]
    if any(value not in block_symbol_by_id for value in entry_blocks):
        raise F2CodecError("entry list references an unknown block")
    block_records: list[str] = []
    for block_id, entries in enumerate(base_entries):
        stream_parts: list[str] = []
        index = 0
        while index < len(entries):
            sequence = sequence_at.get((block_id, index))
            if sequence is not None:
                symbol, width = sequence
                stream_parts.append(symbol)
                index += width
            else:
                stream_parts.append(entries[index])
                index += 1
        stream = "".join(stream_parts)
        cfg = _encode_cfg(
            block_id, edges_by_source[block_id], block_symbol_by_id
        )
        block_records.append(block_symbol_by_id[block_id] + stream + cfg)

    text_definitions: list[tuple[str, str]] = []

    def render(
        instruction_definitions: Sequence[tuple[str, str]],
        sequences: Sequence[tuple[str, str]],
        records: Sequence[str],
        substring_definitions: Sequence[tuple[str, str]],
    ) -> str:
        lines = [
            "A" + architecture,
            "E"
            + "".join(
                block_symbol_by_id[value] for value in entry_blocks
            ),
        ]
        if hex_definitions:
            lines.append("H")
            lines.extend(
                symbol + digits for symbol, digits in hex_definitions
            )
        if substring_definitions:
            lines.append("M")
            lines.extend(
                symbol + expansion
                for symbol, expansion in substring_definitions
            )
        lines.append("D")
        lines.extend(
            symbol + compact
            for symbol, compact in instruction_definitions
        )
        lines.append("S")
        lines.extend(symbol + body for symbol, body in sequences)
        lines.extend(("B2", "".join(records), "X"))
        return (
            f"F2\nC{len(prefix_text.encode('utf-8'))}\n"
            + prefix_text
            + "\n"
            + "\n".join(lines)
            + "\n"
        )

    def replace_raw_compacts(
        value: str, substring: str, symbol: str
    ) -> str:
        return re.sub(
            r"\{([^{}]*)\}",
            lambda match: (
                "{"
                + match.group(1).replace(substring, symbol)
                + "}"
            ),
            value,
        )

    def apply_text_macro(
        substring: str, symbol: str
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, str]],
        list[str],
    ]:
        return (
            [
                (reference, compact.replace(substring, symbol))
                for reference, compact in definitions
            ],
            [
                (
                    reference,
                    replace_raw_compacts(body, substring, symbol),
                )
                for reference, body in sequence_definitions
            ],
            [
                replace_raw_compacts(record, substring, symbol)
                for record in block_records
            ],
        )

    text = render(
        definitions,
        sequence_definitions,
        block_records,
        text_definitions,
    )
    if len(_tokenizer_encode(tokenizer, text)) > TEXT_MACRO_TRIGGER_TOKENS:
        for _ in range(MAX_TEXT_MACROS):
            bodies = [compact for _, compact in definitions]
            for _, body in sequence_definitions:
                bodies.extend(re.findall(r"\{([^{}]*)\}", body))
            for record in block_records:
                bodies.extend(re.findall(r"\{([^{}]*)\}", record))
            candidate_counts: Counter[str] = Counter()
            for body in bodies:
                for ascii_run in re.findall(r"[\x20-\x7e]+", body):
                    for width in range(
                        2, min(12, len(ascii_run)) + 1
                    ):
                        for start in range(
                            len(ascii_run) - width + 1
                        ):
                            candidate_counts[
                                ascii_run[start : start + width]
                            ] += 1
            ranked = []
            for substring, count in candidate_counts.items():
                if count < 2 or substring.strip() != substring:
                    continue
                character_gain = (
                    count * (len(substring) - 1)
                    - len(substring)
                    - 2
                )
                if character_gain > 0:
                    ranked.append(
                        (
                            -character_gain,
                            -count,
                            -len(substring),
                            substring,
                        )
                    )
            ranked.sort()
            if not ranked:
                break
            macro_index = next_symbol_index + len(text_definitions)
            if macro_index >= len(symbols):
                raise F2CodecError(
                    "not enough visible one-token F2 text-macro symbols"
                )
            macro_symbol = symbols[macro_index]
            current_tokens = len(_tokenizer_encode(tokenizer, text))
            best: tuple[
                int,
                str,
                list[tuple[str, str]],
                list[tuple[str, str]],
                list[str],
                str,
            ] | None = None
            for _, _, _, substring in ranked[:96]:
                candidate_definitions, candidate_sequences, candidate_records = (
                    apply_text_macro(substring, macro_symbol)
                )
                candidate_text = render(
                    candidate_definitions,
                    candidate_sequences,
                    candidate_records,
                    [
                        *text_definitions,
                        (macro_symbol, substring),
                    ],
                )
                gain = current_tokens - len(
                    _tokenizer_encode(tokenizer, candidate_text)
                )
                candidate = (
                    gain,
                    substring,
                    candidate_definitions,
                    candidate_sequences,
                    candidate_records,
                    candidate_text,
                )
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            if best is None or best[0] <= 0:
                break
            (
                _,
                substring,
                definitions,
                sequence_definitions,
                block_records,
                text,
            ) = best
            text_definitions.append((macro_symbol, substring))

    used_symbols = (
        list(block_symbol_by_id.values())
        + [symbol for symbol, _ in hex_definitions]
        + [symbol for symbol, _ in definitions]
        + [symbol for symbol, _ in sequence_definitions]
        + [symbol for symbol, _ in text_definitions]
    )
    if len(used_symbols) != len(set(used_symbols)):
        raise F2CodecError("F2 symbol categories overlap")
    if hasattr(tokenizer, "encode_batch"):
        encoded_symbols = tokenizer.encode_batch(
            used_symbols, add_special_tokens=False
        )
        symbol_ids = [
            [int(item) for item in encoded.ids] for encoded in encoded_symbols
        ]
    else:
        symbol_ids = [
            _tokenizer_encode(tokenizer, symbol) for symbol in used_symbols
        ]
    for symbol, ids in zip(used_symbols, symbol_ids):
        if len(ids) != 1 or _tokenizer_decode(tokenizer, ids) != symbol:
            raise F2CodecError(
                "used F2 symbol is not a visible exact one-token scalar"
            )

    reconstructed_prefix, reconstructed = decode_f2(text)
    expected = {
        "architecture": architecture,
        "entry_blocks": entry_blocks,
        "blocks": [
            {
                "id": block_id,
                "instructions": [
                    str(value) for value in block.get("instructions") or []
                ],
            }
            for block_id, block in enumerate(canonical_blocks)
        ],
        "cfg_edges": normalized_edges,
    }
    if reconstructed_prefix != prefix_text:
        raise F2CodecError("F2 constant-prefix byte round trip failed")
    if reconstructed != expected:
        raise F2CodecError("F2 canonical semantic round trip failed")
    return text


F2_SYSTEM_PROMPT = (
    "Return only self-contained equivalent Dart fn0+imports/helpers; "
    "no main/tests/fences/prose. "
    "F2: Cn+n UTF8 prefix; strings/numbers=JSON; external J[n]=@Xn. "
    "E=entry BREFs; "
    "H lines=HREF+hex, HREF=#hex; "
    "M lines=MREF+ASCII, expand MREF substring first; "
    "D lines=IREF+instruction; S lines=SREF+refs/{inline}; "
    "B2 repeats BREF+instructions+CFG; X=end. "
    "CFG T/F/N/L/U/J+BREF=true/false/fallthrough/loop/unconditional/jump; "
    "n/f=to next; t/l+BREF=T/L target plus F next. "
    "Instruction=opcode[:operands]. ne/ae/np/be/ge/le/no/g add j. "
    "Aliases [s,o,u,+,*,x,v,_,/,i,>,k,r,f,<]=[movsd,comisd,"
    "movups,add,mulsd,xorps,movaps,subsd,divsd,cvtsi2sd,shr,movmskpd,"
    "roundsd,cvttsd2si,shl]; others literal. Regs "
    "[A,B,C,D,I,S,P,R,E,F,G,H,K,M,T,V]=[rax,rbx,rcx,rdx,rdi,rsi,rbp,rsp,"
    "r8,r9,r10,r11,r12,r13,r14,r15]; d/w/b narrows; Xn/Yn/Zn=xmm/ymm/zmm. "
    "[Q,D,W,B,X,Y,Z][ or SIZE+seg:[ means "
    "[QWORD,DWORD,WORD,BYTE,XMMWORD,YMMWORD,ZMMWORD] PTR; "
    "PTR [. #h=0xh; [S:,K:,U,X,R+]=[@STUB:,@SDK:,@U,@X,@REL+]; ~=@; "
    "%23/%7E/%25=#/~/%; @BREF=block; fn:@SELF=target; fn:Un=helper @Un. "
    "Bare jcc uses T (L loop); bare jmp uses U/J/L."
)
