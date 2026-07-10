"""
Graph-aware encoder-decoder training pipeline for neural decompilation (Antigravity version).
Supports T5/CodeT5 and Qwen (Causal LM) decoders, and PEFT (LoRA/DoRA) configuration.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Quiet mode (default on): suppress framework warning spam so training logs
# stay readable. Errors are NOT suppressed (silent failures already produced
# fake zero-metric runs). Set GRAPH_QUIET=0 to see everything.
import os
if os.environ.get("GRAPH_QUIET", "1") != "0":
    import warnings
    for _cat in (FutureWarning, UserWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=_cat)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast, PreTrainedModel
    from transformers.configuration_utils import PretrainedConfig
    _old_add_tokens = PreTrainedTokenizerFast._add_tokens
    
    def _patched_add_tokens(self, new_tokens, special_tokens=False):
        dict_or_attr = lambda o, k, d: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
        conv_tokens = []
        for t in new_tokens:
            if isinstance(t, str):
                conv_tokens.append(t)
            else:
                conv_tokens.append(tokenizers.AddedToken(
                    dict_or_attr(t, 'content', str(t)),
                    single_word=dict_or_attr(t, 'single_word', False),
                    lstrip=dict_or_attr(t, 'lstrip', False),
                    rstrip=dict_or_attr(t, 'rstrip', False),
                    normalized=dict_or_attr(t, 'normalized', True),
                    special=dict_or_attr(t, 'special', False)
                ))
        if special_tokens:
            return self._tokenizer.add_special_tokens(conv_tokens)
        return self._tokenizer.add_tokens(conv_tokens)
        
    PreTrainedTokenizerFast._add_tokens = _patched_add_tokens

    # Salesforce CodeT5+ custom config cannot be constructed with no args.
    # Transformers 4.57 calls `self.__class__()` while logging config diffs,
    # which raises before weights load. Fall back to the full dict for repr/json.
    if not getattr(PretrainedConfig, "_codet5p_to_diff_patched", False):
        _old_to_diff_dict = PretrainedConfig.to_diff_dict
        _old_get_non_default_generation_parameters = getattr(
            PretrainedConfig,
            "_get_non_default_generation_parameters",
            None,
        )

        def _full_json_safe_config_dict(config):
            config_dict = config.to_dict()
            if hasattr(config, "dict_torch_dtype_to_str"):
                config.dict_torch_dtype_to_str(config_dict)
            elif hasattr(config, "dict_dtype_to_str"):
                config.dict_dtype_to_str(config_dict)
            return config_dict

        def _patched_to_diff_dict(self):
            try:
                return _old_to_diff_dict(self)
            except AssertionError:
                return _full_json_safe_config_dict(self)

        PretrainedConfig.to_diff_dict = _patched_to_diff_dict

        if _old_get_non_default_generation_parameters is not None:
            def _patched_get_non_default_generation_parameters(self):
                try:
                    return _old_get_non_default_generation_parameters(self)
                except AssertionError:
                    return {}

            PretrainedConfig._get_non_default_generation_parameters = _patched_get_non_default_generation_parameters
        PretrainedConfig._codet5p_to_diff_patched = True
    
    # Support newer transformers quantizer tie check on older Salesforce model
    PreTrainedModel.all_tied_weights_keys = property(
        lambda self: {k: None for k in getattr(self, "_all_tied_weights_keys", [])} if isinstance(getattr(self, "_all_tied_weights_keys", None), (list, set)) else getattr(self, "_all_tied_weights_keys", {}),
        lambda self, val: setattr(self, "_all_tied_weights_keys", val)
    )
    
    PreTrainedModel.get_head_mask = lambda self, head_mask, num_hidden_layers, is_attention_chunked=False: \
        head_mask if head_mask is not None else [None] * num_hidden_layers
    
    # Patch CodeT5pEncoderDecoderModel.tie_weights dynamically inside PreTrainedModel.__init__
    _old_init = PreTrainedModel.__init__
    def _patched_init(self, *args, **kwargs):
        _old_init(self, *args, **kwargs)
        cls = self.__class__
        if cls.__name__ == "CodeT5pEncoderDecoderModel" and not getattr(cls, "_tie_weights_patched", False):
            old_tie_weights = cls.tie_weights
            def patched_tie_weights(self, *args, **kwargs):
                return old_tie_weights(self)
            cls.tie_weights = patched_tie_weights
            cls._tie_weights_patched = True
            print("Successfully patched CodeT5pEncoderDecoderModel.tie_weights dynamically!")
    PreTrainedModel.__init__ = _patched_init
except Exception as e:
    pass

import json
import re
import os
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from datasets import Dataset
from torch_geometric.data import Batch
from models.pyg_cfg_dataset import cfg_to_pyg
from models.hierarchical_graph_encoder_antigravity import LocalBlockEncoder, GraphPoolingEncoder
from scripts.data.dfg_extractor import LightweightDFGExtractor
from scripts.data.cfg_extractor import ensure_cfg_blocks
from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from models.graph_data_collator import GraphDataCollator
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    TrainerCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

ENCODER_MODEL = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")
DECODER_MODEL = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")


@dataclass
class GraphDecompilerConfig:
    train_file: str = os.environ.get("GRAPH_TRAIN_FILE", "data/datasets/dart_all.jsonl")
    eval_file: str = os.environ.get("GRAPH_EVAL_FILE", "data/datasets/test-set.jsonl")
    output_dir: str = os.environ.get("GRAPH_OUTPUT_DIR", "artifacts/graph-decompiler-v1")
    max_input_length: int = 512  # GraphCodeBERT max block length
    max_target_length: int = 768
    max_decoder_prompt_length: int = int(os.environ.get("GRAPH_DECODER_PROMPT_MAX_LENGTH", "768"))
    # LoRA-appropriate default LR (full-finetune 5e-6 barely moves adapters).
    # The strongest artifacts were trained at 1e-4 (see artifacts/*_r16_1e4_*).
    learning_rate: float = float(os.environ.get("GRAPH_LR", "1e-4"))
    # Larger default batch for H100-class GPUs; sweep runners still override via env.
    batch_size: int = int(os.environ.get("GRAPH_BATCH_SIZE", "16"))  # Support custom batch size!
    epochs: int = int(os.environ.get("GRAPH_EPOCHS", "5"))
    use_reasoning: bool = os.environ.get("GRAPH_USE_REASONING", "0") == "1"
    inject_cfg_tags: bool = True
    use_graph_encoder: bool = True


def _world_size() -> int:
    """Number of DDP processes (set by torch.distributed.run); 1 if single-GPU."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    """This process's GPU index within the node under DDP."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))


def _is_distributed() -> bool:
    return _world_size() > 1


def _compact_prompt_block(text: str, max_chars: int = 2000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[truncated]"


def _build_test_call_hint(test_code: str, max_lines: int = 10) -> str:
    """Keep only the useful candidate call examples from a Dart test harness."""
    lines = []
    for line in (test_code or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("void expect") or stripped.startswith("void expectList") or stripped.startswith("void expectMap"):
            break
        if "candidate" in stripped or stripped.startswith("final candidate"):
            lines.append(stripped)
        if len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _fit_assembly_to_prompt_budget(assembly: str, fixed_chars: int) -> str:
    """Legacy char-based assembly trim (fallback only when no tokenizer is given).

    The old 3.1 chars/token heuristic badly underestimates tokens for hex-dense
    disassembly (~1.5 chars/token measured), so it did NOT actually fit the
    budget and the trailing cue was still truncated. Prefer _fit_assembly_tokens,
    which trims by real tokens. This path is kept only for callers that cannot
    supply a tokenizer.
    """
    if os.environ.get("GRAPH_PROMPT_FIT_ASSEMBLY", "0") != "1":
        return assembly
    prompt_tokens = int(os.environ.get("GRAPH_DECODER_PROMPT_MAX_LENGTH", "768"))
    max_chars = int(os.environ.get("GRAPH_PROMPT_MAX_CHARS", str(int(prompt_tokens * 1.6))))
    budget = max(200, max_chars - fixed_chars)
    if len(assembly) <= budget:
        return assembly
    head = int(budget * 0.65)
    tail = budget - head
    return assembly[:head] + "\n; [assembly truncated]\n" + assembly[-tail:]


def _fit_assembly_tokens(assembly, tokenizer, max_prompt_len, prefix_text, suffix_text):
    """Token-exact assembly trim that GUARANTEES the whole prompt (including the
    trailing '<lang> code:' cue) fits within max_prompt_len. Keeps head+tail of
    the assembly so both the prologue and the return/epilogue survive."""
    def n_tokens(text):
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    reserve = n_tokens(prefix_text) + n_tokens(suffix_text) + 8  # margin for joins/specials
    budget = max(64, max_prompt_len - reserve)
    asm_ids = tokenizer(assembly, add_special_tokens=False)["input_ids"]
    if len(asm_ids) <= budget:
        return assembly
    marker = "\n; [assembly truncated]\n"
    keep = max(0, budget - n_tokens(marker))
    head = int(keep * 0.65)
    tail = keep - head
    head_txt = tokenizer.decode(asm_ids[:head]) if head > 0 else ""
    tail_txt = tokenizer.decode(asm_ids[-tail:]) if tail > 0 else ""
    return head_txt + marker + tail_txt


def _clean_asm_for_prompt(assembly: str) -> str:
    """Drop redundant tokens from the decoder prompt with ~zero information loss:
    metadata/slice ';' comment lines, '<symbol+0xoffset>' annotations (the branch
    target hex is already in the operand and the edge lives in the graph), and
    trailing '// =NN' immediate comments. ~25% fewer tokens on ARM64. The
    encoder/CFG channel, which parses the raw assembly separately, is unaffected.
    Enabled with GRAPH_PROMPT_CLEAN_ASM=1."""
    out = []
    for ln in assembly.splitlines():
        if ln.lstrip().startswith(";"):
            continue
        ln = re.sub(r"<[^>]*>", "", ln)
        ln = re.sub(r"//.*$", "", ln)
        ln = ln.rstrip()
        if ln.strip():
            out.append(ln)
    return "\n".join(out)


def build_decoder_prompt(record: dict, tokenizer=None, max_prompt_len: int | None = None) -> str:
    """Build a compact causal-LM prompt so Qwen can use its text prior.

    When GRAPH_PROMPT_FIT_ASSEMBLY=1 and a tokenizer+budget are supplied, the
    assembly is trimmed by real tokens so the trailing '<lang> code:' generation
    cue is never truncated away (the old char-based path lost it on ~100% of
    asm-heavy rows). With fit off, output is byte-identical to the historical
    prompt, preserving comparability for older checkpoints.
    """
    language = record.get("language", record.get("lang", "Dart"))
    assembly = (record.get("assembly") or "").strip()
    function_name = (
        record.get("function")
        or record.get("name")
        or record.get("camel_case_function_name")
        or record.get("python_function_name")
        or ""
    )
    signature = (
        record.get("dart_function_signature")
        or record.get("function_signature")
        or record.get("signature")
        or ""
    )
    tests = _build_test_call_hint(record.get("tests", ""), max_lines=10)
    reasoning = _compact_prompt_block(record.get("reasoning", ""), max_chars=1200)

    parts = [
        f"Convert the following assembly to {language} source code.",
        "Return only valid source code.",
        "Do not include explanations, markdown fences, test code, or placeholder demos.",
        "Include any required imports at the top, for example dart:io or dart:math.",
        "If you call a helper function, define that helper in the same output.",
    ]
    if signature:
        parts.append(f"Implement this exact top-level Dart signature: {signature}.")
        parts.append("Do not replace it with only a void main() demo.")
        parts.append("Do not define the required function inside main(); define it at top level.")
    elif function_name:
        parts.append(f"Target function name: {function_name}.")
    if tests:
        parts.extend([
            "",
            "Unit-test harness excerpt, for signature/call-shape only:",
            tests,
        ])
    if os.environ.get("GRAPH_USE_REASONING", "0") == "1" and reasoning:
        parts.extend([
            "",
            "High-level analysis hint:",
            reasoning,
        ])
    parts.extend([
        "",
        "Assembly:",
    ])

    prefix_text = "\n".join(parts)
    suffix_text = f"\n\n{language} code:"

    # Encoder-carries-assembly ablation (GRAPH_PROMPT_ASSEMBLY_MODE=none): withhold
    # the raw assembly text so the decoder must rely on the graph-prefix channel.
    # The rest of the prompt (instructions / signature / test-shape / cue) is kept
    # identical for a controlled comparison against the full-text arm.
    asm_mode = os.environ.get("GRAPH_PROMPT_ASSEMBLY_MODE", "full").lower()
    if asm_mode in ("none", "graph_only"):
        return prefix_text + "\n; (assembly provided via graph channel)" + suffix_text

    if os.environ.get("GRAPH_PROMPT_CLEAN_ASM", "0") == "1":
        assembly = _clean_asm_for_prompt(assembly)

    if os.environ.get("GRAPH_PROMPT_FIT_ASSEMBLY", "0") == "1":
        if tokenizer is not None and max_prompt_len:
            assembly = _fit_assembly_tokens(assembly, tokenizer, max_prompt_len, prefix_text, suffix_text)
        else:
            assembly = _fit_assembly_to_prompt_budget(assembly, len(prefix_text) + len(suffix_text) + 2)

    # Equivalent to the historical '\n'.join(parts + [assembly, '', cue]).
    return prefix_text + "\n" + assembly + suffix_text


from scripts.training.graph_positions import cumsum_position_ids  # noqa: E402 (shared with GRPO)


def canonicalize_source(source: str) -> str:
    # Strip comments only. Identifiers and whitespace/formatting are preserved so
    # the target remains compilable (renaming to var_N and collapsing whitespace
    # produced uncompilable targets, which capped compile@k at ~0).
    source = re.sub(r"//.*", "", source)
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return source.strip()


def _remove_top_level_main(code: str) -> str:
    """Remove a top-level Dart main() wrapper while preserving solution helpers."""
    main_match = re.search(
        r"^\s*(?:\w+(?:<[^>]*>)?\s+)?main\s*\([^)]*\)\s*(?:async\s*)?\{",
        code,
        flags=re.MULTILINE,
    )
    if not main_match:
        return code.strip()

    start = main_match.start()
    depth = 0
    i = main_match.end() - 1
    end = None
    while i < len(code):
        ch = code[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1

    if end is None:
        return code[:start].strip()
    return (code[:start] + code[end:]).strip()


def target_source_from_record(record: dict, raw_source: str) -> str:
    """Build the supervised target.

    Unit-test records ship reference code with a demo/test main(). For SFT and
    GRPO warm starts, the useful target is the top-level solution function(s);
    keeping the reference main teaches Qwen to emit demos that the pass harness
    strips or rejects.
    """
    if record.get("tests"):
        raw_source = _remove_top_level_main(raw_source)
        raw_source = re.sub(r"^@pragma\(.*\)\s*$", "", raw_source, flags=re.MULTILINE)
    return canonicalize_source(raw_source)


def _is_usable_record(record: dict) -> bool:
    # `source` for the SFT corpus; `dart_source`/`swift_source` for the
    # test-bearing GRPO corpus.
    src = (
        record.get("source")
        or record.get("dart_source")
        or record.get("swift_source")
        or ""
    ).strip()
    if "assembly" not in record or not src:
        return False
    asm = (record.get("assembly") or "").strip()
    if not asm:
        return False
    # Drop degenerate/missing-symbol stubs that cannot be learned.
    if "[MISSING]" in asm or "not found as symbol" in asm:
        return False
    # Drop trivial stubs (e.g. 2-instruction bodies) that carry no signal.
    if len([ln for ln in asm.splitlines() if ln.strip()]) < 3:
        return False
    return True


def load_jsonl(path: str):
    rows = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if _is_usable_record(record):
                rows.append(record)
            else:
                skipped += 1
    if skipped:
        print(f"Filtered {skipped} degenerate/missing records from {path}")
    return rows


def load_jsonl_many(paths: str):
    rows = []
    for part in str(paths).split(","):
        path = part.strip()
        if path:
            rows.extend(load_jsonl(path))
    return rows


def build_dataset(config: GraphDecompilerConfig):
    # Simply load raw records. Formatting and tokenization happen in tokenize_dataset.
    return (
        load_jsonl_many(config.train_file),
        load_jsonl_many(config.eval_file),
    )


def tokenize_dataset(records, tensor_builder: GraphCodeBERTTensorBuilder, dfg_extractor, decoder_tokenizer: PreTrainedTokenizerBase, config):
    converted = []

    for idx, record in enumerate(records):
        # CFG/edges come from the record when precomputed (build_cfg_jsonl), are
        # extracted inline when GRAPH_AUTO_CFG=1, and otherwise degrade to a
        # single 'entry' block. Centralized so SFT, GRPO, and inference agree.
        cfg_blocks, edges = ensure_cfg_blocks(record)

        block_tensors = []
        for block in cfg_blocks:
            dfg = dfg_extractor.extract_block_dfg_structured(block['instructions'])
            tensors = tensor_builder.build_block_tensors(
                block['instructions'],
                dfg,
            )
            block_tensors.append({
                'input_ids': tensors['input_ids'].tolist(),
                'position_ids': tensors['position_ids'].tolist(),
                'token_type_ids': tensors['token_type_ids'].tolist(),
                'attention_mask': tensors['attention_mask'].tolist(),
            })

        # `source` for the standard SFT corpus; `dart_source`/`swift_source` for
        # the GRPO corpus that ships unit tests.
        raw_source = (
            record.get("source")
            or record.get("dart_source")
            or record.get("swift_source")
            or ""
        )
        target_text = target_source_from_record(record, raw_source)
        labels = decoder_tokenizer(
            target_text,
            max_length=config.max_target_length,
            truncation=True,
            padding="max_length",
        )
        decoder_prompt = decoder_tokenizer(
            build_decoder_prompt(record, decoder_tokenizer, config.max_decoder_prompt_length),
            max_length=config.max_decoder_prompt_length,
            truncation=True,
            padding="max_length",
        )

        # Preserve a real EOS target even when pad_token_id == eos_token_id
        # (common for GPT-like/CodeT5+ tokenizers after assigning pad=eos).
        # Masking by token id would hide every EOS and train a model that never
        # stops, causing long rambling generations and compile@k=0.
        label_input_ids = list(labels["input_ids"])
        label_attention = list(labels["attention_mask"])
        eos_id = decoder_tokenizer.eos_token_id
        if eos_id is not None:
            active_len = int(sum(label_attention))
            has_terminal_eos = active_len > 0 and label_input_ids[active_len - 1] == eos_id
            if not has_terminal_eos:
                if active_len < len(label_input_ids):
                    label_input_ids[active_len] = eos_id
                    label_attention[active_len] = 1
                elif label_input_ids:
                    label_input_ids[-1] = eos_id
                    label_attention[-1] = 1

        # Mask only true padding positions. Do not mask by token id, because
        # real EOS may share the pad id.
        label_ids = [
            (token_id if attn == 1 else -100)
            for token_id, attn in zip(label_input_ids, label_attention)
        ]

        converted.append({
            "block_inputs": block_tensors,
            "labels": label_ids,
            "decoder_prompt_input_ids": decoder_prompt["input_ids"],
            "decoder_prompt_attention_mask": decoder_prompt["attention_mask"],
            "cfg": cfg_blocks,
            "edges": edges,
            # Carry language + unit-test harness so the GRPO stage can compute a
            # functional (test-passing) reward. These are ignored by SFT.
            "language": record.get("language", record.get("lang", "dart")),
            "tests": record.get("tests", ""),
        })

        if (idx + 1) % 500 == 0:
            print(f"Tokenized {idx + 1}/{len(records)} examples")

    return Dataset.from_list(converted)


class QwenGraphPrefixAdapter(torch.nn.Module):
    """Resample variable CFG block states into a compact Qwen soft prefix."""

    def __init__(
        self,
        hidden_dim: int,
        num_prefix_tokens: int = 16,
        num_heads: int = 8,
        dropout: float = 0.05,
        gate_init: float = 0.2,
    ):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens

        heads = max(1, min(num_heads, hidden_dim))
        while hidden_dim % heads != 0 and heads > 1:
            heads -= 1

        self.query_tokens = torch.nn.Parameter(torch.empty(num_prefix_tokens, hidden_dim))
        torch.nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        self.query_norm = torch.nn.LayerNorm(hidden_dim)
        self.key_value_norm = torch.nn.LayerNorm(hidden_dim)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
        )

        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
        self.gate_logit = torch.nn.Parameter(torch.logit(torch.tensor(gate_init)))

        # Optional RMS matching (GRAPH_QWEN_PREFIX_RMS_MATCH=1): rescale each
        # prefix vector to the decoder's token-embedding RMS (x learnable
        # scalar) before the gate. Without it, the adapter's output scale is
        # unconstrained "foreign" magnitude; at 16 tokens the decoder adapts,
        # but wide prefixes (64/128) destabilized training (eval_loss rose
        # monotonically after epoch 1 in the x86 p64/p128 arms). Off by
        # default for old-checkpoint compatibility; train and inference must
        # use the same setting.
        self.rms_match = False
        self.rms_scale = None
        self.register_buffer("prefix_target_rms", torch.tensor(0.0), persistent=False)

    def set_rms_target(self, embedding_weight: torch.Tensor) -> None:
        """Enable prefix RMS matching against the decoder embedding table."""
        with torch.no_grad():
            target = embedding_weight.detach().float().pow(2).mean().sqrt()
        self.prefix_target_rms = target.to(self.prefix_target_rms.device)
        if self.rms_scale is None:
            self.rms_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.rms_match = True
        print(f"Qwen prefix RMS matching ON: target token RMS {float(target):.6f}")

    def forward(
        self,
        graph_states: torch.Tensor,
        graph_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = graph_states.size(0)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        key_values = self.key_value_norm(graph_states)

        key_padding_mask = None
        if graph_attention_mask is not None:
            key_padding_mask = graph_attention_mask <= 0
            all_masked = key_padding_mask.all(dim=1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        attended, _ = self.cross_attn(
            self.query_norm(queries),
            key_values,
            key_values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        prefix_core = attended + self.ffn(attended)
        if self.rms_match and self.rms_scale is not None:
            rms = prefix_core.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
            prefix_core = prefix_core * (
                self.prefix_target_rms * self.rms_scale / rms
            ).to(prefix_core.dtype)
        prefix = torch.sigmoid(self.gate_logit).to(prefix_core.dtype) * prefix_core
        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=graph_attention_mask.dtype if graph_attention_mask is not None else torch.float32,
            device=graph_states.device,
        )
        return prefix, prefix_mask


def maybe_override_qwen_prefix_gate(model: torch.nn.Module) -> None:
    """Optionally force the Qwen graph-prefix gate after checkpoint loading."""
    adapter = getattr(model, "qwen_prefix_adapter", None)
    gate = getattr(adapter, "gate_logit", None) if adapter is not None else None
    if gate is None:
        return

    with torch.no_grad():
        current = float(torch.sigmoid(gate.float()).detach().cpu().item())

    raw_override = os.environ.get("GRAPH_QWEN_PREFIX_GATE_OVERRIDE", "").strip()
    if not raw_override:
        print(f"Qwen graph prefix gate: {current:.6f}")
        return

    try:
        target = float(raw_override)
    except ValueError:
        print(f"WARNING: invalid GRAPH_QWEN_PREFIX_GATE_OVERRIDE={raw_override!r}; keeping gate {current:.6f}")
        return

    target = min(max(target, 1e-4), 1.0 - 1e-4)
    with torch.no_grad():
        gate.copy_(torch.logit(torch.tensor(target, dtype=gate.dtype, device=gate.device)))
        updated = float(torch.sigmoid(gate.float()).detach().cpu().item())
    print(f"Qwen graph prefix gate override: {current:.6f} -> {updated:.6f}")


def _resolve_decoder_attention_implementation(decoder_model_name: str, is_qwen: bool) -> str | None:
    """Resolve optional HF attention backend for decoder loading."""
    requested = os.environ.get("GRAPH_ATTN_IMPLEMENTATION", "auto").strip().lower()
    if requested in {"", "none", "default"}:
        return None

    if "/" in requested:
        return requested

    valid = {"auto", "flash_attention_4", "flash_attention_3", "flash_attention_2", "flex_attention", "sdpa", "eager"}
    if requested not in valid:
        print(
            f"Unknown GRAPH_ATTN_IMPLEMENTATION={requested!r}; "
            "using Hugging Face default attention."
        )
        return None

    if requested == "auto":
        cuda_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if (
            is_qwen
            and cuda_capability[0] == 9
            and importlib.util.find_spec("flash_attn_interface") is not None
        ):
            return "flash_attention_3"
        if is_qwen and importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        if is_qwen:
            return "sdpa"
        return None

    if requested == "flash_attention_3" and importlib.util.find_spec("flash_attn_interface") is None:
        print("flash_attention_3 requested, but flash_attn_interface is not importable; falling back to flash_attention_2/sdpa.")
        if importlib.util.find_spec("flash_attn") is not None:
            return "flash_attention_2"
        return "sdpa"

    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print("flash_attention_2 requested, but flash_attn is not importable; falling back to sdpa.")
        return "sdpa"

    return requested


class GraphCodeBERTT5Seq2Seq(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.local_encoder = LocalBlockEncoder()
        self.graph_encoder = GraphPoolingEncoder()

        # Check if decoder is causal
        # Qwen has standard causal LM architecture
        decoder_model_name = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")
        decoder_peft = os.environ.get("GRAPH_DECODER_PEFT", "none").lower()
        
        # Check if we should load the decoder in 4-bit to fit in 6GB VRAM.
        # On large-VRAM GPUs (e.g. H100 80GB) set GRAPH_LOAD_4BIT=0 to train the
        # decoder in full bf16 precision instead of NF4-quantized.
        is_large = "770m" in decoder_model_name.lower() or "2b" in decoder_model_name.lower()
        _4bit_env = os.environ.get("GRAPH_LOAD_4BIT")
        if _4bit_env is not None:
            load_in_4bit = _4bit_env == "1"
        else:
            load_in_4bit = is_large and (decoder_peft in ["lora", "dora"])
        
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = quantization_config
            # Under DDP each process owns exactly one GPU; "auto" would shard or
            # offload the decoder and clash with the distributed wrapper, so pin
            # the quantized decoder to this process's rank instead.
            kwargs["device_map"] = {"": _local_rank()} if _is_distributed() else "auto"
            print(f"Loading decoder {decoder_model_name} with BitsAndBytesConfig NF4 for VRAM efficiency...")

        if "codet5" in decoder_model_name.lower():
            kwargs["use_safetensors"] = False

        is_qwen = "qwen" in decoder_model_name.lower()
        attn_implementation = _resolve_decoder_attention_implementation(decoder_model_name, is_qwen)
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
            print(f"Decoder attention implementation: {attn_implementation}")

        if is_qwen:
            self.t5_model = AutoModelForCausalLM.from_pretrained(decoder_model_name, trust_remote_code=True, **kwargs)
            self.is_causal = True
        else:
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(decoder_model_name, trust_remote_code=True, **kwargs)
            self.is_causal = False

        self.gradient_checkpointing = os.environ.get("GRAPH_GRADIENT_CHECKPOINTING", "0") == "1"
        if self.gradient_checkpointing and hasattr(self.t5_model, "gradient_checkpointing_enable"):
            self.t5_model.gradient_checkpointing_enable()
            if hasattr(self.t5_model.config, "use_cache"):
                self.t5_model.config.use_cache = False
            print("Enabled decoder gradient checkpointing.")

        self.encoder_dim = self.local_encoder.encoder.config.hidden_size
        if hasattr(self.t5_model.config, "decoder") and self.t5_model.config.decoder is not None:
            decoder_cfg = self.t5_model.config.decoder
            if hasattr(decoder_cfg, "hidden_size"):
                self.decoder_dim = decoder_cfg.hidden_size
            elif hasattr(decoder_cfg, "d_model"):
                self.decoder_dim = decoder_cfg.d_model
            elif hasattr(decoder_cfg, "n_embd"):
                self.decoder_dim = decoder_cfg.n_embd
            else:
                raise ValueError("Could not determine decoder hidden dimension from config.decoder")
        elif hasattr(self.t5_model.config, "hidden_size"):
            self.decoder_dim = self.t5_model.config.hidden_size
        elif hasattr(self.t5_model.config, "d_model"):
            self.decoder_dim = self.t5_model.config.d_model
        elif hasattr(self.t5_model.config, "n_embd"):
            self.decoder_dim = self.t5_model.config.n_embd
        else:
            raise ValueError("Could not determine decoder hidden dimension from config")

        # Upgrade to LayerNorm + MLP bridge for robust alignment
        self.projection = torch.nn.Sequential(
            torch.nn.LayerNorm(self.encoder_dim),
            torch.nn.Linear(self.encoder_dim, self.decoder_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        )

        self.qwen_prefix_tokens = int(os.environ.get("GRAPH_QWEN_PREFIX_TOKENS", "16"))
        self.qwen_prefix_adapter = None
        if self.is_causal and self.qwen_prefix_tokens > 0:
            self.qwen_prefix_adapter = QwenGraphPrefixAdapter(
                hidden_dim=self.decoder_dim,
                num_prefix_tokens=self.qwen_prefix_tokens,
                num_heads=int(os.environ.get("GRAPH_QWEN_PREFIX_HEADS", "8")),
                dropout=float(os.environ.get("GRAPH_QWEN_PREFIX_DROPOUT", "0.05")),
                gate_init=float(os.environ.get("GRAPH_QWEN_PREFIX_GATE_INIT", "0.2")),
            )
            print(
                "Enabled Qwen graph glue: "
                f"{self.qwen_prefix_tokens} learned prefix tokens, "
                f"gate_init={os.environ.get('GRAPH_QWEN_PREFIX_GATE_INIT', '0.2')}"
            )
            if os.environ.get("GRAPH_QWEN_PREFIX_RMS_MATCH", "0") == "1":
                self.qwen_prefix_adapter.set_rms_target(
                    self.base_decoder_model.get_input_embeddings().weight
                )
        elif self.is_causal:
            # True text-only mode: the decoder gets NO graph prefix and the
            # graph encoder compute is skipped entirely. (Before 2026-07-04,
            # prefix_tokens=0 silently fell through prepare_decoder_context and
            # prepended the raw variable-length block states instead - i.e. it
            # was MORE graph-conditioned, not less, invalidating no-graph
            # ablation arms.)
            print("Qwen graph prefix DISABLED (GRAPH_QWEN_PREFIX_TOKENS=0): text-only decoder; graph encoder compute skipped.")

        # Setup PEFT & Freezing Configs
        encoder_peft = os.environ.get("GRAPH_ENCODER_PEFT", "none").lower()
        freeze_encoder = os.environ.get("GRAPH_FREEZE_ENCODER", "1") == "1"
        freeze_decoder = os.environ.get("GRAPH_FREEZE_DECODER", "0") == "1"

        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        lora_r = int(os.environ.get("GRAPH_LORA_R", "16"))
        lora_alpha = int(os.environ.get("GRAPH_LORA_ALPHA", "32"))

        # Apply PEFT/Freezing to Local Encoder
        if encoder_peft in ["lora", "dora"]:
            use_dora = (encoder_peft == "dora")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
                use_dora=use_dora,
            )
            self.local_encoder.encoder = get_peft_model(self.local_encoder.encoder, lora_config)
            print(f"Applied PEFT {encoder_peft.upper()} (r={lora_r}, alpha={lora_alpha}) to LocalBlockEncoder")
        elif freeze_encoder:
            for param in self.local_encoder.parameters():
                param.requires_grad = False
            print("Froze LocalBlockEncoder weights")

        # GRAPH_GRADIENT_CHECKPOINTING must also cover the LOCAL BLOCK ENCODER
        # when it is trainable (encoder LoRA/DoRA): a function's blocks are
        # encoded in ONE (num_blocks, 512) forward, and block-split rows reach
        # hundreds of blocks - without checkpointing, one heavy row retains
        # ~25-30 GB of RoBERTa activations for the encoder backward (the
        # step-108 H100 OOM in the x86 G3 arm). use_reentrant=False so
        # gradients still reach the LoRA adapters even though the checkpointed
        # block inputs themselves do not require grad. Frozen encoders skip
        # this: no backward, nothing to checkpoint.
        if self.gradient_checkpointing and encoder_peft in ["lora", "dora"]:
            try:
                self.local_encoder.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:  # older transformers without the kwargs parameter
                self.local_encoder.encoder.gradient_checkpointing_enable()
            print("Enabled local block encoder gradient checkpointing.")

        # Apply PEFT/Freezing to Decoder
        if decoder_peft in ["lora", "dora"]:
            use_dora = (decoder_peft == "dora")
            if "codet5p-2b" in decoder_model_name.lower():
                target_modules = ["qkv_proj", "q_attn", "out_proj", "fc_in", "fc_out"]
            elif self.is_causal:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # T5/seq2seq: attend to all attention + FFN projections for a
                # stronger adapter (q/v only is the conservative minimum).
                target_modules = ["q", "k", "v", "o", "wi", "wo"]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                use_dora=use_dora,
            )
            if load_in_4bit:
                self.t5_model = prepare_model_for_kbit_training(
                    self.t5_model,
                    use_gradient_checkpointing=self.gradient_checkpointing,
                )
            self.t5_model = get_peft_model(self.t5_model, lora_config)
            print(f"Applied PEFT {decoder_peft.upper()} (r={lora_r}, alpha={lora_alpha}) to Decoder")
        elif freeze_decoder:
            for param in self.t5_model.parameters():
                param.requires_grad = False
            print("Froze Decoder weights")

        # Fix Salesforce modeling_codet5p meta device scale_attn bug by re-creating it.
        patched_scale_attn = 0
        scale_attn_devices = set()
        for name, module in self.t5_model.named_modules():
            if module.__class__.__name__ == "CodeT5pAttention":
                if hasattr(module, "scale_attn") and isinstance(module.scale_attn, torch.Tensor):
                    first_param = next(module.parameters(), None)
                    device = first_param.device if first_param is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    head_dim = getattr(module, "head_dim", 32)
                    scale_attn = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=device)).to(dtype=self.t5_model.dtype)
                    if "scale_attn" in module._buffers:
                        module._buffers["scale_attn"] = scale_attn
                    else:
                        try:
                            delattr(module, "scale_attn")
                            module.register_buffer("scale_attn", scale_attn, persistent=False)
                        except Exception:
                            module.scale_attn = scale_attn
                    patched_scale_attn += 1
                    scale_attn_devices.add(str(device))
        if patched_scale_attn:
            devices = ", ".join(sorted(scale_attn_devices))
            print(f"Re-created scale_attn buffers in {patched_scale_attn} CodeT5pAttention modules (initial device(s): {devices}).")

        # Patch self.t5_model.enc_to_dec_proj to Identity if it exists to avoid shape mismatch (e.g. codet5p-2b)
        base_model = self.base_decoder_model
        if hasattr(base_model, "enc_to_dec_proj"):
            base_model.enc_to_dec_proj = torch.nn.Identity()
            print("Successfully patched self.t5_model.enc_to_dec_proj dynamically to torch.nn.Identity()!")

    @property
    def base_decoder_model(self):
        from peft import PeftModel
        if isinstance(self.t5_model, PeftModel):
            return self.t5_model.base_model.model
        return self.t5_model

    @property
    def decoder(self):
        if hasattr(self.base_decoder_model, 'decoder'):
            return self.base_decoder_model.decoder
        return None

    @property
    def lm_head(self):
        if hasattr(self.base_decoder_model, 'lm_head'):
            return self.base_decoder_model.lm_head
        return None

    def _causal_shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """Build causal-LM teacher-forcing inputs from target labels."""
        cfg = self.base_decoder_model.config
        start_token_id = (
            getattr(cfg, "bos_token_id", None)
            or getattr(cfg, "decoder_start_token_id", None)
            or getattr(cfg, "eos_token_id", None)
            or getattr(cfg, "pad_token_id", None)
            or 0
        )
        pad_token_id = (
            getattr(cfg, "pad_token_id", None)
            or getattr(cfg, "eos_token_id", None)
            or start_token_id
        )

        decoder_input_ids = labels.new_full(labels.shape, pad_token_id)
        decoder_input_ids[:, 0] = start_token_id
        decoder_input_ids[:, 1:] = labels[:, :-1].clone()
        decoder_input_ids.masked_fill_(decoder_input_ids == -100, pad_token_id)
        return decoder_input_ids

    def prepare_decoder_context(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare graph states for the selected decoder family."""
        if self.is_causal and self.qwen_prefix_adapter is not None:
            return self.qwen_prefix_adapter(encoder_hidden_states, encoder_attention_mask)
        return encoder_hidden_states, encoder_attention_mask

    def _causal_decode(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        labels,
        decoder_input_ids,
        decoder_prompt_input_ids,
        decoder_prompt_attention_mask,
    ):
        """Causal prefix injection + teacher-forced decode.

        encoder_hidden_states may be zero-width (text-only mode,
        GRAPH_QWEN_PREFIX_TOKENS=0); the cat/mask arithmetic below handles
        prefix_len == 0 without special-casing.
        """
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Causal decoder forward requires labels or decoder_input_ids")
            decoder_input_ids = self._causal_shift_right(labels)
        else:
            pad_token_id = (
                getattr(self.base_decoder_model.config, "pad_token_id", None)
                or getattr(self.base_decoder_model.config, "eos_token_id", None)
                or 0
            )
            decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == -100, pad_token_id)

        decoder_embeds = self.base_decoder_model.get_input_embeddings()(decoder_input_ids)
        prompt_len = 0
        prompt_attention_mask = None
        if decoder_prompt_input_ids is not None:
            decoder_prompt_input_ids = decoder_prompt_input_ids.to(device=decoder_embeds.device)
            prompt_embeds = self.base_decoder_model.get_input_embeddings()(decoder_prompt_input_ids)
            prompt_len = prompt_embeds.size(1)
            if decoder_prompt_attention_mask is None:
                prompt_attention_mask = torch.ones(
                    decoder_prompt_input_ids.shape,
                    dtype=encoder_attention_mask.dtype,
                    device=encoder_attention_mask.device,
                )
            else:
                prompt_attention_mask = decoder_prompt_attention_mask.to(
                    device=encoder_attention_mask.device,
                    dtype=encoder_attention_mask.dtype,
                )
            inputs_embeds = torch.cat([encoder_hidden_states, prompt_embeds, decoder_embeds], dim=1)
        else:
            inputs_embeds = torch.cat([encoder_hidden_states, decoder_embeds], dim=1)

        inputs_embeds = inputs_embeds.to(dtype=self.base_decoder_model.dtype)

        prefix_len = encoder_hidden_states.size(1)
        target_len = decoder_input_ids.size(1)

        mask_parts = [encoder_attention_mask]
        if prompt_attention_mask is not None:
            mask_parts.append(prompt_attention_mask)
        mask_parts.append(
            torch.ones(
                (encoder_attention_mask.size(0), target_len),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        )
        combined_mask = torch.cat(mask_parts, dim=1)

        # Match generate()'s attention-mask-cumsum RoPE positions (the
        # prompt is right-padded mid-sequence). Default on for new runs;
        # GRAPH_CAUSAL_POSITION_IDS=arange reproduces the legacy training
        # convention of checkpoints trained before this fix.
        position_ids = None
        if os.environ.get("GRAPH_CAUSAL_POSITION_IDS", "cumsum").lower() != "arange":
            position_ids = cumsum_position_ids(combined_mask)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits
        target_logits = logits[:, prefix_len + prompt_len:, :]

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                target_logits.reshape(-1, target_logits.size(-1)),
                labels.view(-1),
            )

        return {"loss": loss, "logits": target_logits}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_input_ids=None,
        decoder_prompt_input_ids=None,
        decoder_prompt_attention_mask=None,
        cfg=None,
        edges=None,
        block_inputs=None,
        **kwargs,
    ):
        # Text-only causal mode (GRAPH_QWEN_PREFIX_TOKENS=0): zero-width prefix,
        # no graph compute. The zero-width tensors flow through the same
        # cat/mask code below, so the causal branch needs no special-casing.
        if self.is_causal and self.qwen_prefix_tokens == 0:
            if block_inputs is not None:
                batch_size = len(block_inputs)
            elif labels is not None:
                batch_size = labels.size(0)
            else:
                batch_size = decoder_prompt_input_ids.size(0)
            device = (
                labels.device if labels is not None
                else decoder_prompt_input_ids.device if decoder_prompt_input_ids is not None
                else "cuda"
            )
            encoder_hidden_states = torch.zeros((batch_size, 0, self.decoder_dim), device=device)
            encoder_attention_mask = torch.zeros((batch_size, 0), device=device)
            return self._causal_decode(
                encoder_hidden_states,
                encoder_attention_mask,
                labels,
                decoder_input_ids,
                decoder_prompt_input_ids,
                decoder_prompt_attention_mask,
            )

        block_embeddings_batch = []
        list_of_B_i = []

        # 1. Forward pass through Local Block Encoder
        if block_inputs is not None:
            for block_group in block_inputs:
                if not block_group:
                    list_of_B_i.append(1)
                    block_embeddings_batch.append(torch.zeros((1, self.encoder_dim), device=labels.device if labels is not None else "cuda"))
                    continue

                group_input_ids = torch.stack([
                    torch.tensor(b['input_ids'], dtype=torch.long, device=labels.device if labels is not None else "cuda")
                    for b in block_group
                ])
                group_attention_mask = torch.stack([
                    torch.tensor(b['attention_mask'], dtype=torch.float, device=labels.device if labels is not None else "cuda").squeeze(0)
                    for b in block_group
                ])
                group_position_ids = torch.stack([
                    torch.tensor(b['position_ids'], dtype=torch.long, device=labels.device if labels is not None else "cuda")
                    for b in block_group
                ])
                group_token_type_ids = torch.stack([
                    torch.tensor(b['token_type_ids'], dtype=torch.long, device=labels.device if labels is not None else "cuda")
                    for b in block_group
                ])

                block_embeddings = self.local_encoder(
                    group_input_ids,
                    group_attention_mask,
                    group_position_ids,
                    group_token_type_ids,
                )

                block_embeddings_batch.append(block_embeddings)
                list_of_B_i.append(block_embeddings.size(0))

        # 2. Construct PyG graphs for CFG GNN
        edge_index = None
        edge_attr = None

        if cfg is not None and edges is not None:
            try:
                batch_graphs = []
                for batch_index in range(len(cfg)):
                    node_embeddings = block_embeddings_batch[batch_index]
                    graph_record = {
                        'cfg': cfg[batch_index],
                        'edges': edges[batch_index],
                    }
                    batch_graphs.append(
                        cfg_to_pyg(graph_record, node_embeddings)
                    )

                pyg_batch = Batch.from_data_list(batch_graphs)
                edge_index = pyg_batch.edge_index
                edge_attr = pyg_batch.edge_attr
            except Exception as exc:
                # NEVER fail silently here: without edges the GNN degrades to
                # the dead-CFG behavior that caused the 2026-06 plateau. Keep
                # the batch alive but make the degradation impossible to miss.
                self._graph_batch_failures = getattr(self, "_graph_batch_failures", 0) + 1
                print(
                    f"[graph] WARNING (#{self._graph_batch_failures}): PyG graph "
                    f"construction failed ({exc!r}); GNN runs WITHOUT edges for "
                    f"this batch. If this repeats, the CFG channel is dead - "
                    f"investigate before trusting this run.",
                    file=sys.stderr,
                )
                edge_index = None
                edge_attr = None

        # 3. Concatenate node features and update with GNN and padding sequences
        # Run GNN in Float32 to ensure numerical stability and prevent CUBLAS internal errors in half-precision
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.amp.autocast(device_type=autocast_device, enabled=False):
            graph_inputs = torch.cat(block_embeddings_batch, dim=0).float()

            encoder_hidden_states, encoder_attention_mask = self.graph_encoder(
                graph_inputs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                list_of_B_i=list_of_B_i,
            )

            encoder_hidden_states = self.projection(encoder_hidden_states)
            encoder_hidden_states, encoder_attention_mask = self.prepare_decoder_context(
                encoder_hidden_states,
                encoder_attention_mask,
            )

        encoder_hidden_states = encoder_hidden_states.to(dtype=self.t5_model.dtype)

        # 4. Forward through decoder
        if self.is_causal:
            return self._causal_decode(
                encoder_hidden_states,
                encoder_attention_mask,
                labels,
                decoder_input_ids,
                decoder_prompt_input_ids,
                decoder_prompt_attention_mask,
            )

        else:
            # Seq2Seq Decoder
            if labels is not None and decoder_input_ids is None:
                if hasattr(self.base_decoder_model, '_shift_right'):
                    decoder_input_ids = self.base_decoder_model._shift_right(labels)
                else:
                    cfg = self.base_decoder_model.config
                    decoder_start_token_id = getattr(cfg, 'decoder_start_token_id', None)
                    if decoder_start_token_id is None and hasattr(cfg, 'decoder'):
                        decoder_start_token_id = getattr(cfg.decoder, 'decoder_start_token_id', None)
                    if decoder_start_token_id is None:
                        decoder_start_token_id = getattr(cfg, 'pad_token_id', 0)
                        
                    pad_token_id = getattr(cfg, 'pad_token_id', 0)
                    if pad_token_id is None and hasattr(cfg, 'decoder'):
                        pad_token_id = getattr(cfg.decoder, 'pad_token_id', 0)
                    if pad_token_id is None:
                        pad_token_id = 0
                        
                    shifted_input_ids = labels.new_zeros(labels.shape)
                    shifted_input_ids[..., 1:] = labels[..., :-1].clone()
                    shifted_input_ids[..., 0] = decoder_start_token_id
                    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
                    decoder_input_ids = shifted_input_ids

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

            if hasattr(decoder_outputs, 'logits') and decoder_outputs.logits is not None:
                logits = decoder_outputs.logits
            else:
                sequence_output = decoder_outputs.last_hidden_state
                sequence_output = sequence_output * (self.decoder_dim ** -0.5)
                logits = self.lm_head(sequence_output)

            loss = None
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

            return {"loss": loss, "logits": logits}


def initialize_model(tokenizer):
    return GraphCodeBERTT5Seq2Seq()


class AntigravitySeq2SeqTrainer(Seq2SeqTrainer):
    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        # Under DDP only the main process writes the checkpoint to avoid a race
        # where every rank serializes the same pytorch_model.bin concurrently.
        if not self.args.should_save:
            return
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"AntigravitySeq2SeqTrainer: Saving model state dict to {output_dir}")

        if state_dict is None:
            state_dict = self.model.state_dict()

        # Save only trainable parameters (PEFT adapters, GNN, projection layers) to reduce size and avoid 4-bit serialization issues
        trainable_keys = {name for name, param in self.model.named_parameters() if param.requires_grad}
        trainable_state_dict = {k: v for k, v in state_dict.items() if k in trainable_keys}

        torch.save(trainable_state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        if (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            self.data_collator.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def _resolve_hf_token() -> str:
    return (
        os.environ.get("GRAPH_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )


class HuggingFaceCheckpointUploadCallback(TrainerCallback):
    def __init__(self):
        self.repo_id = os.environ.get("GRAPH_HF_REPO", "").strip()
        self.token = _resolve_hf_token()
        self.private = os.environ.get("GRAPH_HF_PRIVATE", "1") == "1"
        self.enabled = (
            os.environ.get("GRAPH_HF_UPLOAD_CHECKPOINTS", "0") == "1"
            and bool(self.repo_id)
            and bool(self.token)
        )
        self.path_prefix = os.environ.get("GRAPH_HF_PATH_PREFIX", "artifacts/checkpoints").strip("/")

    def on_save(self, args, state, control, **kwargs):
        # Only the main DDP process uploads, to avoid duplicate/racing pushes.
        if not state.is_world_process_zero:
            return control
        if not self.enabled:
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            return control

        try:
            from huggingface_hub import HfApi

            api = HfApi(token=self.token)
            api.create_repo(
                repo_id=self.repo_id,
                repo_type="model",
                private=self.private,
                exist_ok=True,
            )
            api.upload_folder(
                repo_id=self.repo_id,
                repo_type="model",
                folder_path=str(checkpoint_dir),
                path_in_repo=f"{self.path_prefix}/{checkpoint_dir.name}",
                commit_message=f"Upload checkpoint {checkpoint_dir.name}",
            )
            print(f"HF checkpoint upload complete: {self.repo_id}/{self.path_prefix}/{checkpoint_dir.name}")
        except Exception as exc:
            print(f"HF checkpoint upload failed for {checkpoint_dir}: {exc}")

        return control


def main():
    config = GraphDecompilerConfig()
    dfg_extractor = LightweightDFGExtractor()

    print(
        "Graph channel config: "
        f"dfg_mode={os.environ.get('GRAPH_DFG_MODE', 'legacy')} | "
        f"position_scheme={os.environ.get('GRAPH_POSITION_SCHEME', 'legacy')} | "
        f"causal_position_ids={os.environ.get('GRAPH_CAUSAL_POSITION_IDS', 'cumsum')} | "
        f"auto_cfg={os.environ.get('GRAPH_AUTO_CFG', '0')}"
    )

    encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL, trust_remote_code=True)
    decoder_model_name = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name, trust_remote_code=True)
    
    # Configure pad token if it's missing (e.g. Qwen/Causal LMs)
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=config.max_input_length)

    print("Loading datasets...")
    train_records, eval_records = build_dataset(config)

    print("Tokenizing datasets (using unified tensor builder format)...")
    train_dataset = tokenize_dataset(train_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)
    eval_dataset = tokenize_dataset(eval_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)

    model = initialize_model(encoder_tokenizer)
    checkpoint = os.environ.get("GRAPH_CHECKPOINT", "").strip()
    if checkpoint:
        # Fail loudly. A missing/broken checkpoint that was explicitly requested
        # used to only warn, silently turning a warm-start into a cold start
        # (e.g. training _ut on 154 rows from scratch). If a cold start is really
        # intended, omit --sft_checkpoint/--grpo_checkpoint instead.
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"GRAPH_CHECKPOINT={checkpoint!r} does not exist. Refusing to silently "
                f"cold-start. Fix the path, or omit the checkpoint flag to start from base."
            )
        print(f"Loading checkpoint from: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    maybe_override_qwen_prefix_gate(model)

    max_steps = int(os.environ.get("GRAPH_MAX_STEPS", -1))

    grad_accum = int(os.environ.get("GRAPH_GRAD_ACCUM", "2"))
    print(f"Gradient accumulation steps: {grad_accum}")

    # Prefer bf16 on capable GPUs (H100/A100): avoids the fp16 NaN issues seen
    # with codet5p-770m. Falls back to fp16 on older cards, fp32 on CPU.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    save_strategy = os.environ.get("GRAPH_SAVE_STRATEGY", "no")
    save_steps = int(os.environ.get("GRAPH_SAVE_STEPS", "0"))
    save_total_limit = int(os.environ.get("GRAPH_SAVE_TOTAL_LIMIT", "2"))

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs if max_steps < 0 else 1,
        max_steps=max_steps if max_steps > 0 else -1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        eval_strategy="steps" if max_steps > 0 else "epoch",
        eval_steps=max_steps if max_steps > 0 else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" and save_steps > 0 else 500,
        save_total_limit=save_total_limit if save_strategy != "no" and save_total_limit > 0 else None,
        logging_steps=1 if max_steps > 0 else 10,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        # Quiet mode sets TRANSFORMERS_VERBOSITY=error, which would otherwise
        # auto-disable the Trainer progress bar. Keep the bar explicitly.
        disable_tqdm=False,
        remove_unused_columns=False,  # VERY IMPORTANT to keep custom graph columns!
        # The graph branch (GNN edge embedding, prefix adapter) can leave some
        # trainable params unused on a given micro-batch (e.g. a batch whose
        # samples have no CFG edges), which trips DDP's gradient reducer. Enable
        # the unused-param scan only when actually running distributed.
        ddp_find_unused_parameters=True if _is_distributed() else None,
    )


    data_collator = GraphDataCollator(decoder_tokenizer)

    trainer = AntigravitySeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    if os.environ.get("GRAPH_HF_UPLOAD_CHECKPOINTS", "0") == "1":
        trainer.add_callback(HuggingFaceCheckpointUploadCallback())

    print("Starting training...")
    trainer.train()
    print(f"Saving model to {config.output_dir}...")
    trainer.save_model(config.output_dir)


if __name__ == "__main__":
    main()
