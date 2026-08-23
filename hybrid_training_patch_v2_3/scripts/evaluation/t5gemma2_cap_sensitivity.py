#!/usr/bin/env python3
"""Extend only the completions capped by a sealed T5Gemma pass@k run.

The source run is replayed at its original stochastic batch boundaries.  This
is necessary because a sample at batch position N does not have an independent
seed: all return sequences in a generation batch share one RNG stream.  Only
batches containing capped slots are regenerated, and only capped slots replace
the source predictions.  Every replacement must reproduce the exact decoded
source prefix before it is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.evaluation import t5gemma2_f2_passk_inference as base
from scripts.evaluation.durable_evaluation_journal import (
    append_event,
    canonical_sha256,
    journal_record,
    load_journal,
    require_exact_or_write,
    sha256_file,
)


RUN_SCHEMA = "t5gemma2-f2-cap-sensitivity-v1"
JOURNAL_SCHEMA = "t5gemma2-f2-cap-sensitivity-journal-v1"


@dataclass(frozen=True)
class ReplayBatch:
    task_index: int
    task_id: str
    batch_start: int
    batch_count: int
    selected_sample_indices: tuple[int, ...]


def generate_replay_batch(
    *,
    model: Any,
    tokenizer: Any,
    encoder_outputs: Any,
    attention_mask: torch.Tensor,
    decoder_start: int,
    pad_id: int,
    eos_ids: Sequence[int],
    max_new_tokens: int,
    source_cap: int,
    temperature: float,
    top_p: float,
    seed: int,
    count: int,
) -> list[dict[str, Any]]:
    """Generate an original batch and retain a hash of its source-cap prefix."""

    if count <= 0 or source_cap <= 0 or max_new_tokens <= source_cap:
        raise ValueError("replay count/caps are inconsistent")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prefix = torch.tensor([[decoder_start]], dtype=torch.long, device="cuda")
    kwargs = base._generation_kwargs(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=pad_id,
        eos_token_ids=eos_ids,
    )
    kwargs["top_p"] = float(top_p)
    kwargs["num_return_sequences"] = int(count)
    with torch.no_grad():
        generated = model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=prefix,
            **kwargs,
        )
    eos_set = set(eos_ids)
    results: list[dict[str, Any]] = []
    for batch_position, sequence in enumerate(generated.sequences.detach().cpu()):
        actions = base.normalize_generated_seq2seq_ids(
            sequence,
            decoder_prefix_ids=[decoder_start],
            eos_token_ids=eos_ids,
            pad_token_id=pad_id,
        )
        text = base._decode_candidate(tokenizer, actions)
        prefix_actions = actions[:source_cap]
        prefix_text = base._decode_candidate(tokenizer, prefix_actions)
        eos_observed = actions[-1] in eos_set
        results.append(
            {
                "seed": seed,
                "batch_position": batch_position,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "action_tokens": len(actions),
                "eos_observed": eos_observed,
                "max_token_completion": (
                    not eos_observed and len(actions) >= max_new_tokens
                ),
                "prefix_action_tokens": len(prefix_actions),
                "prefix_text_sha256": hashlib.sha256(
                    prefix_text.encode("utf-8")
                ).hexdigest(),
            }
        )
    if len(results) != count:
        raise ValueError("generate returned a different number of replay sequences")
    return results


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def build_replay_plan(
    *,
    source_events: Sequence[Mapping[str, Any]],
    rows: Sequence[base.EvaluationRow],
    expected_arm: str,
    expected_model: Mapping[str, Any],
    expected_heldout: Mapping[str, Any],
    target_max_new_tokens: int,
    expected_capped: int = 0,
) -> tuple[
    Mapping[str, Any],
    list[dict[str, Any]],
    list[ReplayBatch],
    list[dict[str, Any]],
]:
    """Validate the complete source journal and select its capped slots."""

    if not source_events or source_events[0].get("event") != "header":
        raise ValueError("source generation journal has no header")
    source_contract = source_events[0].get("contract")
    if not isinstance(source_contract, dict):
        raise ValueError("source generation header has no run contract")
    if (
        source_contract.get("schema") != base.INFERENCE_SCHEMA
        or source_contract.get("arm") != expected_arm
        or source_contract.get("model") != expected_model
        or source_contract.get("heldout") != expected_heldout
        or source_contract.get("no_frontier_api") is not True
        or source_contract.get("tests_exposed_to_model") is not False
        or source_contract.get("targets_exposed_to_model") is not False
        or source_contract.get("source_truncation") is not False
    ):
        raise ValueError("source generation contract differs from this exact arm/data")
    sampling = source_contract.get("sampling")
    runtime = source_contract.get("runtime")
    if not isinstance(sampling, dict) or not isinstance(runtime, dict):
        raise ValueError("source generation contract lacks sampling/runtime")
    num_samples = _positive_int(sampling.get("num_samples"), "source num_samples")
    batch_size = _positive_int(
        sampling.get("generation_batch_size"), "source generation_batch_size"
    )
    source_cap = _positive_int(
        sampling.get("max_new_tokens"), "source max_new_tokens"
    )
    _positive_int(sampling.get("max_source_tokens"), "source max_source_tokens")
    if (
        batch_size > num_samples
        or sampling.get("seed_policy")
        != "seed+task_index*100003+batch_start"
        or type(sampling.get("seed")) is not int
        or int(sampling["seed"]) < 0
        or type(sampling.get("temperature")) not in (int, float)
        or not math.isfinite(float(sampling["temperature"]))
        or float(sampling["temperature"]) <= 0
        or type(sampling.get("top_p")) not in (int, float)
        or not 0.0 < float(sampling["top_p"]) <= 1.0
        or sampling.get("top_k") != 0
        or sampling.get("decoder_prefix_is_not_output") is not True
        or sampling.get("sampled_eos_retained") is not True
        or sampling.get("fabricated_eos") is not False
    ):
        raise ValueError("source sampling contract is unsupported")
    if target_max_new_tokens <= source_cap:
        raise ValueError("sensitivity cap must exceed the source cap")
    if (
        runtime.get("attn_implementation") not in {"eager", "sdpa"}
        or type(runtime.get("bf16")) is not bool
    ):
        raise ValueError("source runtime contract is unsupported")

    source_terminals, complete = base._journal_state(
        source_events,
        contract=source_contract,
        rows=rows,
        num_samples=num_samples,
    )
    if not complete:
        raise ValueError("source generation journal is incomplete")

    selected: list[dict[str, Any]] = []
    by_batch: dict[tuple[int, int], list[int]] = {}
    for task_index, terminal in enumerate(source_terminals):
        for sample_index, candidate in enumerate(terminal["candidates"]):
            if candidate["max_token_completion"] is not True:
                continue
            if (
                candidate["eos_observed"] is not False
                or int(candidate["action_tokens"]) != source_cap
            ):
                raise ValueError("source capped candidate is internally inconsistent")
            selected.append(
                {
                    "task_index": task_index,
                    "task_id": terminal["task_id"],
                    "sample_index": sample_index,
                    "source_text_sha256": candidate["text_sha256"],
                }
            )
            batch_start = (sample_index // batch_size) * batch_size
            by_batch.setdefault((task_index, batch_start), []).append(sample_index)
    if expected_capped and len(selected) != expected_capped:
        raise ValueError(
            f"source capped-slot count differs: expected={expected_capped}, "
            f"observed={len(selected)}"
        )
    if not selected:
        raise ValueError("source run has no capped completions to extend")

    batches: list[ReplayBatch] = []
    for (task_index, batch_start), sample_indices in sorted(by_batch.items()):
        batches.append(
            ReplayBatch(
                task_index=task_index,
                task_id=rows[task_index].task_id,
                batch_start=batch_start,
                batch_count=min(batch_size, num_samples - batch_start),
                selected_sample_indices=tuple(sorted(sample_indices)),
            )
        )
    return source_contract, source_terminals, batches, selected


def _candidate_is_valid(
    candidate: Mapping[str, Any],
    *,
    sample_index: int,
    source_candidate: Mapping[str, Any],
    source_cap: int,
) -> bool:
    text = candidate.get("text")
    return bool(
        isinstance(text, str)
        and candidate.get("sample_index") == sample_index
        and candidate.get("text_sha256")
        == hashlib.sha256(text.encode("utf-8")).hexdigest()
        and type(candidate.get("action_tokens")) is int
        and int(candidate["action_tokens"]) > source_cap
        and type(candidate.get("eos_observed")) is bool
        and type(candidate.get("max_token_completion")) is bool
        and candidate.get("source_text_sha256")
        == source_candidate.get("text_sha256")
        and candidate.get("prefix_text_sha256")
        == source_candidate.get("text_sha256")
        and candidate.get("prefix_action_tokens") == source_cap
        and candidate.get("source_prefix_verified") is True
    )


def sensitivity_journal_state(
    events: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    batches: Sequence[ReplayBatch],
    source_terminals: Sequence[Mapping[str, Any]],
    source_cap: int,
) -> tuple[list[dict[str, Any]], bool, str]:
    if not events:
        return [], False, ""
    header = events[0]
    if (
        header.get("event") != "header"
        or header.get("schema") != JOURNAL_SCHEMA
        or header.get("contract") != contract
        or header.get("contract_sha256") != canonical_sha256(contract)
    ):
        raise ValueError("sensitivity journal header differs from the exact run")
    terminals: list[dict[str, Any]] = []
    for event in events[1:]:
        if event.get("event") == "complete":
            if (
                len(terminals) != len(batches)
                or event.get("schema") != JOURNAL_SCHEMA
                or int(event.get("replayed_batches", -1)) != len(batches)
                or int(event.get("selected_slots", -1))
                != sum(len(batch.selected_sample_indices) for batch in batches)
                or not isinstance(event.get("predictions_canonical_sha256"), str)
            ):
                raise ValueError("sensitivity completion event is inconsistent")
            if event is not events[-1]:
                raise ValueError("events appear after sensitivity completion")
            return terminals, True, str(event["predictions_canonical_sha256"])
        position = len(terminals)
        expected = batches[position] if position < len(batches) else None
        candidates = event.get("candidates")
        if (
            expected is None
            or event.get("event") != "batch_terminal"
            or event.get("schema") != JOURNAL_SCHEMA
            or event.get("schedule_index") != position
            or event.get("task_index") != expected.task_index
            or event.get("task_id") != expected.task_id
            or event.get("batch_start") != expected.batch_start
            or event.get("batch_count") != expected.batch_count
            or not isinstance(candidates, list)
            or len(candidates) != len(expected.selected_sample_indices)
        ):
            raise ValueError("sensitivity batch terminal differs from the schedule")
        # The source hash comparison above only checks shape.  Bind it to the
        # source terminal below through the corresponding task identity.
        source_terminal = source_terminals[expected.task_index]
        if (
            event.get("source_sha256")
            != source_terminal.get("source_sha256")
            or source_terminal.get("task_id") != expected.task_id
        ):
            raise ValueError("sensitivity batch source differs from the source run")
        for candidate, sample_index in zip(
            candidates, expected.selected_sample_indices, strict=True
        ):
            source_candidate = source_terminal["candidates"][sample_index]
            if not isinstance(candidate, dict) or not _candidate_is_valid(
                candidate,
                sample_index=sample_index,
                source_candidate=source_candidate,
                source_cap=source_cap,
            ):
                raise ValueError("sensitivity candidate record is invalid")
        terminals.append(dict(event))
    return terminals, False, ""


def merge_predictions(
    *,
    source_terminals: Sequence[Mapping[str, Any]],
    sensitivity_terminals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    replacements: dict[tuple[int, int], str] = {}
    for terminal in sensitivity_terminals:
        task_index = int(terminal["task_index"])
        for candidate in terminal["candidates"]:
            key = (task_index, int(candidate["sample_index"]))
            if key in replacements:
                raise ValueError("duplicate sensitivity replacement slot")
            replacements[key] = str(candidate["text"])
    predictions: list[dict[str, Any]] = []
    for task_index, terminal in enumerate(source_terminals):
        predictions.append(
            {
                "id": terminal["task_id"],
                "predictions": [
                    replacements.get((task_index, sample_index), candidate["text"])
                    for sample_index, candidate in enumerate(terminal["candidates"])
                ],
            }
        )
    return predictions


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("T5Gemma cap sensitivity requires CUDA")
    rows, heldout_record = base.load_heldout_rows(
        dataset=args.dataset,
        dataset_seal=args.dataset_seal,
        f2_jsonl=args.f2_jsonl,
        f2_manifest=args.f2_manifest,
        limit=0,
    )
    checkpoint = Path(args.sft_checkpoint).expanduser().resolve()
    checkpoint_contract, model_record = base._checkpoint_record(checkpoint, args.arm)
    source_path = Path(args.source_journal).expanduser().resolve()
    source_events = load_journal(source_path)
    (
        source_contract,
        source_terminals,
        batches,
        selected,
    ) = build_replay_plan(
        source_events=source_events,
        rows=rows,
        expected_arm=args.arm,
        expected_model=model_record,
        expected_heldout=heldout_record,
        target_max_new_tokens=args.max_new_tokens,
        expected_capped=args.expected_capped,
    )
    sampling = dict(source_contract["sampling"])
    runtime = dict(source_contract["runtime"])
    if (
        runtime.get("torch") != str(torch.__version__)
        or runtime.get("cuda") != str(torch.version.cuda or "")
    ):
        raise ValueError(
            "source torch/CUDA runtime differs; exact stochastic replay is not sealed"
        )
    source_cap = int(sampling["max_new_tokens"])
    selected_sha256 = canonical_sha256(selected)
    output_path = Path(args.output).expanduser().resolve()
    provenance_path = Path(str(output_path) + ".provenance.json")
    sensitivity_journal = Path(
        args.journal or (str(output_path) + ".generation.journal.jsonl")
    ).expanduser().resolve()
    contract = {
        "schema": RUN_SCHEMA,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "arm": args.arm,
        "model": model_record,
        "heldout": heldout_record,
        "source_generation_journal": journal_record(source_path),
        "source_contract_sha256": canonical_sha256(source_contract),
        "source_max_new_tokens": source_cap,
        "target_max_new_tokens": args.max_new_tokens,
        "selected_slots": len(selected),
        "selected_slots_sha256": selected_sha256,
        "replayed_batches": len(batches),
        "exact_original_batch_boundaries": True,
        "require_source_prefix_identity": True,
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda or ""),
            "bf16": runtime["bf16"],
            "attn_implementation": runtime["attn_implementation"],
        },
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "targets_exposed_to_model": False,
    }
    events = load_journal(sensitivity_journal)
    if not events:
        if output_path.exists() or provenance_path.exists():
            raise ValueError("published output exists without its sensitivity journal")
        append_event(
            sensitivity_journal,
            {
                "event": "header",
                "schema": JOURNAL_SCHEMA,
                "contract": contract,
                "contract_sha256": canonical_sha256(contract),
            },
        )
        events = load_journal(sensitivity_journal)
    terminals, complete, completed_predictions_sha256 = sensitivity_journal_state(
        events,
        contract=contract,
        batches=batches,
        source_terminals=source_terminals,
        source_cap=source_cap,
    )

    if not complete:
        model, tokenizer, loaded_record = base.load_policy(
            checkpoint=checkpoint,
            arm=args.arm,
            bf16=bool(runtime["bf16"]),
            attn_implementation=str(runtime["attn_implementation"]),
        )
        if loaded_record != model_record:
            raise ValueError("loaded model record differs from sensitivity preflight")
        decoder_start, pad_id, eos_ids = base._decoder_special_ids(model, tokenizer)
        device = torch.device("cuda")
        for schedule_index in range(len(terminals), len(batches)):
            replay = batches[schedule_index]
            row = rows[replay.task_index]
            input_ids, attention_mask = base._encode_source(
                tokenizer,
                row.source,
                max_source_tokens=int(sampling["max_source_tokens"]),
                device=device,
            )
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
            generated = generate_replay_batch(
                model=model,
                tokenizer=tokenizer,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_start=decoder_start,
                pad_id=pad_id,
                eos_ids=eos_ids,
                max_new_tokens=args.max_new_tokens,
                source_cap=source_cap,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                seed=base.sample_seed(
                    int(sampling["seed"]),
                    replay.task_index,
                    replay.batch_start,
                ),
                count=replay.batch_count,
            )
            selected_candidates: list[dict[str, Any]] = []
            source_terminal = source_terminals[replay.task_index]
            for sample_index in replay.selected_sample_indices:
                generated_candidate = generated[sample_index - replay.batch_start]
                source_candidate = source_terminal["candidates"][sample_index]
                if (
                    generated_candidate.get("prefix_action_tokens") != source_cap
                    or generated_candidate.get("prefix_text_sha256")
                    != source_candidate.get("text_sha256")
                ):
                    raise ValueError(
                        f"{replay.task_id}[{sample_index}]: regenerated prefix "
                        "differs from the sealed 4096-token source"
                    )
                if int(generated_candidate["action_tokens"]) <= source_cap:
                    raise ValueError(
                        f"{replay.task_id}[{sample_index}]: sensitivity did not "
                        "extend beyond the source cap"
                    )
                selected_candidates.append(
                    {
                        "sample_index": sample_index,
                        **generated_candidate,
                        "source_text_sha256": source_candidate["text_sha256"],
                        "source_prefix_verified": True,
                    }
                )
            terminal = append_event(
                sensitivity_journal,
                {
                    "event": "batch_terminal",
                    "schema": JOURNAL_SCHEMA,
                    "schedule_index": schedule_index,
                    "task_index": replay.task_index,
                    "task_id": replay.task_id,
                    "batch_start": replay.batch_start,
                    "batch_count": replay.batch_count,
                    "source_sha256": row.source_sha256,
                    "encoder_tokens": int(input_ids.size(1)),
                    "candidates": selected_candidates,
                },
            )
            terminals.append(terminal)
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "sensitivity_batch": schedule_index + 1,
                        "sensitivity_batches": len(batches),
                        "task_id": replay.task_id,
                        "selected_samples": list(replay.selected_sample_indices),
                        "remaining_max_token_completions": sum(
                            candidate["max_token_completion"]
                            for candidate in selected_candidates
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        predictions = merge_predictions(
            source_terminals=source_terminals,
            sensitivity_terminals=terminals,
        )
        append_event(
            sensitivity_journal,
            {
                "event": "complete",
                "schema": JOURNAL_SCHEMA,
                "replayed_batches": len(batches),
                "selected_slots": len(selected),
                "predictions_canonical_sha256": canonical_sha256(predictions),
            },
        )
        events = load_journal(sensitivity_journal)
        terminals, complete, completed_predictions_sha256 = (
            sensitivity_journal_state(
                events,
                contract=contract,
                batches=batches,
                source_terminals=source_terminals,
                source_cap=source_cap,
            )
        )
    if not complete:
        raise RuntimeError("sensitivity journal did not reach completion")

    predictions = merge_predictions(
        source_terminals=source_terminals,
        sensitivity_terminals=terminals,
    )
    if canonical_sha256(predictions) != completed_predictions_sha256:
        raise ValueError("sensitivity completion hash differs from merged predictions")
    require_exact_or_write(output_path, predictions)
    remaining_capped = sum(
        candidate["max_token_completion"]
        for terminal in terminals
        for candidate in terminal["candidates"]
    )
    provenance = {
        "schema": base.PROVENANCE_SCHEMA,
        "architecture": "native_t5gemma2_encoder_decoder",
        "arm": args.arm,
        "output_sha256": sha256_file(output_path),
        "num_rows": len(predictions),
        "num_samples": int(sampling["num_samples"]),
        "model": model_record,
        "heldout": heldout_record,
        "sampling": {
            **sampling,
            "max_new_tokens": args.max_new_tokens,
            "cap_sensitivity_replacements_only": True,
        },
        "max_token_completions": remaining_capped,
        "generation_journal": journal_record(sensitivity_journal),
        "cap_sensitivity": {
            "source_generation_journal": journal_record(source_path),
            "source_contract_sha256": canonical_sha256(source_contract),
            "source_max_new_tokens": source_cap,
            "target_max_new_tokens": args.max_new_tokens,
            "selected_slots": len(selected),
            "selected_slots_sha256": selected_sha256,
            "replayed_batches": len(batches),
            "unchanged_slots": len(rows) * int(sampling["num_samples"])
            - len(selected),
            "source_prefixes_verified": len(selected),
            "remaining_max_token_completions": remaining_capped,
        },
        "no_frontier_api": True,
        "tests_exposed_to_model": False,
        "targets_exposed_to_model": False,
        "sft_checkpoint_contract_sha256": canonical_sha256(checkpoint_contract),
    }
    require_exact_or_write(provenance_path, provenance)
    result = {
        "arm": args.arm,
        "rows": len(predictions),
        "samples": len(predictions) * int(sampling["num_samples"]),
        "extended_slots": len(selected),
        "replayed_batches": len(batches),
        "remaining_max_token_completions": remaining_capped,
        "output": str(output_path),
        "output_sha256": provenance["output_sha256"],
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_seal", required=True)
    parser.add_argument("--f2_jsonl", required=True)
    parser.add_argument("--f2_manifest", required=True)
    parser.add_argument("--sft_checkpoint", required=True)
    parser.add_argument("--arm", choices=["base", "sft"], required=True)
    parser.add_argument("--source_journal", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", default="")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--expected_capped", type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or args.expected_capped < 0:
        parser.error("token cap must be positive; expected count must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
