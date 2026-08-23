#!/usr/bin/env python3
"""Single-seed HOLD gate for typed fold-only Arm B.

The mathematical/coverage checks are deliberately reused byte-for-byte from
the preregistered pass-3 gate, but this wrapper gives the measured arm its own
schema and labels.  A fold result may be vetoed by one seed; it can never be
promoted without at least three matched seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.evaluation import audit_t5gemma2_typed_pass3_promotion as shared
from scripts.evaluation.durable_evaluation_journal import require_exact_or_write, sha256_file


REPORT_SCHEMA = "t5gemma2-typed-fold-single-seed-promotion-gate-v1"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    delegated = argparse.Namespace(
        pass3_score=args.fold_score,
        expected_pass3_score_sha256=args.expected_fold_score_sha256,
        pass3_predictions=args.fold_predictions,
        expected_pass3_predictions_sha256=args.expected_fold_predictions_sha256,
        pass3_provenance=args.fold_provenance,
        expected_pass3_provenance_sha256=args.expected_fold_provenance_sha256,
        update58_score=args.update58_score,
        expected_update58_score_sha256=args.expected_update58_score_sha256,
        update58_predictions=args.update58_predictions,
        expected_update58_predictions_sha256=args.expected_update58_predictions_sha256,
        update58_provenance=args.update58_provenance,
        expected_update58_provenance_sha256=args.expected_update58_provenance_sha256,
        collapse_checker=args.collapse_checker,
        expected_collapse_checker_sha256=args.expected_collapse_checker_sha256,
        output="unused",
    )
    report = shared.audit(delegated)
    report["schema"] = REPORT_SCHEMA
    report["arm"] = "typed_fold_rs_sft_union_v1"
    report["shared_gate_logic"] = {
        "path": str(Path(shared.__file__).resolve()),
        "sha256": sha256_file(Path(shared.__file__).resolve()),
        "semantics": "same_full175_k10_seed42_diversity_and_pairing_gate_as_pass3",
    }
    report["inputs"]["fold"] = report["inputs"].pop("pass3")
    report["metrics"]["fold"] = report["metrics"].pop("pass3")
    report["decision"]["promotion_status"] = "HOLD_REQUIRES_3PLUS_MATCHED_SEEDS"
    report["decision"]["promoted_checkpoint"] = None
    report["decision"]["verpo_status"] = "HOLD"
    report["automatic_promotion_performed"] = False
    report["verpo_launched"] = False
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fold-score", required=True)
    parser.add_argument("--expected-fold-score-sha256", required=True)
    parser.add_argument("--fold-predictions", required=True)
    parser.add_argument("--expected-fold-predictions-sha256", required=True)
    parser.add_argument("--fold-provenance", required=True)
    parser.add_argument("--expected-fold-provenance-sha256", required=True)
    parser.add_argument("--update58-score", required=True)
    parser.add_argument("--expected-update58-score-sha256", required=True)
    parser.add_argument("--update58-predictions", required=True)
    parser.add_argument("--expected-update58-predictions-sha256", required=True)
    parser.add_argument("--update58-provenance", required=True)
    parser.add_argument("--expected-update58-provenance-sha256", required=True)
    parser.add_argument("--collapse-checker", required=True)
    parser.add_argument("--expected-collapse-checker-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit(args)
    require_exact_or_write(Path(args.output).expanduser().resolve(), report)
    print(__import__("json").dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
