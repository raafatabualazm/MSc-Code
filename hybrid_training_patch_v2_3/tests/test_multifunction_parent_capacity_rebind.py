from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.preprocessing.rebind_multifunction_parent_capacity import (
    CapacityRebindError,
    rebind_parent_capacity,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class MultifunctionParentCapacityRebindTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_dataset = self.root / "source.jsonl"
        self.target_dataset = self.root / "target.jsonl"
        data = '{"task_id":"a"}\n{"task_id":"b"}\n'
        self.source_dataset.write_text(data, encoding="utf-8")
        self.target_dataset.write_text(data, encoding="utf-8")
        self.source_contract = self.root / "source_contract.json"
        self.target_contract = self.root / "target_contract.json"
        _write_json(
            self.source_contract,
            {
                "fixed_representation": "v2",
                "max_target_tokens": 12_288,
                "max_total_tokens": 24_576,
            },
        )
        _write_json(
            self.target_contract,
            {
                "fixed_representation": "v2",
                "max_target_tokens": 24_576,
                "max_total_tokens": 36_864,
            },
        )
        self.source_seal = self.root / "source_rich.seal.json"
        source_seal = {
            "schema": "compact-public-private-join-seal-v1",
            "selected_role": "fit",
            "rows": 2,
            "output_sha256": _sha(self.source_dataset),
            "contract_sha256": _sha(self.source_contract),
            "output": {
                "path": str(self.source_dataset.resolve()),
                "sha256": _sha(self.source_dataset),
                "bytes": self.source_dataset.stat().st_size,
            },
            "rich_expansion_provenance": {"membership": "sealed"},
        }
        _write_json(self.source_seal, source_seal)
        self.generic_target_seal = self.root / "target_generic.seal.json"
        _write_json(
            self.generic_target_seal,
            {
                "schema": "compact-public-private-join-seal-v1",
                "selected_role": "fit",
                "rows": 2,
                "output_sha256": _sha(self.target_dataset),
                "contract_sha256": _sha(self.target_contract),
            },
        )
        self.output_seal = self.root / "target_rich.seal.json"
        self.output_receipt = self.root / "target_rich.receipt.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self) -> dict[str, object]:
        return rebind_parent_capacity(
            source_rich_seal=self.source_seal,
            source_dataset=self.source_dataset,
            source_contract=self.source_contract,
            target_dataset=self.target_dataset,
            target_contract=self.target_contract,
            generic_target_seal=self.generic_target_seal,
            output_seal=self.output_seal,
            output_receipt=self.output_receipt,
        )

    def test_rebind_changes_only_contract_and_is_idempotent(self) -> None:
        first = self._run()
        source = json.loads(self.source_seal.read_text(encoding="utf-8"))
        target = json.loads(self.output_seal.read_text(encoding="utf-8"))
        observed_contract = target.pop("contract_sha256")
        expected_contract = source.pop("contract_sha256")
        self.assertNotEqual(observed_contract, expected_contract)
        self.assertEqual(observed_contract, _sha(self.target_contract))
        self.assertEqual(target, source)
        self.assertTrue(
            first["invariants"]["rich_target_seal_changed_only_contract_sha256"]
        )
        second = self._run()
        self.assertEqual(first, second)

    def test_rejects_dataset_difference(self) -> None:
        self.target_dataset.write_text('{"task_id":"other"}\n', encoding="utf-8")
        with self.assertRaisesRegex(
            CapacityRebindError, "not byte-identical"
        ):
            self._run()

    def test_rejects_non_capacity_contract_change(self) -> None:
        target = json.loads(self.target_contract.read_text(encoding="utf-8"))
        target["fixed_representation"] = "changed"
        _write_json(self.target_contract, target)
        generic = json.loads(
            self.generic_target_seal.read_text(encoding="utf-8")
        )
        generic["contract_sha256"] = _sha(self.target_contract)
        _write_json(self.generic_target_seal, generic)
        with self.assertRaisesRegex(
            CapacityRebindError, "capacity"
        ):
            self._run()


if __name__ == "__main__":
    unittest.main()
