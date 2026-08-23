from __future__ import annotations

import unittest

from scripts.preprocessing.build_multifunction_executable_view import (
    ExecutableViewError,
)
from scripts.preprocessing.migrate_multifunction_executable_capacity import (
    validate_capacity_only_contract_change,
)


class ExecutableCapacityContractTests(unittest.TestCase):
    def test_accepts_exact_capacity_increase(self) -> None:
        source = {
            "schema": "contract-v1",
            "max_source_tokens": 9000,
            "max_target_tokens": 3072,
            "max_total_tokens": 12288,
            "source_token_expansions": {"1": [2, 3]},
        }
        target = dict(source) | {
            "max_target_tokens": 24576,
            "max_total_tokens": 36864,
        }
        result = validate_capacity_only_contract_change(source, target)
        self.assertEqual(
            result["observed_changed_fields"],
            ["max_target_tokens", "max_total_tokens"],
        )
        self.assertTrue(
            result["all_non_capacity_fields_byte_semantically_identical"]
        )

    def test_rejects_non_capacity_change(self) -> None:
        source = {
            "schema": "contract-v1",
            "max_target_tokens": 3072,
            "max_total_tokens": 12288,
            "codec_sha256": "a" * 64,
        }
        target = dict(source) | {
            "max_target_tokens": 24576,
            "max_total_tokens": 36864,
            "codec_sha256": "b" * 64,
        }
        with self.assertRaisesRegex(
            ExecutableViewError, "non-capacity contract fields"
        ):
            validate_capacity_only_contract_change(source, target)

    def test_rejects_missing_or_decreasing_capacity_change(self) -> None:
        source = {
            "schema": "contract-v1",
            "max_target_tokens": 3072,
            "max_total_tokens": 12288,
        }
        with self.assertRaisesRegex(
            ExecutableViewError, "must change exactly"
        ):
            validate_capacity_only_contract_change(
                source,
                dict(source) | {"max_target_tokens": 24576},
            )
        with self.assertRaisesRegex(
            ExecutableViewError, "positive strict capacity increase"
        ):
            validate_capacity_only_contract_change(
                source,
                dict(source)
                | {
                    "max_target_tokens": 1024,
                    "max_total_tokens": 36864,
                },
            )


if __name__ == "__main__":
    unittest.main()
