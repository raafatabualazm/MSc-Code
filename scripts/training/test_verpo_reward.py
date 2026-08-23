"""Unit tests for the paper-faithful VeRPO group reward."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.graph_grpo_decompiler_antigravity import (
    reward_configuration,
    verpo_group_rewards,
)
from scripts.training import graph_grpo_decompiler_antigravity as grpo_module


def detail(test_passes, *, compiled=True, full_pass=False):
    passes = [bool(value) for value in test_passes]
    return {
        "reward": 0.0,
        "test_passes": passes,
        "compiled": compiled,
        "pass_ratio": sum(passes) / len(passes) if passes else 0.0,
        "passed": sum(passes),
        "total": len(passes),
        "full_pass": full_pass,
        "status": "tested",
    }


class ScriptedVeRPOReward:
    reward_mode = "verpo"
    verpo_alpha = 2.0
    verpo_anchor_weight = 1.0
    verpo_density_norm = True

    def __init__(self, details):
        self.details = iter(details)

    def compute_reward_details(self, _completion, _tests):
        return next(self.details)


class VeRPOGroupRewardTests(unittest.TestCase):
    def reward(self, group, *, density_norm=True, anchor_weight=1.0):
        return verpo_group_rewards(
            group,
            [-1.0] * len(group),
            alpha=2.0,
            anchor_weight=anchor_weight,
            density_norm=density_norm,
        )

    def test_partial_passes_break_an_all_binary_fail_tie(self):
        group = [
            detail([True, True, False, False]),
            detail([True, False, True, False]),
            detail([False, False, True, True]),
            detail([False, False, False, False]),
        ]

        rewards = self.reward(group)

        self.assertGreater(max(rewards) - min(rewards), 0.0)
        self.assertTrue(all(value > rewards[3] for value in rewards[:3]))

    def test_rare_test_pass_is_worth_more_than_common_test_pass(self):
        group = [
            detail([True, False]),
            detail([True, False]),
            detail([True, False]),
            detail([False, True]),
        ]

        rewards = self.reward(group)

        self.assertGreater(rewards[3], rewards[0])

    def test_gaussian_kde_changes_distinct_difficulty_weights(self):
        # Column pass rates are 1.00, 0.75, 0.50, and 0.25. An exact-cluster
        # correction would be inert because every rate is unique; Gaussian KDE
        # still accounts for neighboring difficulty levels.
        group = [
            detail([True, True, True, True], full_pass=True),
            detail([True, True, True, False]),
            detail([True, True, False, False]),
            detail([True, False, False, False]),
        ]

        with_kde = self.reward(group, density_norm=True)
        without_kde = self.reward(group, density_norm=False)

        self.assertNotEqual(with_kde, without_kde)
        self.assertLess(with_kde[3], without_kde[3])

    def test_only_true_full_suite_pass_receives_global_anchor(self):
        group = [
            detail([True, True, True], full_pass=True),
            detail([True, True, False]),
            detail([True, False, False]),
            detail([False, False, False]),
        ]

        local_only = self.reward(group, anchor_weight=0.0)
        anchored = self.reward(group, anchor_weight=1.0)

        self.assertAlmostEqual(anchored[0], local_only[0] + 1.0)
        for index in range(1, len(group)):
            self.assertAlmostEqual(anchored[index], local_only[index])
        self.assertGreater(anchored[0], max(anchored[1:]))

    def test_noncompiling_candidate_gets_no_arbitrary_static_penalty(self):
        group = [
            detail([True, False]),
            detail([False, True]),
            detail([False, False]),
            detail([False, False], compiled=False),
        ]
        original = [-1.0, -1.0, -1.0, -2.0]

        rewards = verpo_group_rewards(
            group,
            original,
            alpha=2.0,
            anchor_weight=1.0,
        )

        self.assertEqual(rewards[2], 0.0)
        self.assertEqual(rewards[3], 0.0)

    def test_identical_outcomes_remain_uniform(self):
        group = [detail([True, False, False]) for _ in range(4)]

        rewards = self.reward(group)

        self.assertTrue(all(value == rewards[0] for value in rewards))

    def test_group_without_test_evidence_passes_through(self):
        group = [detail([], compiled=False) for _ in range(3)]
        original = [-1.0, -2.0, -0.5]

        rewards = verpo_group_rewards(
            group,
            original,
            alpha=2.0,
            anchor_weight=1.0,
        )

        self.assertEqual(rewards, original)

    def test_invalid_hyperparameters_fail_closed(self):
        group = [detail([True]), detail([False])]
        original = [-1.0, -1.0]

        with self.assertRaises(ValueError):
            verpo_group_rewards(group, original, alpha=0.0, anchor_weight=1.0)
        with self.assertRaises(ValueError):
            verpo_group_rewards(group, original, alpha=2.0, anchor_weight=-1.0)
        with self.assertRaises(ValueError):
            verpo_group_rewards(
                group,
                original,
                alpha=2.0,
                anchor_weight=1.0,
                epsilon=0.0,
            )

    def test_reward_configuration_records_paper_settings(self):
        reward = SimpleNamespace(
            reward_mode="verpo",
            verpo_alpha=2.0,
            verpo_anchor_weight=1.0,
            verpo_density_norm=True,
        )

        self.assertEqual(
            reward_configuration(reward),
            {
                "difficulty_alpha": 2.0,
                "anchor_weight": 1.0,
                "density_normalization": "gaussian_kde",
                "advantage_normalization": "mean",
            },
        )

    def test_calculate_rewards_applies_verpo_to_complete_groups(self):
        group = [
            detail([True, False]),
            detail([False, True]),
            detail([False, False]),
            detail([True, True], full_pass=True),
        ]
        expected = self.reward(group)

        with patch.object(
            grpo_module, "_dart_per_test_reward", ScriptedVeRPOReward(group)
        ):
            rewards, stats = grpo_module.calculate_rewards(
                ["candidate solution"] * 4,
                references=[""] * 4,
                languages=["dart"] * 4,
                tests=["test harness"] * 4,
                return_stats=True,
                group_size=4,
            )

        for actual, wanted in zip(rewards.tolist(), expected):
            self.assertAlmostEqual(actual, wanted, places=6)
        self.assertEqual(stats["perfect_flags"], [0.0, 0.0, 0.0, 1.0])

    def test_calculate_rewards_rejects_incomplete_verpo_group(self):
        group = [
            detail([True, False]),
            detail([False, True]),
            detail([False, False]),
        ]

        with patch.object(
            grpo_module, "_dart_per_test_reward", ScriptedVeRPOReward(group)
        ):
            with self.assertRaisesRegex(ValueError, "divisible by group_size"):
                grpo_module.calculate_rewards(
                    ["candidate solution"] * 3,
                    references=[""] * 3,
                    languages=["dart"] * 3,
                    tests=["test harness"] * 3,
                    group_size=2,
                )


if __name__ == "__main__":
    unittest.main()
