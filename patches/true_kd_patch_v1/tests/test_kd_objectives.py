from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

TRAINING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from kd_objectives_antigravity import (  # noqa: E402
    accumulation_window_size,
    dense_forward_kl,
    is_accumulation_boundary,
    sparse_topk_tail_forward_kl,
)


class DenseForwardKlTests(unittest.TestCase):
    def test_identical_distributions_have_zero_kl(self) -> None:
        logits = torch.tensor([[1.0, -0.5, 2.0]], requires_grad=True)
        loss = dense_forward_kl(logits, logits.detach(), temperature=2.0)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)

    def test_full_vocab_forward_kl_matches_manual_value(self) -> None:
        teacher_probs = torch.tensor([[0.5, 0.3, 0.2]])
        student_probs = torch.tensor([[0.4, 0.4, 0.2]])
        actual = dense_forward_kl(
            student_probs.log(),
            teacher_probs.log(),
            temperature=1.0,
        )
        expected = (
            teacher_probs
            * (teacher_probs.log() - student_probs.log())
        ).sum()
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_temperature_squared_scaling_matches_definition(self) -> None:
        student_logits = torch.tensor([[1.0, -0.5, 0.25]])
        teacher_logits = torch.tensor([[0.0, 2.0, -1.0]])
        temperature = 2.5
        actual = dense_forward_kl(
            student_logits,
            teacher_logits,
            temperature=temperature,
        )
        teacher_logp = torch.log_softmax(teacher_logits / temperature, dim=-1)
        student_logp = torch.log_softmax(student_logits / temperature, dim=-1)
        expected = (
            teacher_logp.exp() * (teacher_logp - student_logp)
        ).sum() * temperature**2
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_gradient_moves_student_toward_teacher(self) -> None:
        student_logits = torch.tensor([[0.0, 0.0]], requires_grad=True)
        teacher_logits = torch.tensor([[3.0, -3.0]])
        loss = dense_forward_kl(student_logits, teacher_logits)
        loss.backward()
        self.assertLess(float(student_logits.grad[0, 0]), 0.0)
        self.assertGreater(float(student_logits.grad[0, 1]), 0.0)


class SparseTopkTailKlTests(unittest.TestCase):
    def test_matches_kl_on_topk_plus_tail_partition(self) -> None:
        # Student probabilities are [0.4, 0.1, 0.3, 0.2].  Teacher exposes
        # token 0=0.5, token 2=0.2, and tail=0.3.
        student_probs = torch.tensor([[0.4, 0.1, 0.3, 0.2]])
        top_ids = torch.tensor([[0, 2]])
        teacher_top = torch.tensor([[math.log(0.5), math.log(0.2)]])
        teacher_tail = torch.tensor([0.3])
        actual = sparse_topk_tail_forward_kl(
            student_probs.log(),
            top_ids,
            teacher_top,
            teacher_tail,
        )
        expected = (
            0.5 * math.log(0.5 / 0.4)
            + 0.2 * math.log(0.2 / 0.3)
            + 0.3 * math.log(0.3 / 0.3)
        )
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_tail_is_not_renormalized_away(self) -> None:
        student_probs = torch.tensor([[0.2, 0.2, 0.2, 0.4]])
        loss = sparse_topk_tail_forward_kl(
            student_probs.log(),
            torch.tensor([[0, 1]]),
            torch.tensor([[math.log(0.4), math.log(0.1)]]),
            torch.tensor([0.5]),
        )
        expected = (
            0.4 * math.log(0.4 / 0.2)
            + 0.1 * math.log(0.1 / 0.2)
            + 0.5 * math.log(0.5 / 0.6)
        )
        self.assertAlmostEqual(float(loss), expected, places=6)

    def test_sparse_temperature_other_than_one_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature=1"):
            sparse_topk_tail_forward_kl(
                torch.zeros(1, 4),
                torch.tensor([[0]]),
                torch.tensor([[math.log(0.5)]]),
                torch.tensor([0.5]),
                temperature=2.0,
            )

    def test_duplicate_top_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            sparse_topk_tail_forward_kl(
                torch.zeros(1, 4),
                torch.tensor([[0, 0]]),
                torch.tensor([[math.log(0.2), math.log(0.2)]]),
                torch.tensor([0.6]),
            )

    def test_top_plus_tail_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            sparse_topk_tail_forward_kl(
                torch.zeros(1, 4),
                torch.tensor([[0]]),
                torch.tensor([[math.log(0.2)]]),
                torch.tensor([0.7]),
            )


class GradientAccumulationTests(unittest.TestCase):
    def test_remainder_window_uses_true_divisor(self) -> None:
        sizes = [
            accumulation_window_size(
                index, total_batches=10, gradient_accumulation_steps=4
            )
            for index in range(10)
        ]
        self.assertEqual(sizes, [4, 4, 4, 4, 4, 4, 4, 4, 2, 2])
        boundaries = [
            index
            for index in range(10)
            if is_accumulation_boundary(
                index, total_batches=10, gradient_accumulation_steps=4
            )
        ]
        self.assertEqual(boundaries, [3, 7, 9])


if __name__ == "__main__":
    unittest.main()
