from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

TRAINING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "training"
sys.path.insert(0, str(TRAINING_DIR))

from true_distribution_kd_antigravity import (  # noqa: E402
    _batch_loss,
    _validate_checkpoint_binding,
)


class _Body(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyCausalLm(torch.nn.Module):
    def __init__(self, vocab: int = 5, hidden: int = 4) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.model = _Body(self.embedding)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)

    def get_base_model(self):
        return self

    def get_output_embeddings(self):
        return self.lm_head


def _args(mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        mode=mode,
        dtype="fp32",
        position_chunk_size=1,
        temperature=2.0 if mode == "dense_full_kl" else 1.0,
        kd_weight=1.0,
        hard_ce_weight=0.1,
    )


class TrainerCoreTests(unittest.TestCase):
    def test_checkpoint_binding_uses_contract_semantics_not_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tokenizer_sha = "a" * 64
            payload = {
                "schema": "direct-compact-causal-v1",
                "tokenizer_json_sha256": tokenizer_sha,
                "base_vocab_size": 5,
            }
            requested_contract = root / "requested.json"
            saved_contract = root / "saved.json"
            saved_tokenizer = root / "tokenizer.json"
            requested_contract.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
            saved_contract.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            saved_tokenizer.write_text(
                json.dumps({"model": {"vocab": {}}}),
                encoding="utf-8",
            )
            layout = {
                "contract": saved_contract,
                "tokenizer_json": saved_tokenizer,
            }
            _validate_checkpoint_binding(
                layout,
                identity="test",
                contract_path=requested_contract,
                tokenizer_sha256=tokenizer_sha,
            )
            changed = dict(payload, base_vocab_size=6)
            saved_contract.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs semantically"):
                _validate_checkpoint_binding(
                    layout,
                    identity="test",
                    contract_path=requested_contract,
                    tokenizer_sha256=tokenizer_sha,
                )
            saved_contract.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different tokenizer"):
                _validate_checkpoint_binding(
                    layout,
                    identity="test",
                    contract_path=requested_contract,
                    tokenizer_sha256="b" * 64,
                )

    def test_dense_loss_uses_every_supervised_position_and_backpropagates(
        self,
    ) -> None:
        torch.manual_seed(3)
        student = _TinyCausalLm()
        teacher = _TinyCausalLm()
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 2, 3]]),
        }
        loss, metrics = _batch_loss(
            _args("dense_full_kl"),
            student=student,
            teacher=teacher,
            batch=batch,
        )
        self.assertEqual(metrics["positions"], 2.0)
        self.assertTrue(math.isfinite(metrics["kd_loss"]))
        loss.backward()
        self.assertIsNotNone(student.lm_head.weight.grad)
        self.assertGreater(float(student.lm_head.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_sparse_loss_aligns_topk_tail_with_target_positions(self) -> None:
        torch.manual_seed(4)
        student = _TinyCausalLm()
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 2, 3]]),
            "teacher_top_token_ids": torch.tensor([[[2, 1], [3, 4]]]),
            "teacher_top_logprobs": torch.tensor(
                [
                    [
                        [math.log(0.6), math.log(0.2)],
                        [math.log(0.7), math.log(0.1)],
                    ]
                ],
                dtype=torch.float64,
            ),
            "teacher_top_mask": torch.ones(1, 2, 2, dtype=torch.bool),
            "teacher_tail_mass": torch.tensor([[0.2, 0.2]], dtype=torch.float64),
            "teacher_position_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        loss, metrics = _batch_loss(
            _args("sparse_topk_tail_kl"),
            student=student,
            teacher=None,
            batch=batch,
        )
        self.assertEqual(metrics["positions"], 2.0)
        self.assertTrue(math.isfinite(float(loss.detach())))
        loss.backward()
        self.assertGreater(float(student.lm_head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
