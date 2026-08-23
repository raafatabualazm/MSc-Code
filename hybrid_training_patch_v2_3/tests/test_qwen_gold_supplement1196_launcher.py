from __future__ import annotations

import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    WORKSPACE_ROOT
    / "fixed_training_launchers"
    / "run_qwen38_gold_supplement_1196.sh"
)
WRAPPER = (
    WORKSPACE_ROOT
    / "fixed_training_launchers"
    / "qwen38_gold_supplement1196_supervisor.sh"
)
SUPERVISOR = (
    WORKSPACE_ROOT
    / "fixed_training_launchers"
    / "qwen38_gold_supplement1196.supervisor.conf"
)


class QwenGoldSupplement1196LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")
        cls.wrapper = WRAPPER.read_text(encoding="utf-8")
        cls.supervisor = SUPERVISOR.read_text(encoding="utf-8")

    def test_output_and_lock_match_complete_union_launcher(self) -> None:
        self.assertIn(
            '${WORKSPACE}/locks/qwen38_union2776.lock',
            self.launcher,
        )
        self.assertIn(
            '${UNION_ROOT}/direct_compact_multifunction_gold_sft_union2776',
            self.launcher,
        )
        self.assertIn(
            '${EXPANDED_ROOT}/supplement1196_multifunction_binary.jsonl',
            self.launcher,
        )
        self.assertIn("--expected-train-rows 1196", self.launcher)
        self.assertIn("--epochs 1.0", self.launcher)
        self.assertIn("--learning_rate 2e-5", self.launcher)

    def test_only_gold_continuation_is_trainable(self) -> None:
        self.assertEqual(
            self.launcher.count(
                '"${PYTHON}" -m scripts.training.direct_compact_qwen_decompiler'
            ),
            1,
        )
        self.assertNotIn("SEQUENCE_TRAIN=", self.launcher)
        self.assertNotIn("COT_TRAIN=", self.launcher)
        self.assertNotIn("sequence_forward_kl", self.launcher)
        self.assertNotIn("qwen_cot_sft", self.launcher)
        self.assertNotIn("--sequence_distribution_nll", self.launcher)

    def test_training_is_fit_only_and_resumable(self) -> None:
        self.assertIn("--no_eval_during_training", self.launcher)
        self.assertNotIn("--eval_file", self.launcher)
        self.assertIn("--resume_from_checkpoint auto", self.launcher)
        self.assertIn(
            'source.get("heldout_loaded_during_training") is not False',
            self.launcher,
        )
        self.assertIn(
            'target.get("heldout_loaded_during_migration") is not False',
            self.launcher,
        )
        self.assertNotIn("HELDOUT_DATASET=", self.launcher)
        self.assertNotIn("HELDOUT_SEAL=", self.launcher)

    def test_supervisor_runs_the_sealed_launcher(self) -> None:
        self.assertIn(
            "/workspace/fixed_training_launchers/"
            "run_qwen38_gold_supplement_1196.sh",
            self.wrapper,
        )
        self.assertIn(
            "command=/opt/supervisor-scripts/"
            "qwen38_gold_supplement1196.sh",
            self.supervisor,
        )
        self.assertIn("autostart=false", self.supervisor)
        self.assertIn("autorestart=unexpected", self.supervisor)
        self.assertIn("stopasgroup=true", self.supervisor)
        self.assertIn("killasgroup=true", self.supervisor)


if __name__ == "__main__":
    unittest.main()
