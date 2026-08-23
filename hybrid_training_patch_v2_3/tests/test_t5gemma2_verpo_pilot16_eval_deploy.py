from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "deploy" / "vast" / "t5gemma2_verpo_pilot16_eval.sh"
CONFIG = ROOT / "deploy" / "vast" / "t5gemma2-verpo-pilot16-eval.conf"


def test_eval_requires_completed_exact_sixteen_update_pilot() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "checkpoint-optstep-000016" in text
    assert "result.get(\"updates\") != 16" in text
    assert "result.get(\"status\") != \"complete\"" in text
    assert (
        "5d2f91531938079dfa032741aeef3161607d378274d6c92f60d81c184f8a7c86"
        in text
    )
    assert "len(lines) != 16" in text
    assert 'train_state}" != EXITED' in text
    assert "supervisorctl start" not in text
    assert "supervisorctl restart" not in text


def test_eval_reuses_exact_historical_generation_and_scoring_contract() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for fragment in (
        "564993a53a7f5891749f76f349bb6e41531d2a4cbdc2d721a41be21679d793d9",
        "2d2d0d40eac8061290427c585be6385f147d002d82def912af88bca3a3a8fe19",
        "249a173a89d5094a293105c0df7b947a73785f36e722159d265a4c8f5dbba7c6",
        "16f27a9d96df73e4e5c3e4f43ced4cd3b46574bf3dc9cceb5beadb382c76e14d",
        "e98d2f7dea3d12a17a4287d77ba324b48e50bff0ba3ca62c765bd85349b43334",
        "--num_samples 10",
        "--generation_batch_size 10",
        "--max_source_tokens 32768",
        "--max_new_tokens 4096",
        "--temperature 0.8",
        "--top_p 0.95",
        "--seed 42",
        "--k 10",
        "--timeout 30",
        "--stability_runs 2",
    ):
        assert fragment in text
    assert "sampling[0] == sampling[1]" in text
    assert "slot_coordinates[0] == slot_coordinates[1]" in text
    assert "score_contracts[0] == score_contracts[1]" in text
    assert "task_orders[0] == task_orders[1]" in text


def test_eval_never_serializes_heldout_tests_or_targets_to_model() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "dev_multifunction_binary.seal.json" in text
    assert "dev_multifunction_binary_f2.jsonl.manifest.json" in text
    assert 'provenance.get("tests_exposed_to_model") is not False' in text
    assert 'provenance.get("targets_exposed_to_model") is not False' in text
    assert '"tests_exposed_to_model": False' in text
    assert "same_task_order_and_sources" in text


def test_eval_supervisor_job_is_manual_and_fail_closed() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "[program:t5gemma-verpo-pilot16-eval]" in text
    assert "autostart=false" in text
    assert "autorestart=unexpected" in text
    assert "exitcodes=0,78" in text
    assert "stopasgroup=true" in text
    assert "killasgroup=true" in text
