from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from scripts.training.t5gemma2_api_rs_sft_rescue import schedule_capacity


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "deploy" / "vast" / "t5gemma2_api_rs_sft_claude_production.sh"
CONFIG_PATH = ROOT / "deploy" / "vast" / "t5gemma2-api-rs-sft-claude-production.conf"
PROBE_LAUNCHER_PATH = ROOT / "deploy" / "vast" / "t5gemma2_api_rs_sft_claude_probe.sh"
TRANCHE2_LAUNCHER_PATH = (
    ROOT / "deploy" / "vast" / "t5gemma2_api_rs_sft_claude_production_tranche2.sh"
)
TRANCHE2_CONFIG_PATH = (
    ROOT / "deploy" / "vast" / "t5gemma2-api-rs-sft-claude-production-tranche2.conf"
)
OPUS_RESIDUAL_LAUNCHER_PATH = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2_api_rs_sft_claude_opus_production_residual_probe.sh"
)
OPUS_RESIDUAL_CONFIG_PATH = (
    ROOT
    / "deploy"
    / "vast"
    / "t5gemma2-api-rs-sft-claude-opus-production-residual-probe.conf"
)


def test_production_launcher_consumes_only_completed_two_epoch_pilot() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "${WORKSPACE}/artifacts/t5gemma2_local_rs_sft_pilot_2epoch_v1" in launcher
    assert 'LOCAL_PILOT_REPORT="${LOCAL_PILOT_DIR}/harvest_report.json"' in launcher
    assert '.schema == "t5gemma2-local-rs-sft-pilot-report-v1"' in launcher
    assert '.status == "complete"' in launcher
    assert '--pilot_journal "${LOCAL_PILOT_JOURNAL}"' in launcher
    assert "--exploratory_terminal_prefix" not in launcher
    for digest in (
        "5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58",
        "9f670cf606f7fc68e157508e2e064a1954280fef9063c6ec5239e3e6ca63be1d",
        "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab",
        "21613e2c7513e203e31a4690f84b0e6d11fa1c7fa6a20725d859486a30bccac3",
    ):
        assert digest in launcher
    assert ".pilot.tasks == 200" in launcher
    assert ".pilot.accepted_unique_targets == 9" in launcher


def test_production_launcher_is_isolated_and_has_exact_sonnet_contract() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "t5gemma2_api_rs_sft_claude_production_2epoch_v1" in launcher
    assert "t5gemma2_api_rs_sft_claude_probe_prefix10_v1" not in launcher
    assert "--provider anthropic" in launcher
    assert "--model claude-sonnet-5" in launcher
    assert "--anthropic_thinking adaptive" in launcher
    assert "--anthropic_effort high" in launcher
    assert "--max_tasks 188" in launcher
    assert "--max_calls 188" in launcher
    assert "--max_parents_per_task 1" in launcher
    assert "--samples_per_parent 1" in launcher
    assert "--max_output_tokens 8192" in launcher
    assert "--max_usd 13.25" in launcher
    assert "--input_usd_per_million 2" in launcher
    assert "--output_usd_per_million 10" in launcher


def test_production_launcher_has_explicit_aggregate_token_caps() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--max_input_tokens_per_call 49152" in launcher
    assert "--max_input_tokens_total 9240576" in launcher
    assert "--max_output_tokens_total 1540096" in launcher
    assert "--max_total_tokens 10780672" in launcher

    capacity, record = schedule_capacity(
        max_calls=188,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=8192,
        max_input_tokens_total=9240576,
        max_output_tokens_total=1540096,
        max_total_tokens=10780672,
        max_usd=Decimal("13.25"),
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
    )
    # Dollar reservation is binding: 73 worst-case calls cost $13.156352,
    # while 74 would cost $13.336576 and violate the hard cap.
    assert capacity == 73
    assert record["reservation_policy"] == (
        "full_per_call_caps_for_every_scheduled_call"
    )


def test_production_launcher_preserves_secret_and_holdback_boundaries() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "${WORKSPACE}/secrets/Anthropic.env" in launcher
    assert "stat.S_IMODE(path.stat().st_mode) & 0o077" in launcher
    assert "source ${SECRET_FILE}" not in launcher
    assert 'export ANTHROPIC_API_KEY="${anthropic_key}"' in launcher
    assert "unset anthropic_key" in launcher
    assert (
        '--private_holdback "${FEEDBACK_DIR}/reward_holdback.private.jsonl"' in launcher
    )
    assert "--stability_runs 2" in launcher


def test_supervisor_config_targets_only_production_launcher() -> None:
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert "[program:t5gemma-api-rs-sft-claude-production]" in config
    assert (
        "command=/opt/supervisor-scripts/" "t5gemma2_api_rs_sft_claude_production.sh"
    ) in config
    assert "autostart=false" in config
    assert "t5gemma-api-rs-sft-claude-probe" not in config


def test_existing_probe_launcher_remains_unchanged_in_scope() -> None:
    probe = PROBE_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--exploratory_terminal_prefix 10" in probe
    assert "--max_tasks 5" in probe
    assert "--max_calls 5" in probe
    assert "--max_usd 0.95" in probe


def test_tranche2_is_bound_to_exact_completed_tranche1_and_pilot() -> None:
    launcher = TRANCHE2_LAUNCHER_PATH.read_text(encoding="utf-8")

    for digest in (
        # Completed two-epoch local pilot journal, chain head, and report.
        "5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58",
        "9f670cf606f7fc68e157508e2e064a1954280fef9063c6ec5239e3e6ca63be1d",
        "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab",
        # Completed tranche-one journal, chain head, and report.
        "b2b6dfbb3d0a3efd5cbadee09e134c24fa7594f6df1238833d25a7b671c9af10",
        "61108ea2c34fc7776c5b5797d103429a6ac979fae5f3d0385d74ce807f69afae",
        "fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad",
        "056834fb23af50bc14222254baec5c985b3223179698e34429a408c989a7ccf7",
        "b142fc681a538e2d7356caba0a3a7ce5fc0f4edf435f06b8ccef789e9cd1cf0e",
        # Exact rollout, F2, and complementary private holdback.
        "14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c",
        "c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3",
        "dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f",
    ):
        assert digest in launcher
    assert '.status == "complete"' in launcher
    assert ".schedule.scheduled_tasks == 73" in launcher
    assert ".schedule.scheduled_calls == 73" in launcher
    assert ".verification.verified_unique_hard_targets == 23" in launcher
    assert '.budget_charged.estimated_usd == "3.914444000"' in launcher
    assert ".contract.selection.seed == 42" in launcher
    assert ".contract.selection.max_tasks_before_budget == 188" in launcher


def test_tranche2_has_disjoint_offset_and_exact_sonnet_contract() -> None:
    launcher = TRANCHE2_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert (
        'OUTPUT_DIR="${T5GEMMA_CLAUDE_PRODUCTION_TRANCHE2_OUTPUT_DIR:'
        "-${WORKSPACE}/artifacts/"
        't5gemma2_api_rs_sft_claude_production_2epoch_tranche2_v1}"' in launcher
    )
    assert "--provider anthropic" in launcher
    assert "--model claude-sonnet-5" in launcher
    assert "--anthropic_thinking adaptive" in launcher
    assert "--anthropic_effort high" in launcher
    assert "--seed 42" in launcher
    assert "--eligible_task_offset 73" in launcher
    assert "--max_tasks 115" in launcher
    assert "--max_calls 115" in launcher
    assert "--max_parents_per_task 1" in launcher
    assert "--samples_per_parent 1" in launcher
    assert "--max_output_tokens 8192" in launcher
    assert "--max_usd 21.00" in launcher
    assert "--input_usd_per_million 2" in launcher
    assert "--output_usd_per_million 10" in launcher


def test_tranche2_reserves_all_115_calls_below_authorized_total() -> None:
    launcher = TRANCHE2_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--max_input_tokens_per_call 49152" in launcher
    assert "--max_input_tokens_total 5652480" in launcher
    assert "--max_output_tokens_total 942080" in launcher
    assert "--max_total_tokens 6594560" in launcher

    capacity, record = schedule_capacity(
        max_calls=115,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=8192,
        max_input_tokens_total=5652480,
        max_output_tokens_total=942080,
        max_total_tokens=6594560,
        max_usd=Decimal("21.00"),
        input_usd_per_million=Decimal("2"),
        output_usd_per_million=Decimal("10"),
    )
    assert capacity == 115
    assert record["worst_case_usd_nanos_per_call"] == 180_224_000
    tranche2_reservation = (
        Decimal(record["worst_case_usd_nanos_per_call"])
        * Decimal(capacity)
        / Decimal(1_000_000_000)
    )
    assert tranche2_reservation == Decimal("20.725760")
    total_exposure = Decimal("0.310832") + Decimal("3.914444") + tranche2_reservation
    assert total_exposure == Decimal("24.951036")
    assert total_exposure < Decimal("30")
    assert "$24.951036" in launcher


def test_tranche2_preserves_secret_and_holdback_boundaries() -> None:
    launcher = TRANCHE2_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "${WORKSPACE}/secrets/Anthropic.env" in launcher
    assert "stat.S_IMODE(path.stat().st_mode) & 0o077" in launcher
    assert "source ${SECRET_FILE}" not in launcher
    assert 'export ANTHROPIC_API_KEY="${anthropic_key}"' in launcher
    assert "unset anthropic_key" in launcher
    assert (
        "--private_holdback "
        '"${FEEDBACK_DIR}/reward_holdback.private.jsonl"' in launcher
    )
    assert "--stability_runs 2" in launcher
    assert ".privacy_invariants.private_holdback_sent_to_provider == false" in launcher
    assert ".privacy_invariants.gold_sent_to_provider == false" in launcher


def test_tranche2_supervisor_config_is_isolated_and_manual() -> None:
    config = TRANCHE2_CONFIG_PATH.read_text(encoding="utf-8")

    assert "[program:t5gemma-api-rs-sft-claude-production-tranche2]" in config
    assert (
        "command=/opt/supervisor-scripts/"
        "t5gemma2_api_rs_sft_claude_production_tranche2.sh"
    ) in config
    assert "autostart=false" in config
    assert (
        "stdout_logfile=/workspace/logs/"
        "t5gemma-api-rs-sft-claude-production-tranche2.log"
    ) in config


def test_opus_residual_probe_pins_completed_pilot_and_both_sonnet_runs() -> None:
    launcher = OPUS_RESIDUAL_LAUNCHER_PATH.read_text(encoding="utf-8")

    for digest in (
        # Pilot journal, chain head, and report.
        "5b2753c8f9d3b1fa403ff2352105f62657059118b30655955e0be34597941a58",
        "9f670cf606f7fc68e157508e2e064a1954280fef9063c6ec5239e3e6ca63be1d",
        "b6c47842f84a8a213015c900bd9ef9977dd42b58ac73489ed97b7845a989efab",
        # Sonnet tranche-one journal, chain head, and report.
        "b2b6dfbb3d0a3efd5cbadee09e134c24fa7594f6df1238833d25a7b671c9af10",
        "61108ea2c34fc7776c5b5797d103429a6ac979fae5f3d0385d74ce807f69afae",
        "fe51ceca919a13b1d39a54263dd3c394e8feff9f0ceb9b27b1d7cf199d54d1ad",
        "2bbd8ccc486734a7aed738e9cb705105e79778162f2fbb99798895e9142611d3",
        # Sonnet tranche-two journal, chain head, and report.
        "4bdeb9e6f5a0d3063b6d454d91bde65596ef788a7edd08d67045fa545b6481d6",
        "1acc0053ccf0f03553dbbe5477fed09e079261c112984eb4cb35a673032a4ba2",
        "99c0b04099d83fff0af79b36c4aa0248161fedbd1e3ec7992509d23de25f2da4",
        "e31d438ade29469b5a742c16f4dc4708b6b8491a6aa8843fad29ee20d8114b1b",
        # Exact rollout, F2, and complementary private holdback.
        "14139ed29281ffcf9a713d4ee09fb8d0f67dff613bb170c09c2a7f5c62a6252c",
        "c3b0a25678eb531cc54f73e5e46515b6f869a8e3a197a6d36a6ff412823689c3",
        "dbc21d2ba875ea4532a0602d2d07b0457eb99b1ff906c3e4613f9608e5e0ae3f",
    ):
        assert digest in launcher
    assert '--prior_success_report "${TRANCHE1_REPORT}"' in launcher
    assert '--prior_success_report "${TRANCHE2_REPORT}"' in launcher
    assert "--require_prior_schedules_disjoint" in launcher
    assert "--require_prior_schedule_complete_coverage" in launcher


def test_opus_residual_probe_selects_exact_123_to_20_contract() -> None:
    launcher = OPUS_RESIDUAL_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--expected_prior_scheduled_tasks 188" in launcher
    assert "--expected_prior_verified_tasks 65" in launcher
    assert "--expected_residual_tasks 123" in launcher
    assert (
        "--expected_prior_scheduled_task_ids_sha256 "
        "a52cacfb325a927dc758fca3e03608dfdb479b108f26791f2b916983f1de6994" in launcher
    )
    assert (
        "--expected_prior_verified_task_ids_sha256 "
        "2797d436c1b596a6771b9c1494c15ce95cf91046c17a04f081c0847dad36bc7a" in launcher
    )
    assert (
        "--expected_residual_task_ids_sha256 "
        "262532ddcc7ee9b1b03125d7600d96e0ded5886971038985061523c0fcf1c4e6" in launcher
    )
    assert (
        "--expected_scheduled_task_ids_sha256 "
        "9ddf1a24954de70810a94cf44ae0b1ebc8fc13ca50703a2b1715325855451f8f" in launcher
    )
    assert "--model claude-opus-5" in launcher
    assert "--anthropic_thinking adaptive" in launcher
    assert "--anthropic_effort high" in launcher
    assert "--max_tasks 20" in launcher
    assert "--max_calls 20" in launcher
    assert "--max_parents_per_task 1" in launcher
    assert "--samples_per_parent 1" in launcher
    assert "--max_output_tokens 8192" in launcher


def test_opus_residual_probe_reserves_exact_twenty_call_budget() -> None:
    launcher = OPUS_RESIDUAL_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "--max_input_tokens_per_call 49152" in launcher
    assert "--max_input_tokens_total 983040" in launcher
    assert "--max_output_tokens_total 163840" in launcher
    assert "--max_total_tokens 1146880" in launcher
    assert "--max_usd 9.0112" in launcher
    assert "--input_usd_per_million 5" in launcher
    assert "--output_usd_per_million 25" in launcher

    capacity, record = schedule_capacity(
        max_calls=20,
        max_input_tokens_per_call=49152,
        max_output_tokens_per_call=8192,
        max_input_tokens_total=983040,
        max_output_tokens_total=163840,
        max_total_tokens=1146880,
        max_usd=Decimal("9.0112"),
        input_usd_per_million=Decimal("5"),
        output_usd_per_million=Decimal("25"),
    )
    assert capacity == 20
    assert record["worst_case_usd_nanos_per_call"] == 450_560_000
    reservation = (
        Decimal(record["worst_case_usd_nanos_per_call"])
        * Decimal(capacity)
        / Decimal(1_000_000_000)
    )
    assert reservation == Decimal("9.011200")
    assert Decimal("9.981860") + reservation == Decimal("18.993060")
    assert Decimal("18.993060") < Decimal("30")


def test_opus_residual_probe_is_isolated_private_and_manual() -> None:
    launcher = OPUS_RESIDUAL_LAUNCHER_PATH.read_text(encoding="utf-8")
    config = OPUS_RESIDUAL_CONFIG_PATH.read_text(encoding="utf-8")

    assert (
        "t5gemma2_api_rs_sft_claude_opus_production_residual_probe_2epoch_v1"
        in launcher
    )
    assert "--exploratory_terminal_prefix" not in launcher
    assert "${WORKSPACE}/secrets/Anthropic.env" in launcher
    assert "source ${SECRET_FILE}" not in launcher
    assert (
        "--private_holdback "
        '"${FEEDBACK_DIR}/reward_holdback.private.jsonl"' in launcher
    )
    assert "--stability_runs 2" in launcher
    assert (
        "[program:t5gemma-api-rs-sft-claude-opus-production-residual-probe]" in config
    )
    assert (
        "command=/opt/supervisor-scripts/"
        "t5gemma2_api_rs_sft_claude_opus_production_residual_probe.sh" in config
    )
    assert "autostart=false" in config
