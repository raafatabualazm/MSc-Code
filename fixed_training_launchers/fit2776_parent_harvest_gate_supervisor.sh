#!/bin/bash
# Stop the legacy 1,580-task service only after its complete K=8 parent
# artifacts are durably materialized. The expanded union consumes them later.
set -Eeuo pipefail

utils=/opt/supervisor-scripts/utils
# shellcheck disable=SC1091
. "${utils}/logging.sh" ""
# shellcheck disable=SC1091
. "${utils}/environment.sh"

source /venv/main/bin/activate
cd /workspace

root=/workspace/artifacts/direct_compact_qwen38_inline_cfg_v2
journal="${root}/qwen_teacher.journal.jsonl"
manifest="${root}/qwen_mc_sequence_train.build.json"

parent_ready() {
  /venv/main/bin/python - "${journal}" "${manifest}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

journal = Path(sys.argv[1])
manifest = Path(sys.argv[2])
if not journal.is_file() or not manifest.is_file():
    raise SystemExit(1)
terminal = set()
with journal.open("r", encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(1)
        if row.get("event") == "teacher_slot_terminal":
            terminal.add((str(row.get("task_id") or ""), int(row.get("sample_index", -1))))
if len(terminal) != 12_640:
    raise SystemExit(1)
value = json.loads(manifest.read_text(encoding="utf-8"))
counts = value.get("counts") or {}
record = (value.get("inputs") or {}).get("teacher_journal") or {}
digest = hashlib.sha256(journal.read_bytes()).hexdigest()
if (
    value.get("schema") != "direct-compact-mc-sequence-forward-kl-nll-build-v1"
    or int(counts.get("teacher_draw_rows", -1)) != 12_640
    or int(counts.get("output_rows", -1)) != 12_640
    or int(counts.get("gold_replay_rows", -1)) != 0
    or record.get("sha256") != digest
):
    raise SystemExit(1)
print(digest)
PY
}

while ! journal_sha256="$(parent_ready)"; do
  status="$(supervisorctl status qwen38_kd 2>/dev/null || true)"
  if [[ "${status}" == *"FATAL"* ]]; then
    printf 'FIT2776_PARENT_GATE_FATAL status=%s\n' "${status}" >&2
    exit 2
  fi
  printf 'FIT2776_PARENT_GATE_WAIT status=%s\n' "${status:-unknown}"
  sleep 30
done

status="$(supervisorctl status qwen38_kd 2>/dev/null || true)"
if [[ "${status}" == *"RUNNING"* || "${status}" == *"STARTING"* ]]; then
  supervisorctl stop qwen38_kd
fi
printf 'FIT2776_PARENT_HARVEST_SEALED tasks=1580 draws=12640 journal_sha256=%s legacy_sequence_gpu_skipped=true\n' \
  "${journal_sha256}"
