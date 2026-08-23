import concurrent.futures
import hashlib
import json
import os
import shutil


SOURCE = "/workspace/artifacts"
DESTINATION = "/workspace/hf_public_t5gemma2_verpo_artifacts_v1"
LATEST_DESTINATION = "/workspace/hf_public_t5gemma2_verpo_latest_v1"
LATEST = {
    "t5gemma2_4b4b_compiler_verpo_pilot16_2epoch_v1": "checkpoint-optstep-000016",
    "t5gemma2_4b4b_compiler_verpo_smoke_2epoch_v1": "checkpoint-optstep-000001",
    "t5gemma2_4b4b_compiler_verpo_smoke_2epoch_v2": "checkpoint-optstep-000001",
    "t5gemma2_4b4b_compiler_verpo_smoke_2epoch_v3": "checkpoint-optstep-000001",
    "t5gemma2_4b4b_enriched_sft_2epoch_v1": "checkpoint-optstep-000348",
    "t5gemma2_4b4b_enriched_sft_v1": "checkpoint-optstep-000174",
    "t5gemma2_4b4b_mixed_rs_sft_final_v1": "checkpoint-optstep-000426",
    "t5gemma2_4b4b_mixed_rs_sft_kimi_pass2_v1": "checkpoint-optstep-000013",
    "t5gemma2_4b4b_typed_c2_verpo_pilot150_v1": "checkpoint-optstep-000150",
    "t5gemma2_4b4b_typed_contract_sft_2epoch_v1": "checkpoint-optstep-000348",
    "t5gemma2_4b4b_typed_direct_rs_sft_225_v1": "checkpoint-optstep-000058",
    "t5gemma2_4b4b_typed_direct_rs_sft_pass2_local190_dual_v1": "checkpoint-optstep-000054",
    "t5gemma2_4b4b_typed_direct_rs_sft_pass3_c001_c002_v1": "checkpoint-optstep-000006",
    "t5gemma2_4b4b_typed_fold_gold_replay_v2": "checkpoint-optstep-000058",
    "t5gemma2_4b4b_typed_fold_rs_sft_union_v1": "checkpoint-optstep-000058",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


os.makedirs(DESTINATION, exist_ok=True)
os.makedirs(LATEST_DESTINATION, exist_ok=True)
sources = []
for family in sorted(os.listdir(SOURCE)):
    family_path = os.path.join(SOURCE, family)
    if not family.startswith("t5gemma2_4b4b_") or not os.path.isdir(family_path):
        continue
    for checkpoint in sorted(os.listdir(family_path)):
        checkpoint_path = os.path.join(family_path, checkpoint)
        if not checkpoint.startswith("checkpoint-optstep-") or not os.path.isdir(checkpoint_path):
            continue
        adapter_dir = os.path.join(checkpoint_path, "adapter")
        weight = os.path.join(adapter_dir, "adapter_model.safetensors")
        config = os.path.join(adapter_dir, "adapter_config.json")
        if not (os.path.isfile(weight) and os.path.isfile(config)):
            continue
        sources.extend([(family, checkpoint, weight), (family, checkpoint, config)])

if len(sources) != 90:
    raise SystemExit(f"expected 90 adapter files, found {len(sources)}")

records = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    digests = list(executor.map(lambda item: sha256(item[2]), sources))

for (family, checkpoint, source), digest in zip(sources, digests):
    repo_path = os.path.join(family, checkpoint, os.path.basename(source))
    destination = os.path.join(DESTINATION, repo_path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if not os.path.exists(destination):
        os.link(source, destination)
    if os.path.getsize(destination) != os.path.getsize(source):
        raise SystemExit(f"hardlink size mismatch: {destination}")
    records.append(
        {
            "base_model": "google/t5gemma-2-4b-4b",
            "checkpoint": checkpoint,
            "experiment_family": family,
            "repo_path": repo_path.replace(os.sep, "/"),
            "sha256": digest,
            "size": os.path.getsize(source),
            "source_path": os.path.relpath(source, "/workspace").replace(os.sep, "/"),
            "promotion_note": (
                "non-promoted research pilot"
                if family == "t5gemma2_4b4b_typed_c2_verpo_pilot150_v1"
                else "checkpoint name is not a quality or promotion claim"
            ),
        }
    )
    if LATEST.get(family) == checkpoint:
        latest_destination = os.path.join(LATEST_DESTINATION, repo_path)
        os.makedirs(os.path.dirname(latest_destination), exist_ok=True)
        if not os.path.exists(latest_destination):
            os.link(source, latest_destination)

notice_source = "/tmp/HF_MODIFIED_NOTICE.md"
for root in (DESTINATION, LATEST_DESTINATION):
    for family in sorted(LATEST if root == LATEST_DESTINATION else {r["experiment_family"] for r in records}):
        family_dir = os.path.join(root, family)
        if not os.path.isdir(family_dir):
            continue
        for checkpoint in os.listdir(family_dir):
            checkpoint_dir = os.path.join(family_dir, checkpoint)
            if os.path.isdir(checkpoint_dir):
                shutil.copy2(notice_source, os.path.join(checkpoint_dir, "MODIFIED_NOTICE.md"))

for root, selected in (
    (DESTINATION, records),
    (
        LATEST_DESTINATION,
        [r for r in records if LATEST.get(r["experiment_family"]) == r["checkpoint"]],
    ),
):
    with open(os.path.join(root, "manifest.jsonl"), "w", encoding="utf-8", newline="\n") as output:
        for record in sorted(selected, key=lambda item: item["repo_path"]):
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    with open(os.path.join(root, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as output:
        for record in sorted(selected, key=lambda item: item["repo_path"]):
            output.write(f"{record['sha256']}  {record['repo_path']}\n")

print(json.dumps({"all_files": len(records), "all_bytes": sum(r["size"] for r in records), "latest_files": sum(LATEST.get(r["experiment_family"]) == r["checkpoint"] for r in records), "latest_bytes": sum(r["size"] for r in records if LATEST.get(r["experiment_family"]) == r["checkpoint"])}, sort_keys=True))
