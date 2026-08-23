import hashlib
import json
import os
import sys
import tempfile

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "raafatabualazm/t5gemma2-verpo-artifacts"
LOCAL_ROOT = "/workspace/hf_public_t5gemma2_verpo_artifacts_v1"

token = sys.stdin.readline().strip()
if not token:
    raise SystemExit("missing token on stdin")

expected_paths = set()
expected_sizes = {}
for directory, _, filenames in os.walk(LOCAL_ROOT):
    if "/.cache/" in directory:
        continue
    for filename in filenames:
        path = os.path.join(directory, filename)
        relative = os.path.relpath(path, LOCAL_ROOT).replace(os.sep, "/")
        expected_paths.add(relative)
        expected_sizes[relative] = os.path.getsize(path)

with open(os.path.join(LOCAL_ROOT, "manifest.jsonl"), encoding="utf-8") as source:
    manifest = [json.loads(line) for line in source if line.strip()]
expected_weight_hashes = {
    record["repo_path"]: record["sha256"]
    for record in manifest
    if record["repo_path"].endswith("adapter_model.safetensors")
}

api = HfApi(token=token)
info = api.model_info(REPO_ID, expand=["siblings"])
tree = list(api.list_repo_tree(REPO_ID, repo_type="model", recursive=True, expand=True))
remote_files = {item.path: item for item in tree if hasattr(item, "size")}
remote_paths = set(remote_files)
allowed_extra = {".gitattributes"}

missing = sorted(expected_paths - remote_paths)
extra = sorted(remote_paths - expected_paths - allowed_extra)
size_mismatches = [
    {
        "path": path,
        "expected": expected_sizes[path],
        "remote": remote_files[path].size,
    }
    for path in sorted(expected_paths & remote_paths)
    if remote_files[path].size != expected_sizes[path]
]

hash_mismatches = []
missing_remote_hash = []
for path, expected_hash in sorted(expected_weight_hashes.items()):
    item = remote_files.get(path)
    lfs = getattr(item, "lfs", None) if item is not None else None
    if isinstance(lfs, dict):
        remote_hash = lfs.get("sha256") or lfs.get("oid")
    else:
        remote_hash = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if remote_hash and remote_hash.startswith("sha256:"):
        remote_hash = remote_hash.split(":", 1)[1]
    if not remote_hash:
        missing_remote_hash.append(path)
    elif remote_hash != expected_hash:
        hash_mismatches.append(
            {"path": path, "expected": expected_hash, "remote": remote_hash}
        )

with tempfile.TemporaryDirectory(prefix="hf-verify-") as temporary:
    downloaded_manifest = hf_hub_download(
        repo_id=REPO_ID,
        filename="manifest.jsonl",
        repo_type="model",
        token=token,
        local_dir=temporary,
        force_download=True,
    )
    local_manifest_hash = hashlib.sha256(
        open(os.path.join(LOCAL_ROOT, "manifest.jsonl"), "rb").read()
    ).hexdigest()
    hub_manifest_hash = hashlib.sha256(open(downloaded_manifest, "rb").read()).hexdigest()

result = {
    "repo_id": REPO_ID,
    "repo_url": f"https://huggingface.co/{REPO_ID}",
    "private": info.private,
    "revision": info.sha,
    "expected_files": len(expected_paths),
    "remote_files": len(remote_paths),
    "expected_bytes": sum(expected_sizes.values()),
    "matched_expected_bytes": sum(
        remote_files[path].size for path in expected_paths & remote_paths
    ),
    "missing": missing,
    "extra": extra,
    "size_mismatches": size_mismatches,
    "weight_hashes_expected": len(expected_weight_hashes),
    "weight_hashes_verified": len(expected_weight_hashes)
    - len(hash_mismatches)
    - len(missing_remote_hash),
    "missing_remote_weight_hashes": missing_remote_hash,
    "weight_hash_mismatches": hash_mismatches,
    "local_manifest_sha256": local_manifest_hash,
    "hub_manifest_sha256": hub_manifest_hash,
    "manifest_match": local_manifest_hash == hub_manifest_hash,
}
print(json.dumps(result, sort_keys=True))

if (
    info.private
    or missing
    or extra
    or size_mismatches
    or hash_mismatches
    or missing_remote_hash
    or local_manifest_hash != hub_manifest_hash
):
    raise SystemExit("Hub verification failed")
