#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="/workspace"
readonly STAMP="20260804"
readonly MANIFEST="/tmp/workspace_nocheckpoints_${STAMP}.manifest.jsonl"
readonly FILELIST="/tmp/workspace_nocheckpoints_${STAMP}.files0"
readonly ARCHIVE="/tmp/workspace_nocheckpoints_${STAMP}.tar.zst"

python3 - <<'PY'
import hashlib
import json
import os
import stat

root = "/workspace"
manifest = "/tmp/workspace_nocheckpoints_20260804.manifest.jsonl"
filelist = "/tmp/workspace_nocheckpoints_20260804.files0"
records = []

for directory, child_dirs, child_files in os.walk(root, topdown=True, followlinks=False):
    kept_dirs = []
    for child in child_dirs:
        if child.startswith("checkpoint"):
            continue
        if directory == root and child in {"secrets", "tmp", ".hf_home"}:
            continue
        kept_dirs.append(child)
    child_dirs[:] = kept_dirs

    for filename in child_files:
        path = os.path.join(directory, filename)
        try:
            metadata = os.lstat(path)
        except OSError:
            continue
        if not stat.S_ISREG(metadata.st_mode):
            continue

        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)

        records.append(
            {
                "path": path.lstrip("/"),
                "size": metadata.st_size,
                "sha256": digest.hexdigest(),
                "mtime_ns": metadata.st_mtime_ns,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        )

records.sort(key=lambda record: record["path"])
with open(manifest, "w", encoding="utf-8", newline="\n") as manifest_file:
    with open(filelist, "wb") as filelist_file:
        for record in records:
            manifest_file.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
            filelist_file.write(record["path"].encode("utf-8") + b"\0")

print(f"MANIFEST_FILES {len(records)}")
print(f"MANIFEST_BYTES {sum(record['size'] for record in records)}")
PY

rm -f -- "$ARCHIVE"
tar -C / --null --no-recursion --files-from="$FILELIST" -cf - \
    | zstd -T0 -3 -f -o "$ARCHIVE"

sha256sum "$MANIFEST" "$FILELIST" "$ARCHIVE"
stat -c 'ARCHIVE_BYTES %s' "$ARCHIVE"
