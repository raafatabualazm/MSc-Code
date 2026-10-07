#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import json
import os
import stat

root = "/workspace"
output = "/tmp/workspace_nocheckpoints_20260804.layout.json"
directories = []
symlinks = []

for directory, child_dirs, child_files in os.walk(root, topdown=True, followlinks=False):
    kept_dirs = []
    for child in child_dirs:
        child_path = os.path.join(directory, child)
        if child.startswith("checkpoint"):
            continue
        if directory == root and child in {"secrets", "tmp", ".hf_home"}:
            continue
        if os.path.islink(child_path):
            symlinks.append(
                {
                    "path": child_path.lstrip("/"),
                    "target": os.readlink(child_path),
                }
            )
            continue
        kept_dirs.append(child)
    child_dirs[:] = kept_dirs

    metadata = os.lstat(directory)
    directories.append(
        {
            "path": directory.lstrip("/"),
            "mode": stat.S_IMODE(metadata.st_mode),
            "mtime_ns": metadata.st_mtime_ns,
        }
    )

    for filename in child_files:
        path = os.path.join(directory, filename)
        if os.path.islink(path):
            symlinks.append({"path": path.lstrip("/"), "target": os.readlink(path)})

directories.sort(key=lambda record: record["path"])
symlinks.sort(key=lambda record: record["path"])
with open(output, "w", encoding="utf-8", newline="\n") as destination:
    json.dump(
        {
            "schema": "workspace-portable-layout-v1",
            "root": root,
            "checkpoint_directories_excluded": True,
            "root_directories_excluded": [".hf_home", "secrets", "tmp"],
            "directories": directories,
            "symlinks": symlinks,
        },
        destination,
        sort_keys=True,
        separators=(",", ":"),
    )
    destination.write("\n")

print(f"LAYOUT_DIRECTORIES {len(directories)}")
print(f"LAYOUT_SYMLINKS {len(symlinks)}")
PY

sha256sum /tmp/workspace_nocheckpoints_20260804.layout.json
