import collections
import os


root = "/workspace/artifacts"
parents = collections.defaultdict(
    lambda: {"checkpoint_dirs": 0, "files": 0, "bytes": 0, "names": []}
)

for directory, child_dirs, child_files in os.walk(root):
    basename = os.path.basename(directory)
    if not basename.startswith("checkpoint"):
        continue

    relative = os.path.relpath(directory, root)
    parent = relative.split(os.sep)[0]
    file_count = 0
    byte_count = 0
    for checkpoint_directory, _, checkpoint_files in os.walk(directory):
        for filename in checkpoint_files:
            path = os.path.join(checkpoint_directory, filename)
            if os.path.islink(path):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            file_count += 1
            byte_count += size

    info = parents[parent]
    info["checkpoint_dirs"] += 1
    info["files"] += file_count
    info["bytes"] += byte_count
    info["names"].append(basename)
    child_dirs[:] = []

print(f"MODEL_PARENTS\t{len(parents)}")
for parent, info in sorted(parents.items()):
    print(
        "\t".join(
            [
                str(info["bytes"]),
                str(info["files"]),
                str(info["checkpoint_dirs"]),
                parent,
                ",".join(sorted(info["names"])),
            ]
        )
    )
