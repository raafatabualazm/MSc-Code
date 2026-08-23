import json
import os
import re
import struct
import zipfile


ROOT = "/workspace/artifacts"
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".toml"}
CREDENTIAL_PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "openai_style_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "private_key_block": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}
PRIVATE_REFERENCE_PATTERNS = {
    "holdback_reference": re.compile(rb"holdback", re.IGNORECASE),
    "private_reference": re.compile(rb"private", re.IGNORECASE),
    "secret_reference": re.compile(rb"secret", re.IGNORECASE),
}


def is_checkpoint_path(path: str) -> bool:
    relative = os.path.relpath(path, ROOT)
    return any(part.startswith("checkpoint") for part in relative.split(os.sep)[:-1])


def scan_bytes(path: str, payload: bytes, credential_hits, reference_hits) -> None:
    for category, pattern in CREDENTIAL_PATTERNS.items():
        if pattern.search(payload):
            credential_hits.append({"category": category, "path": os.path.relpath(path, ROOT)})
    for category, pattern in PRIVATE_REFERENCE_PATTERNS.items():
        if pattern.search(payload):
            reference_hits.append({"category": category, "path": os.path.relpath(path, ROOT)})


files = []
credential_hits = []
reference_hits = []
checkpoint_dirs = set()

for directory, _, filenames in os.walk(ROOT):
    for filename in filenames:
        path = os.path.join(directory, filename)
        if not is_checkpoint_path(path) or os.path.islink(path):
            continue
        relative = os.path.relpath(path, ROOT)
        checkpoint_component = next(
            part for part in relative.split(os.sep) if part.startswith("checkpoint")
        )
        parent = relative.split(os.sep)[0]
        checkpoint_dirs.add(os.path.join(parent, checkpoint_component))
        size = os.path.getsize(path)
        files.append({"path": relative, "size": size})

        suffix = os.path.splitext(filename)[1].lower()
        if suffix in TEXT_SUFFIXES:
            with open(path, "rb") as source:
                scan_bytes(path, source.read(), credential_hits, reference_hits)
        elif suffix == ".safetensors":
            with open(path, "rb") as source:
                header_length_bytes = source.read(8)
                if len(header_length_bytes) == 8:
                    header_length = struct.unpack("<Q", header_length_bytes)[0]
                    if header_length <= 128 * 1024 * 1024:
                        scan_bytes(path, source.read(header_length), credential_hits, reference_hits)
        elif suffix == ".pt" and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    member_name = member.filename
                    basename = member_name.rsplit("/", 1)[-1]
                    is_tensor_storage = "/data/" in member_name and basename.isdigit()
                    if is_tensor_storage or member.file_size > 32 * 1024 * 1024:
                        continue
                    scan_bytes(path, archive.read(member), credential_hits, reference_hits)

files.sort(key=lambda record: record["path"])
credential_hits.sort(key=lambda record: (record["path"], record["category"]))
reference_hits.sort(key=lambda record: (record["path"], record["category"]))

print(
    json.dumps(
        {
            "checkpoint_directories": len(checkpoint_dirs),
            "files": len(files),
            "bytes": sum(record["size"] for record in files),
            "credential_hits": credential_hits,
            "private_reference_hits": reference_hits,
            "file_name_private_markers": [
                record["path"]
                for record in files
                if any(marker in record["path"].lower() for marker in ("holdback", "private", "secret"))
            ],
        },
        sort_keys=True,
    )
)
