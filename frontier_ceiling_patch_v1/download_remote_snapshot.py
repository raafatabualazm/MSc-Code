#!/usr/bin/env python3
"""Resumably download and SHA-256 verify one remote snapshot archive."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import paramiko


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def key_fingerprint(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=args.host,
        port=args.port,
        username=args.username,
        allow_agent=True,
        look_for_keys=True,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    transport = client.get_transport()
    if transport is None:
        client.close()
        raise RuntimeError("SSH transport is unavailable after connect")
    actual = key_fingerprint(transport.get_remote_server_key())
    if actual != args.host_key_sha256:
        client.close()
        raise RuntimeError(
            f"remote host-key fingerprint mismatch: expected "
            f"{args.host_key_sha256}, got {actual}"
        )
    transport.set_keepalive(30)
    return client


def read_ready(
    args: argparse.Namespace,
    progress_path: Path,
) -> tuple[dict[str, str], bytes]:
    deadline = time.monotonic() + args.ready_timeout_seconds
    while True:
        client: paramiko.SSHClient | None = None
        try:
            client = connect(args)
            sftp = client.open_sftp()
            with sftp.open(args.remote_ready, "rb") as handle:
                raw = handle.read()
            values: dict[str, str] = {}
            for line in raw.decode("utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator:
                    values[key] = value
            required = {"archive", "archive_bytes", "archive_sha256", "completed_at"}
            if not required.issubset(values):
                raise RuntimeError("remote ready record is incomplete")
            if values["archive"] != args.remote_archive:
                raise RuntimeError("remote ready record points to another archive")
            if len(values["archive_sha256"]) != 64:
                raise RuntimeError("remote ready record has an invalid SHA-256")
            return values, raw
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for remote snapshot readiness")
            atomic_json(
                progress_path,
                {
                    "schema": "resumable-remote-snapshot-download-v1",
                    "status": "waiting_for_remote_snapshot",
                    "updated_at": utc_now(),
                    "host": args.host,
                    "remote_ready": args.remote_ready,
                },
            )
            print(
                f"{utc_now()} waiting for {args.remote_ready}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
        finally:
            if client is not None:
                client.close()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(args: argparse.Namespace) -> int:
    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    progress_path = local_dir / "download_progress.json"
    ready_values, ready_raw = read_ready(args, progress_path)
    expected_bytes = int(ready_values["archive_bytes"])
    expected_sha256 = ready_values["archive_sha256"].lower()
    archive_name = Path(args.remote_archive).name
    final_path = local_dir / archive_name
    part_path = local_dir / (archive_name + ".part")
    ready_path = local_dir / (Path(args.remote_ready).name)
    ready_path.write_bytes(ready_raw)

    if final_path.is_file():
        if final_path.stat().st_size != expected_bytes:
            raise RuntimeError(
                f"existing final archive has the wrong size: {final_path}"
            )
        actual = hash_file(final_path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"existing final archive has the wrong SHA-256: {final_path}"
            )
        atomic_json(
            progress_path,
            {
                "schema": "resumable-remote-snapshot-download-v1",
                "status": "complete",
                "updated_at": utc_now(),
                "archive": str(final_path),
                "archive_bytes": expected_bytes,
                "archive_sha256": actual,
                "resumed": True,
            },
        )
        return 0

    if part_path.exists() and part_path.stat().st_size > expected_bytes:
        raise RuntimeError(f"partial archive exceeds remote size: {part_path}")

    started_at = utc_now()
    last_report_at = 0.0
    last_report_bytes = part_path.stat().st_size if part_path.exists() else 0
    failures = 0
    while True:
        offset = part_path.stat().st_size if part_path.exists() else 0
        if offset == expected_bytes:
            break
        client: paramiko.SSHClient | None = None
        try:
            client = connect(args)
            sftp = client.open_sftp()
            remote_stat = sftp.stat(args.remote_archive)
            if remote_stat.st_size != expected_bytes:
                raise RuntimeError("remote archive size changed after ready record")
            with sftp.open(args.remote_archive, "rb") as remote:
                remote.seek(offset)
                try:
                    remote.prefetch(
                        file_size=expected_bytes,
                        max_concurrent_requests=64,
                    )
                except TypeError:
                    remote.prefetch(file_size=expected_bytes)
                with part_path.open("ab", buffering=0) as local:
                    while offset < expected_bytes:
                        chunk = remote.read(
                            min(args.chunk_bytes, expected_bytes - offset)
                        )
                        if not chunk:
                            raise EOFError("remote archive ended before declared size")
                        local.write(chunk)
                        offset += len(chunk)
                        now = time.monotonic()
                        if (
                            now - last_report_at >= args.report_seconds
                            or offset - last_report_bytes >= 256 * 1024 * 1024
                        ):
                            percent = 100.0 * offset / expected_bytes
                            value = {
                                "schema": "resumable-remote-snapshot-download-v1",
                                "status": "downloading",
                                "started_at": started_at,
                                "updated_at": utc_now(),
                                "host": args.host,
                                "remote_archive": args.remote_archive,
                                "local_partial": str(part_path),
                                "downloaded_bytes": offset,
                                "expected_bytes": expected_bytes,
                                "percent": percent,
                                "connection_failures": failures,
                            }
                            atomic_json(progress_path, value)
                            print(
                                f"{value['updated_at']} "
                                f"{offset}/{expected_bytes} ({percent:.2f}%)",
                                flush=True,
                            )
                            last_report_at = now
                            last_report_bytes = offset
            failures = 0
        except Exception as exc:
            failures += 1
            print(
                f"{utc_now()} transfer interruption {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if failures > args.max_reconnects:
                raise
            time.sleep(min(30, 2 + failures))
        finally:
            if client is not None:
                client.close()

    atomic_json(
        progress_path,
        {
            "schema": "resumable-remote-snapshot-download-v1",
            "status": "verifying_sha256",
            "started_at": started_at,
            "updated_at": utc_now(),
            "local_partial": str(part_path),
            "archive_bytes": expected_bytes,
            "expected_sha256": expected_sha256,
        },
    )
    actual_sha256 = hash_file(part_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"downloaded archive SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    os.replace(part_path, final_path)
    atomic_json(
        progress_path,
        {
            "schema": "resumable-remote-snapshot-download-v1",
            "status": "complete",
            "started_at": started_at,
            "completed_at": utc_now(),
            "archive": str(final_path),
            "archive_bytes": expected_bytes,
            "archive_sha256": actual_sha256,
            "ready_record": str(ready_path),
        },
    )
    print(
        f"{utc_now()} SNAPSHOT_DOWNLOAD_COMPLETE "
        f"bytes={expected_bytes} sha256={actual_sha256} path={final_path}",
        flush=True,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="root")
    parser.add_argument("--host-key-sha256", required=True)
    parser.add_argument("--remote-archive", required=True)
    parser.add_argument("--remote-ready", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--ready-timeout-seconds", type=int, default=21_600)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--chunk-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--report-seconds", type=int, default=10)
    parser.add_argument("--max-reconnects", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(download(parse_args()))
