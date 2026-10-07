"""Read-only Anthropic transport probe; never prints credentials."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import httpx
import anthropic


def load_key() -> str:
    value = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if value:
        return value
    for raw in Path("/workspace/Anthropic.env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("ANTHROPIC_API_KEY is missing")


def main() -> None:
    environment_key = os.environ.get("ANTHROPIC_API_KEY")
    if environment_key is not None:
        print(
            "environment_key_length",
            len(environment_key),
            "stripped_length",
            len(environment_key.strip()),
            "has_carriage_return",
            environment_key.endswith("\r"),
        )
    addresses = socket.getaddrinfo(
        "api.anthropic.com", 443, type=socket.SOCK_STREAM
    )
    print("dns_families", sorted({entry[0] for entry in addresses}))
    response = httpx.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": load_key(),
            "anthropic-version": "2023-06-01",
        },
        timeout=30.0,
    )
    try:
        body = response.json()
    except json.JSONDecodeError:
        body = {}
    print(
        "http_status",
        response.status_code,
        "body_type",
        body.get("type"),
        "error_type",
        (body.get("error") or {}).get("type"),
        "model_count",
        len(body.get("data") or []),
    )
    client = anthropic.Anthropic(
        api_key=load_key(),
        base_url="https://api.anthropic.com",
        max_retries=0,
        timeout=30.0,
    )
    try:
        models = client.models.list(limit=5)
        print("sdk_models_ok", len(models.data))
    except Exception as exc:
        print("sdk_models_error", exception_chain(exc))
    try:
        count = client.messages.count_tokens(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Return only: OK"}],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        print("sdk_count_ok", count.input_tokens)
    except Exception as exc:
        print("sdk_count_error", exception_chain(exc))
    prompt_path = Path(
        "/workspace/artifacts/frontier_ceiling_two_enrichments/runs/"
        "anthropic_sonnet5_batch_screen_k2_w1/opus/prompts.jsonl"
    )
    if prompt_path.is_file():
        first = json.loads(prompt_path.read_text(encoding="utf-8").splitlines()[0])
        prompt_messages = first["messages"]
        system = prompt_messages[0]["content"]
        messages = prompt_messages[1:]
        payload = {
            "model": "claude-sonnet-5",
            "system": system,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        print(
            "sealed_probe_task",
            first["task_id"],
            "serialized_bytes",
            len(json.dumps(payload).encode("utf-8")),
        )
        try:
            count = client.messages.count_tokens(**payload)
            print("sdk_sealed_count_ok", count.input_tokens)
        except Exception as exc:
            print("sdk_sealed_count_error", exception_chain(exc))
        try:
            direct = httpx.post(
                "https://api.anthropic.com/v1/messages/count_tokens",
                headers={
                    "x-api-key": load_key(),
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            direct_body = direct.json()
            print(
                "direct_sealed_count",
                direct.status_code,
                direct_body.get("type"),
                (direct_body.get("error") or {}).get("type"),
                direct_body.get("input_tokens"),
            )
        except Exception as exc:
            print("direct_sealed_count_error", exception_chain(exc))


def exception_chain(exc: BaseException) -> list[tuple[str, str]]:
    seen: set[int] = set()
    result: list[tuple[str, str]] = []
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append((type(current).__name__, str(current)[:500]))
        current = current.__cause__ or current.__context__
    return result


if __name__ == "__main__":
    main()
