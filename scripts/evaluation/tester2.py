import json
import os
import statistics
import csv
import time
import random
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from codebleu import CodeBLEUCalculator

# ── Configuration ──────────────────────────────────────────────────────────
API_KEY = "sk-82kl3oqtxwjt4x"
API_BASE_URL = "http://127.0.0.1:18000/v1/chat/completions"

# Fetch available model from the endpoint
def get_model_name():
    """Auto-detect model name from /v1/models endpoint."""
    try:
        resp = requests.get(
            "http://127.0.0.1:18000/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            name = models[0]["id"]
            print(f"Auto-detected model: {name}")
            return name
    except Exception as e:
        print(f"Could not auto-detect model: {e}")
    return "default"

MODEL = get_model_name()

BATCH_SIZE = 4          # keep low for single-GPU vLLM to avoid proxy timeouts
MAX_TOKENS = 8192
TEMPERATURE = 0.2
TOP_P = 0.99

# ── Retry configuration ───────────────────────────────────────────────────
API_MAX_RETRIES = 5
API_RETRY_BASE_DELAY = 2
API_RETRY_MAX_DELAY = 120
ENTRY_MAX_RETRIES = 3

# ── Resume support ─────────────────────────────────────────────────────────
RESULTS_CACHE = "results/cache/eval_results_cache_decompiler_v5.jsonl"
FAILED_LOG = "results/cache/eval_failed_entries_decompiler_v5.jsonl"

# ── Evaluators & score storage ─────────────────────────────────────────────
dart_eval = CodeBLEUCalculator('dart')
swift_eval = CodeBLEUCalculator('swift')
dart_scores = []
swift_scores = []
scores = []

_file_lock = threading.Lock()

# ── Prompt template ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a reverse engineering expert with advanced knowledge in assembly "
    "and multiple programming languages. You decompile assembly into idiomatic, "
    "clear source code."
)

USER_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the task and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
Please convert the following assembly code to idiomatic and clear {lang} code.

### Assembly:
{assembly}

### Response:
"""


def entry_hash(entry: dict) -> str:
    raw = json.dumps({"lang": entry["language"], "asm": entry["assembly"]}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_completed() -> dict[str, dict]:
    completed = {}
    cache_path = Path(RESULTS_CACHE)
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    completed[rec["id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def save_result(entry_id: str, lang: str, score: float):
    with _file_lock:
        with open(RESULTS_CACHE, "a") as f:
            f.write(json.dumps({"id": entry_id, "lang": lang, "score": score}) + "\n")


def save_failed(entry_id: str, entry: dict, error: str):
    with _file_lock:
        with open(FAILED_LOG, "a") as f:
            f.write(json.dumps({
                "id": entry_id,
                "lang": entry.get("language", "?"),
                "error": str(error),
            }) + "\n")


def call_api(lang: str, assembly: str) -> str:
    """Call the OpenAI-compatible API with exponential backoff + jitter."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(lang=lang, assembly=assembly),
            },
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    last_exception = None

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                API_BASE_URL, headers=headers, json=payload, timeout=600
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = min(float(retry_after), API_RETRY_MAX_DELAY)
                else:
                    wait = min(API_RETRY_BASE_DELAY * (2 ** (attempt - 1)), API_RETRY_MAX_DELAY)
                wait += random.uniform(0, 1)
                print(f"  [attempt {attempt}/{API_MAX_RETRIES}] Rate limited, waiting {wait:.1f}s")
                time.sleep(wait)
                last_exception = requests.exceptions.HTTPError(f"429 Rate Limited")
                continue

            if resp.status_code == 400:
                print(f"  [DEBUG 400] Response body: {resp.text[:500]}")
            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data or not data["choices"]:
                raise ValueError(f"Malformed API response: missing choices. Keys: {list(data.keys())}")

            content = data["choices"][0].get("message", {}).get("content")
            if not content:
                raise ValueError("Empty content in API response")

            return content

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            print(f"  [attempt {attempt}/{API_MAX_RETRIES}] HTTP {status}: {e}")
            last_exception = e
            if isinstance(status, int) and status >= 500:
                wait = min(API_RETRY_BASE_DELAY * (2 ** (attempt - 1)), API_RETRY_MAX_DELAY)
                wait += random.uniform(0, 1)
                time.sleep(wait)
            elif isinstance(status, int) and 400 <= status < 500 and status != 429:
                raise

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"  [attempt {attempt}/{API_MAX_RETRIES}] Network error: {e}")
            last_exception = e
            wait = min(API_RETRY_BASE_DELAY * (2 ** (attempt - 1)), API_RETRY_MAX_DELAY)
            wait += random.uniform(0, 1)
            time.sleep(wait)

        except ValueError as e:
            print(f"  [attempt {attempt}/{API_MAX_RETRIES}] Response error: {e}")
            last_exception = e
            time.sleep(API_RETRY_BASE_DELAY)

    raise RuntimeError(f"API call failed after {API_MAX_RETRIES} retries: {last_exception}")


def extract_code(response: str, lang: str) -> str:
    try:
        marker = f"```{lang.lower()}"
        if marker in response:
            return response.split(marker)[1].split("```")[0].strip()
        if "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()
    except Exception as e:
        print(f"Extraction error: {e}")
        return response.strip()


def evaluate_entry(idx: int, total: int, entry: dict) -> dict:
    lang = entry["language"]
    source = entry["source"]
    assembly = entry["assembly"]
    eid = entry_hash(entry)

    last_error = None

    for entry_attempt in range(1, ENTRY_MAX_RETRIES + 1):
        try:
            if entry_attempt > 1:
                print(f"  ↻ Entry retry {entry_attempt}/{ENTRY_MAX_RETRIES} for {idx}/{total}")

            print(f"Processing {idx}/{total} — {lang} [{eid}]")
            response = call_api(lang, assembly)
            code = extract_code(response, lang)

            if lang.lower() == "dart":
                score = dart_eval.compute_codebleu(source, code)["codebleu"]
            elif lang.lower() == "swift":
                score = swift_eval.compute_codebleu(source, code)["codebleu"]
            else:
                score = 0.0

            print(f"  ✓ {lang} score: {score:.4f}")
            save_result(eid, lang, score)
            return {"lang": lang, "score": score, "id": eid}

        except Exception as e:
            last_error = e
            print(f"  ✗ Entry {idx} attempt {entry_attempt} failed: {e}")
            if entry_attempt < ENTRY_MAX_RETRIES:
                time.sleep(API_RETRY_BASE_DELAY * entry_attempt)

    save_failed(eid, entry, last_error)
    raise RuntimeError(f"Entry {idx} [{eid}] failed after {ENTRY_MAX_RETRIES} attempts: {last_error}")


def write_stats(lang_scores: list[float], lang_name: str):
    if not lang_scores:
        return
    avg = statistics.mean(lang_scores)
    stdv = statistics.stdev(lang_scores) if len(lang_scores) > 1 else 0.0
    header = ["Min", "Max", "Average", "Standard_Deviation"]
    row = [min(lang_scores), max(lang_scores), avg, stdv]
    fname = f"{lang_name.lower()}_statistics_decompiler_v5.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)
    print(f"{lang_name} stats written to {fname}: avg={avg:.4f}, std={stdv:.4f}")


def main():
    with open("data/datasets/test-set2.jsonl", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f if line.strip()]

    total = len(data_lines)
    print(f"Loaded {total} test entries. Model: {MODEL}")

    completed = load_completed()
    if completed:
        print(f"Resuming — {len(completed)} entries already completed, skipping them.")

    for rec in completed.values():
        s = rec["score"]
        scores.append(s)
        if rec["lang"].lower() == "dart":
            dart_scores.append(s)
        elif rec["lang"].lower() == "swift":
            swift_scores.append(s)

    remaining = []
    for i, entry in enumerate(data_lines):
        eid = entry_hash(entry)
        if eid not in completed:
            remaining.append((i + 1, entry))

    if not remaining:
        print("All entries already completed!")
    else:
        print(f"Processing {len(remaining)} remaining entries...")

        failed_count = 0

        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as pool:
            futures = {
                pool.submit(evaluate_entry, idx, total, entry): (idx, entry)
                for idx, entry in remaining
            }

            for future in as_completed(futures):
                idx, entry = futures[future]
                try:
                    result = future.result()
                    lang = result["lang"]
                    score = result["score"]
                    scores.append(score)
                    if lang.lower() == "dart":
                        dart_scores.append(score)
                    elif lang.lower() == "swift":
                        swift_scores.append(score)
                except Exception as e:
                    failed_count += 1
                    print(f"PERMANENTLY FAILED entry {idx} ({entry.get('language', '?')}): {e}")

        if failed_count:
            print(f"\n⚠ {failed_count} entries failed — see {FAILED_LOG}")

    write_stats(dart_scores, "Dart")
    write_stats(swift_scores, "Swift")

    if scores:
        ok = len(scores)
        print(f"\nOverall: avg={statistics.mean(scores):.4f}, n={ok}/{total}")
    else:
        print("\nNo scores collected.")


if __name__ == "__main__":
    main()

