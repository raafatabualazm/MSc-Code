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
OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Change this to whatever model you're targeting on OpenRouter
# e.g. "qwen/qwen3-4b", "meta-llama/llama-3-70b-instruct", "deepseek/deepseek-v3", etc.
MODEL = "qwen/qwen3-8b"

BATCH_SIZE = 8          # concurrent API requests
MAX_TOKENS = 8192
TEMPERATURE = 0.2
TOP_P = 0.99

# ── Retry configuration ───────────────────────────────────────────────────
API_MAX_RETRIES = 5         # retries for a single API call
API_RETRY_BASE_DELAY = 2    # base seconds for exponential backoff
API_RETRY_MAX_DELAY = 120   # cap backoff at 2 minutes
ENTRY_MAX_RETRIES = 3       # retries for the full evaluate pipeline per entry

# ── Resume support ─────────────────────────────────────────────────────────
RESULTS_CACHE = "eval_results_cache_qwen3_8b.jsonl"   # completed results saved here
FAILED_LOG = "eval_failed_entries_qwen3_8b.jsonl"     # entries that exhausted all retries

# ── Evaluators & score storage ─────────────────────────────────────────────
dart_eval = CodeBLEUCalculator('dart')
swift_eval = CodeBLEUCalculator('swift')
dart_scores = []
swift_scores = []
scores = []

# Thread-safe file writing
_file_lock = threading.Lock()

# ── Prompt template (same as original) ─────────────────────────────────────
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
    """Stable hash for an entry so we can skip already-completed ones on resume."""
    raw = json.dumps({"lang": entry["language"], "asm": entry["assembly"]}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_completed() -> dict[str, dict]:
    """Load previously completed results from cache file."""
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
    """Append a completed result to the cache (thread-safe)."""
    with _file_lock:
        with open(RESULTS_CACHE, "a") as f:
            f.write(json.dumps({"id": entry_id, "lang": lang, "score": score}) + "\n")


def save_failed(entry_id: str, entry: dict, error: str):
    """Log a permanently failed entry (thread-safe)."""
    with _file_lock:
        with open(FAILED_LOG, "a") as f:
            f.write(json.dumps({
                "id": entry_id,
                "lang": entry.get("language", "?"),
                "error": str(error),
            }) + "\n")


def call_openrouter(lang: str, assembly: str) -> str:
    """Call the OpenRouter API with exponential backoff + jitter."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/your-repo",
        "X-Title": "Decompiler Eval",
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
                OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=300
            )

            # Handle rate-limit Retry-After header
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait = min(float(retry_after), API_RETRY_MAX_DELAY)
                else:
                    wait = min(API_RETRY_BASE_DELAY * (2 ** (attempt - 1)), API_RETRY_MAX_DELAY)
                wait += random.uniform(0, 1)  # jitter
                print(f"  [attempt {attempt}/{API_MAX_RETRIES}] Rate limited, waiting {wait:.1f}s")
                time.sleep(wait)
                last_exception = requests.exceptions.HTTPError(f"429 Rate Limited")
                continue

            resp.raise_for_status()
            data = resp.json()

            # Validate response structure
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
                # Client errors (other than rate limit) — don't retry
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
    """Extract the code block from the model response."""
    try:
        marker = f"```{lang.lower()}"
        if marker in response:
            return response.split(marker)[1].split("```")[0].strip()

        # Fallback: try generic code fence
        if "```" in response:
            return response.split("```")[1].split("```")[0].strip()

        # No fences — return as-is
        return response.strip()
    except Exception as e:
        print(f"Extraction error: {e}")
        return response.strip()


def evaluate_entry(idx: int, total: int, entry: dict) -> dict:
    """Run inference + scoring with entry-level retry."""
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
            response = call_openrouter(lang, assembly)
            code = extract_code(response, lang)

            if lang.lower() == "dart":
                score = dart_eval.compute_codebleu(source, code)["codebleu"]
            elif lang.lower() == "swift":
                score = swift_eval.compute_codebleu(source, code)["codebleu"]
            else:
                score = 0.0

            print(f"  ✓ {lang} score: {score:.4f}")

            # Persist to cache for resume
            save_result(eid, lang, score)
            return {"lang": lang, "score": score, "id": eid}

        except Exception as e:
            last_error = e
            print(f"  ✗ Entry {idx} attempt {entry_attempt} failed: {e}")
            if entry_attempt < ENTRY_MAX_RETRIES:
                time.sleep(API_RETRY_BASE_DELAY * entry_attempt)

    # All entry retries exhausted
    save_failed(eid, entry, last_error)
    raise RuntimeError(f"Entry {idx} [{eid}] failed after {ENTRY_MAX_RETRIES} attempts: {last_error}")


def write_stats(lang_scores: list[float], lang_name: str):
    if not lang_scores:
        return
    avg = statistics.mean(lang_scores)
    stdv = statistics.stdev(lang_scores) if len(lang_scores) > 1 else 0.0
    header = ["Min", "Max", "Average", "Standard_Deviation"]
    row = [min(lang_scores), max(lang_scores), avg, stdv]
    fname = f"{lang_name.lower()}_statistics_openrouter_qwen3_8b.csv"
    with open(fname, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)
    print(f"{lang_name} stats written to {fname}: avg={avg:.4f}, std={stdv:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    with open("test-set2.jsonl", encoding="utf-8") as f:
        data_lines = [json.loads(line) for line in f if line.strip()]

    total = len(data_lines)
    print(f"Loaded {total} test entries. Model: {MODEL}")

    # ── Resume: load already-completed entries ─────────────────────────────
    completed = load_completed()
    if completed:
        print(f"Resuming — {len(completed)} entries already completed, skipping them.")

    # Restore scores from cache
    for rec in completed.values():
        s = rec["score"]
        scores.append(s)
        if rec["lang"].lower() == "dart":
            dart_scores.append(s)
        elif rec["lang"].lower() == "swift":
            swift_scores.append(s)

    # Filter out already-done entries
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

    # ── Write statistics ───────────────────────────────────────────────────
    write_stats(dart_scores, "Dart")
    write_stats(swift_scores, "Swift")

    if scores:
        ok = len(scores)
        print(f"\nOverall: avg={statistics.mean(scores):.4f}, n={ok}/{total}")
    else:
        print("\nNo scores collected.")


if __name__ == "__main__":
    main()